"""
Chandra OCR 2 engine — Vision-Language Model by Datalab (Qwen2-VL based).
Hỗ trợ layout detection + OCR + table extraction trong một model duy nhất.
Install: pip install "chandra-ocr[hf]"
Model:   datalab-to/chandra-ocr-2  (~7B params)
VRAM:    ~8GB (CUDA fp16)  hoặc  RAM (CPU, chậm hơn nhiều)
"""
from __future__ import annotations

import gc
import os
import re
import time
from html.parser import HTMLParser
from typing import ClassVar

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from config.settings import Settings
from ocrdoc.core.models import BoundingBox, PageResult, Table, TableCell, TextBlock
from ocrdoc.core.vram_manager import VRAMManager
from ocrdoc.engines.base import BaseEngine

# Chandra block label → TextBlock.block_type
_LABEL_TO_BLOCK_TYPE: dict[str, str] = {
    "Text": "paragraph",
    "List-Group": "paragraph",
    "Footnote": "paragraph",
    "Code-Block": "paragraph",
    "Section-Header": "heading",
    "Page-Header": "header",
    "Page-Footer": "footer",
    "Caption": "caption",
}
_TABLE_LABELS = frozenset({"Table"})


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    return " ".join(text.split())


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._current_cell = []
            self._in_cell = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._current_row.append(" ".join(self._current_cell).strip())
            self._in_cell = False
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._current_cell.append(data.strip())


class ChandraEngine(BaseEngine):
    """
    Chandra OCR 2 — thay thế Surya OCR cho tài liệu không có cấu trúc.
    Trả về layout có bbox + text + phát hiện bảng tự động.
    """

    name: ClassVar[str] = "chandra"

    def __init__(self, settings: Settings, vram_manager: VRAMManager) -> None:
        self._settings = settings
        self._vram = vram_manager
        self._manager = None
        self._ready = False

    # ── BaseEngine ────────────────────────────────────────────────────────

    # Nếu VRAM trống < ngưỡng này → tự động dùng 4-bit quantization
    _LOW_VRAM_THRESHOLD_MB: ClassVar[int] = 7000

    def warm_up(self) -> None:
        if self._ready:
            return
        self._vram.free_vram()

        try:
            from chandra.model import InferenceManager  # noqa: F401
        except ImportError:
            raise ImportError(
                'chandra-ocr chưa được cài. Chạy: pip install "chandra-ocr[hf]"'
            )

        if self._settings.paddle_device.lower() in ("gpu", "cuda"):
            os.environ["TORCH_DEVICE"] = "cuda"
        else:
            os.environ["TORCH_DEVICE"] = "cpu"

        # ── Kiểm tra VRAM → chọn chiến lược load ──────────────────────
        use_4bit = False
        try:
            import torch
            if torch.cuda.is_available():
                free_mb = torch.cuda.mem_get_info()[0] // (1024 ** 2)
                if free_mb < self._LOW_VRAM_THRESHOLD_MB:
                    try:
                        import bitsandbytes  # noqa: F401
                        use_4bit = True
                        logger.info(
                            f"ChandraEngine: VRAM thấp ({free_mb}MB < "
                            f"{self._LOW_VRAM_THRESHOLD_MB}MB). "
                            "Dùng 4-bit NF4 quantization để tiết kiệm VRAM."
                        )
                    except ImportError:
                        raise RuntimeError(
                            f"Chandra 7B cần ≥{self._LOW_VRAM_THRESHOLD_MB}MB VRAM "
                            f"(hiện còn {free_mb}MB). Để chạy với 4GB, cài:\n"
                            "    pip install bitsandbytes\n"
                            "Hoặc dùng Vintern 1B (mode=vintern, chỉ cần ~2GB VRAM)."
                        )
        except RuntimeError:
            raise
        except Exception:
            pass

        from chandra.model import InferenceManager
        if use_4bit:
            self._manager = self._load_with_4bit(InferenceManager)
        else:
            logger.info("ChandraEngine: đang load model (fp16, cần ~14GB VRAM)...")
            self._manager = InferenceManager(method="hf")

        self._ready = True
        logger.success("ChandraEngine: model sẵn sàng")

    @staticmethod
    def _patch_params4bit() -> None:
        """
        Patch bitsandbytes.Params4bit.__new__ để ignore kwarg '_is_hf_initialized'
        do transformers 5.x truyền vào nhưng bitsandbytes 0.49.x chưa accept.
        """
        try:
            from bitsandbytes.nn.modules import Params4bit
            _orig = Params4bit.__new__

            def _new(cls, *args, **kwargs):
                kwargs.pop("_is_hf_initialized", None)
                return _orig(cls, *args, **kwargs)

            Params4bit.__new__ = staticmethod(_new)
            logger.debug("ChandraEngine: đã patch Params4bit._is_hf_initialized")
        except Exception as e:
            logger.debug(f"ChandraEngine: patch Params4bit bỏ qua — {e}")

    @staticmethod
    def _load_with_4bit(InferenceManager):
        """
        Monkeypatch chandra.model.load_model để inject BitsAndBytesConfig NF4.
        Giảm VRAM từ ~14GB (bf16) xuống ~4GB (int4), cho phép chạy trên RTX 3050.
        """
        import torch
        import chandra.model as _cm
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )
        from chandra.model import settings as _cs

        # Fix incompatibility giữa transformers 5.x và bitsandbytes 0.49.x
        ChandraEngine._patch_params4bit()

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,        # double quant: tiết kiệm thêm ~0.4GB
            bnb_4bit_quant_type="nf4",             # NF4 tốt hơn fp4 cho văn bản
            llm_int8_enable_fp32_cpu_offload=True, # layer offload CPU dùng fp32 thay vì int4
        )

        # Giới hạn GPU ở 3.5GB — phần còn lại offload CPU ở fp32
        free_mb = 3500
        try:
            free_mb = min(3500, torch.cuda.mem_get_info()[0] // (1024 ** 2) - 300)
        except Exception:
            pass
        max_memory = {0: f"{free_mb}MiB", "cpu": "48GiB"}

        def _patched_load_model():
            logger.info("ChandraEngine [4-bit]: đang load model NF4 (CPU offload fp32)...")
            model = AutoModelForImageTextToText.from_pretrained(
                _cs.MODEL_CHECKPOINT,
                quantization_config=bnb,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
            ).eval()
            processor = AutoProcessor.from_pretrained(_cs.MODEL_CHECKPOINT)
            processor.tokenizer.padding_side = "left"
            model.processor = processor
            logger.success("ChandraEngine [4-bit]: model NF4 đã load xong")
            return model

        original = _cm.load_model
        _cm.load_model = _patched_load_model
        try:
            manager = InferenceManager(method="hf")
        finally:
            _cm.load_model = original   # luôn restore dù có lỗi

        return manager

    def process(self, image: np.ndarray, lang: list[str] | None = None, doc_type: str = "structured") -> PageResult:
        if not self._ready:
            self.warm_up()

        h, w = image.shape[:2]
        t0 = time.perf_counter()

        # BGR numpy → PIL RGB
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        try:
            text_blocks, tables = self._run_inference(pil_image, w, h)
        except Exception as exc:
            logger.error(f"ChandraEngine inference lỗi: {exc}")
            text_blocks, tables = [], []

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return PageResult(
            page_number=1,
            width=w,
            height=h,
            text_blocks=text_blocks,
            tables=tables,
            engine_used=self.name,
            processing_time_ms=elapsed_ms,
        )

    def release(self) -> None:
        self._manager = None
        self._ready = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.debug("ChandraEngine: released")

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Private ───────────────────────────────────────────────────────────

    def _run_inference(
        self, pil_image: Image.Image, w: int, h: int
    ) -> tuple[list[TextBlock], list[Table]]:
        from chandra.model.schema import BatchInputItem

        batch = [BatchInputItem(image=pil_image, prompt_type="ocr_layout")]
        results = self._manager.generate(batch)
        result = results[0]

        if getattr(result, "error", None):
            logger.warning(f"ChandraEngine: inference error — {result.error}")
            return self._parse_markdown_fallback(
                getattr(result, "markdown", "") or "", w, h
            )

        chunks = self._extract_chunks(result, pil_image)
        if chunks:
            return self._process_chunks(chunks, w, h)

        markdown = getattr(result, "markdown", "") or ""
        if markdown:
            logger.debug("ChandraEngine: không có chunks, dùng markdown fallback")
            return self._parse_markdown_fallback(markdown, w, h)

        return [], []

    def _extract_chunks(self, result, pil_image: Image.Image) -> list[dict]:
        # Thử attribute trực tiếp
        raw_chunks = getattr(result, "chunks", None)
        if isinstance(raw_chunks, list) and raw_chunks:
            return raw_chunks

        # Thử parse_chunks helper của chandra
        try:
            from chandra.output import parse_chunks
            raw = getattr(result, "raw", "") or ""
            chunks = parse_chunks(raw, pil_image, bbox_scale=1000)
            if isinstance(chunks, list) and chunks:
                return chunks
        except (ImportError, Exception):
            pass

        return []

    def _process_chunks(
        self, chunks: list[dict], w: int, h: int
    ) -> tuple[list[TextBlock], list[Table]]:
        text_blocks: list[TextBlock] = []
        tables: list[Table] = []

        for chunk in chunks:
            label: str = chunk.get("label", "Text")
            bbox_norm: list = chunk.get("bbox") or [0, 0, 1000, 1000]
            content: str = chunk.get("content", "")

            # Bbox từ đơn vị 1000 (Chandra scale) → pixels
            x0 = max(0, int(bbox_norm[0] * w / 1000))
            y0 = max(0, int(bbox_norm[1] * h / 1000))
            x1 = min(w, int(bbox_norm[2] * w / 1000))
            y1 = min(h, int(bbox_norm[3] * h / 1000))
            bbox = BoundingBox.from_xyxy(x0, y0, x1, y1)

            if label in _TABLE_LABELS:
                table = self._parse_table_html(content, bbox)
                if table:
                    tables.append(table)
            else:
                text = _strip_html(content)
                if text:
                    block_type = _LABEL_TO_BLOCK_TYPE.get(label, "paragraph")
                    text_blocks.append(
                        TextBlock(
                            text=text,
                            confidence=0.95,
                            bbox=bbox,
                            block_type=block_type,
                        )
                    )

        return text_blocks, tables

    def _parse_table_html(self, content: str, bbox: BoundingBox) -> Table | None:
        try:
            parser = _TableHTMLParser()
            parser.feed(content)
            rows = parser.rows
            if not rows:
                return None
            n_rows = len(rows)
            n_cols = max(len(r) for r in rows)
            cells = [
                TableCell(row=r, col=c, text=t)
                for r, row in enumerate(rows)
                for c, t in enumerate(row)
            ]
            return Table(bbox=bbox, rows=n_rows, cols=n_cols, cells=cells)
        except Exception as exc:
            logger.warning(f"ChandraEngine: parse table HTML lỗi — {exc}")
            return None

    def _parse_markdown_fallback(
        self, markdown: str, w: int, h: int
    ) -> tuple[list[TextBlock], list[Table]]:
        lines = [ln.strip() for ln in markdown.splitlines() if ln.strip()]
        if not lines:
            return [], []

        line_height = max(1, h // len(lines))
        text_blocks: list[TextBlock] = []

        for i, line in enumerate(lines):
            block_type = "heading" if line.startswith("#") else "paragraph"
            text = line.lstrip("#").strip()
            if text:
                y0 = i * line_height
                y1 = min(h, (i + 1) * line_height)
                text_blocks.append(
                    TextBlock(
                        text=text,
                        confidence=0.9,
                        bbox=BoundingBox.from_xyxy(0, y0, w, y1),
                        block_type=block_type,
                    )
                )

        return text_blocks, []
