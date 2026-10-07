from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger

from config.settings import Settings, settings as default_settings
from ocrdoc.core.classifier import DocumentClassifier
from ocrdoc.core.models import OCRResult, PageResult
from ocrdoc.core.vram_manager import VRAMManager
from ocrdoc.engines.chandra_engine import ChandraEngine
from ocrdoc.engines.easyocr_engine import EasyOCREngine
from ocrdoc.engines.marker_engine import MarkerEngine
from ocrdoc.engines.openocr_engine import OpenOCREngine
from ocrdoc.engines.paddle_engine import PaddleEngine
from ocrdoc.engines.vintern_engine import VinternEngine
from ocrdoc.exporters import get_exporter
from ocrdoc.postprocessing.merger import ResultMerger
from ocrdoc.postprocessing.reading_order import ReadingOrderSorter
from ocrdoc.postprocessing.vietnamese import VietnameseNormalizer
from ocrdoc.preprocessing.pipeline import PreprocessingPipeline


class OCROrchestrator:
    def __init__(self, settings: Settings | None = None):
        self._settings = settings or default_settings
        self._classifier = DocumentClassifier()
        self._preprocessor = PreprocessingPipeline(dpi=self._settings.default_dpi)
        self._vram = VRAMManager()
        self._paddle = PaddleEngine(self._settings, self._vram)
        self._chandra = ChandraEngine(self._settings, self._vram)
        self._openocr = OpenOCREngine(self._settings, self._vram)
        self._easyocr = EasyOCREngine(self._settings, self._vram)
        self._vintern = VinternEngine(self._settings, self._vram)
        self._marker = MarkerEngine(self._settings, self._vram)
        self._vi_normalizer = VietnameseNormalizer()
        self._reading_order = ReadingOrderSorter()
        self._merger = ResultMerger()

    # ── Public API ──────────────────────────────────────────────────────────

    def process(
        self,
        source: Path | bytes,
        filename: str = "document",
        mode: Literal["auto", "structured", "unstructured", "chandra", "openocr", "easyocr", "vintern", "marker"] = "auto",
        output_formats: list[str] | None = None,
        lang: list[str] | None = None,
    ) -> dict[str, str | bytes]:
        """
        Pipeline đầy đủ: input → preprocessing → OCR → postprocessing → export.

        Returns:
            dict với key là format ("json", "txt", "markdown", "xlsx") và value là nội dung.
        """
        output_formats = output_formats or ["json", "txt"]
        lang = lang or ["vi", "en"]
        t_start = time.perf_counter()

        # 1. Load images
        pages_images = self._load_images(source, filename)
        logger.info(f"Loaded {len(pages_images)} trang từ '{filename}'")

        # 2. Phân loại document (dùng trang đầu tiên)
        doc_type = self._classify(pages_images[0], mode)
        logger.info(f"Document type: {doc_type}")

        # 3. OCR từng trang
        engine = self._select_engine(mode, doc_type)
        page_results: list[PageResult] = []

        with self._vram.acquire(engine.name):
            if not engine.is_ready:
                engine.warm_up()
            for i, page_img in enumerate(pages_images):
                logger.info(f"  OCR trang {i + 1}/{len(pages_images)}...")
                page_result = engine.process(page_img, lang=lang, doc_type=doc_type)
                page_results.append(page_result)

        # 4. Postprocessing
        page_results = [self._postprocess_page(p) for p in page_results]

        # 5. Merge
        total_ms = (time.perf_counter() - t_start) * 1000
        ocr_result = self._merger.merge(
            pages=page_results,
            source_path=str(filename),
            doc_type=doc_type,
            total_time_ms=total_ms,
        )

        logger.success(f"OCR hoàn tất: {total_ms:.0f}ms, {len(page_results)} trang")

        # 6. Export
        return self._export(ocr_result, output_formats)

    def process_and_stream(
        self,
        source: Path | bytes,
        filename: str = "document",
        mode: Literal["auto", "structured", "unstructured", "chandra", "openocr", "easyocr", "marker"] = "auto",
        output_formats: list[str] | None = None,
        lang: list[str] | None = None,
    ):
        """Generator version cho SSE streaming. Yield dicts với event/progress info."""
        output_formats = output_formats or ["json", "txt"]
        lang = lang or ["vi", "en"]
        t_start = time.perf_counter()

        pages_images = self._load_images(source, filename)
        total_pages = len(pages_images)
        yield {"event": "start", "total_pages": total_pages}

        doc_type = self._classify(pages_images[0], mode)
        yield {"event": "classified", "doc_type": doc_type}

        engine = self._select_engine(mode, doc_type)
        page_results: list[PageResult] = []

        with self._vram.acquire(engine.name):
            if not engine.is_ready:
                engine.warm_up()
            for i, page_img in enumerate(pages_images):
                yield {"event": "ocr_start", "page": i + 1, "total": total_pages, "engine": engine.name}
                page_result = engine.process(page_img, lang=lang, doc_type=doc_type)
                page_results.append(page_result)
                yield {"event": "ocr_done", "page": i + 1, "total": total_pages}

        page_results = [self._postprocess_page(p) for p in page_results]
        total_ms = (time.perf_counter() - t_start) * 1000
        ocr_result = self._merger.merge(page_results, str(filename), doc_type, total_ms)
        outputs = self._export(ocr_result, output_formats)
        yield {"event": "done", "results": outputs, "total_ms": total_ms}

    def startup(self) -> None:
        logger.info("OCROrchestrator: startup — pre-warming PaddleEngine...")
        try:
            with self._vram.acquire(self._paddle.name, timeout_s=120.0):
                self._paddle.warm_up()
            logger.success("PaddleEngine warm-up hoàn tất")
        except Exception as e:
            logger.warning(f"Warm-up thất bại (sẽ load lần đầu request): {e}")

    def shutdown(self) -> None:
        for engine in (self._paddle, self._chandra, self._openocr, self._easyocr, self._vintern, self._marker):
            if engine.is_ready:
                engine.release()
        logger.info("OCROrchestrator: shutdown")

    # ── Private helpers ─────────────────────────────────────────────────────

    def _load_images(self, source: Path | bytes, filename: str) -> list[np.ndarray]:
        if isinstance(source, (Path, str)):
            return self._preprocessor.process_path(
                Path(source), max_pages=self._settings.max_pdf_pages
            )
        else:
            return self._preprocessor.process_bytes(
                source, filename=filename, max_pages=self._settings.max_pdf_pages
            )

    def _select_engine(self, mode: str, doc_type: str):
        if mode == "easyocr":
            return self._easyocr
        if mode == "openocr":
            return self._openocr
        if mode == "chandra":
            return self._chandra
        if mode == "vintern":
            return self._vintern
        if mode == "marker":
            return self._marker
        return self._paddle

    def _classify(
        self,
        first_page: np.ndarray,
        mode: str,
    ) -> Literal["structured", "unstructured"]:
        if mode == "structured":
            return "structured"
        if mode in ("unstructured", "chandra", "openocr", "easyocr", "vintern", "marker"):
            return "unstructured"
        return self._classifier.classify(first_page)

    def _postprocess_page(self, page: PageResult) -> PageResult:
        normalized_blocks = []
        for block in page.text_blocks:
            normalized_text = self._vi_normalizer.normalize(block.text)
            normalized_blocks.append(block.model_copy(update={"text": normalized_text}))

        sorted_blocks = self._reading_order.sort(normalized_blocks, page_width=page.width)

        return page.model_copy(update={"text_blocks": sorted_blocks})

    def _export(self, result: OCRResult, formats: list[str]) -> dict[str, str | bytes]:
        outputs: dict[str, str | bytes] = {}
        for fmt in formats:
            try:
                exporter = get_exporter(fmt)
                content = exporter.export(result)
                # xlsx là bytes, các format khác là str
                if isinstance(content, bytes):
                    outputs[fmt] = base64.b64encode(content).decode("utf-8")
                else:
                    outputs[fmt] = content
            except Exception as e:
                logger.error(f"Export '{fmt}' thất bại: {e}")
                outputs[fmt] = f"[Lỗi export: {e}]"
        return outputs
