"""
Vintern-1B engine — Vietnamese VLM (InternVL2 + Qwen2 backbone).
Model: 5CD-AI/Vintern-1B-v2  (~1B params, ~2GB VRAM)
Install deps: pip install einops timm sentencepiece tiktoken
"""
from __future__ import annotations

import gc
import re
import time
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from config.settings import Settings
from ocrdoc.core.models import BoundingBox, PageResult, TextBlock
from ocrdoc.core.vram_manager import VRAMManager
from ocrdoc.engines.base import BaseEngine

_OCR_PROMPT = (
    "<image>\n"
    "Hãy đọc và trích xuất toàn bộ văn bản trong ảnh này. "
    "Giữ nguyên thứ tự đọc từ trên xuống dưới, từ trái sang phải. "
    "Chỉ trả về văn bản, không giải thích thêm."
)

_STRUCTURED_PROMPT = (
    "<image>\n"
    "Hãy đọc và trích xuất toàn bộ văn bản trong ảnh này. "
    "Nếu có bảng, hãy trình bày bảng theo dạng markdown. "
    "Nếu có tiêu đề, hãy đánh dấu bằng dấu ##. "
    "Giữ nguyên thứ tự đọc. Chỉ trả về nội dung, không giải thích."
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE = 448


# ── Image preprocessing (dynamic tiling) ──────────────────────────────────

def _build_transform():
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _dynamic_preprocess(image: Image.Image, min_num=1, max_num=12):
    w, h = image.size
    aspect = w / h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    best, best_diff = (1, 1), float("inf")
    for r in target_ratios:
        diff = abs(aspect - r[0] / r[1])
        if diff < best_diff or (diff == best_diff and w * h > 0.5 * IMG_SIZE**2 * r[0] * r[1]):
            best_diff, best = diff, r

    tw, th = IMG_SIZE * best[0], IMG_SIZE * best[1]
    resized = image.resize((tw, th))
    patches = [
        resized.crop((
            (i % best[0]) * IMG_SIZE, (i // best[0]) * IMG_SIZE,
            ((i % best[0]) + 1) * IMG_SIZE, ((i // best[0]) + 1) * IMG_SIZE,
        ))
        for i in range(best[0] * best[1])
    ]
    if best[0] * best[1] != 1:
        patches.append(image.resize((IMG_SIZE, IMG_SIZE)))
    return patches


# ── HF cache compatibility patches (transformers 5.x) ─────────────────────

def _patch_file(path: Path, old: str, new: str, sentinel: str) -> None:
    content = path.read_text(encoding="utf-8")
    if sentinel in content or old not in content:
        return
    path.write_text(content.replace(old, new, 1), encoding="utf-8")
    logger.debug(f"Patched {path.name}: {sentinel!r}")


def _apply_patches(snapshot_dir: Path) -> None:
    dirs = [snapshot_dir]
    modules_root = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    for vendor in modules_root.glob("_5CD*"):
        for mdl in vendor.glob("Vintern*"):
            for commit in mdl.iterdir():
                if commit.is_dir():
                    dirs.append(commit)

    for d in dirs:
        cfg = d / "configuration_internvl_chat.py"
        if cfg.exists():
            _patch_file(cfg,
                "        if llm_config['architectures'][0] == 'LlamaForCausalLM':",
                "        arch = llm_config.get('architectures', ['Qwen2ForCausalLM'])  # t5fix\n"
                "        if arch[0] == 'LlamaForCausalLM':", "t5fix")
            _patch_file(cfg,
                "        elif llm_config['architectures'][0] == 'Qwen2ForCausalLM':",
                "        elif arch[0] == 'Qwen2ForCausalLM':", "arch[0]")
            _patch_file(cfg,
                "            raise ValueError('Unsupported architecture: {}'.format(llm_config['architectures'][0]))",
                "            raise ValueError('Unsupported architecture: {}'.format(arch[0]))", "arch[0])")

        mdl = d / "modeling_internvl_chat.py"
        if mdl.exists():
            _patch_file(mdl,
                "    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'Qwen2DecoderLayer']",
                "    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'Qwen2DecoderLayer']\n"
                "    all_tied_weights_keys = {}  # t5fix", "t5fix")
            _patch_file(mdl,
                "            return_dict=return_dict,\n            use_cache=True,",
                "            use_cache=True,  # return_dict removed t5fix", "return_dict removed t5fix")


def _ensure_tokenizer_json(snapshot_dir: Path) -> None:
    tok_json = snapshot_dir / "tokenizer.json"
    if tok_json.exists() and tok_json.stat().st_size > 100_000:
        return
    logger.info("VinternEngine: tải tokenizer.json từ Qwen2-0.5B...")
    import shutil
    from huggingface_hub import hf_hub_download
    src = hf_hub_download("Qwen/Qwen2-0.5B", "tokenizer.json")
    shutil.copy2(src, tok_json)


# ── Engine ─────────────────────────────────────────────────────────────────

class VinternEngine(BaseEngine):
    """
    Vintern-1B-v2 — VLM 1B fine-tuned cho tiếng Việt.
    Nhẹ (~2GB VRAM), chạy được trên RTX 3050 Laptop.
    Tốt cho: văn bản tự do, báo cáo, giấy tờ scan tiếng Việt.
    """

    name: ClassVar[str] = "vintern"
    MODEL_ID: ClassVar[str] = "5CD-AI/Vintern-1B-v2"

    def __init__(self, settings: Settings, vram_manager: VRAMManager):
        self._settings = settings
        self._vram = vram_manager
        self._model = None
        self._tokenizer = None
        self._local_dir: str = ""
        self._ready = False

    def warm_up(self) -> None:
        if self._ready:
            return
        logger.info(f"VinternEngine: đang load {self.MODEL_ID} ...")
        self._vram.free_vram()

        for pkg in ("einops", "timm", "sentencepiece", "tiktoken"):
            try:
                __import__(pkg)
            except ImportError:
                raise ImportError(
                    f"VinternEngine yêu cầu '{pkg}'. "
                    "Cài: pip install einops timm sentencepiece tiktoken"
                )

        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        # Resolve local path — thử offline trước (nhanh), fallback download nếu chưa có
        try:
            local_dir = Path(snapshot_download(self.MODEL_ID, local_files_only=True))
            logger.debug("VinternEngine: dùng model từ cache (offline)")
        except Exception:
            logger.info("VinternEngine: chưa có cache, đang download...")
            local_dir = Path(snapshot_download(self.MODEL_ID))
        self._local_dir = str(local_dir)

        # Patches cho transformers 5.x + tokenizer.json
        _ensure_tokenizer_json(local_dir)
        _apply_patches(local_dir)

        device = (
            "cuda"
            if self._settings.paddle_device.lower() in ("gpu", "cuda")
            and torch.cuda.is_available()
            else "cpu"
        )
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self._model = AutoModel.from_pretrained(
            self._local_dir,
            dtype=dtype,                   # transformers 5.x: dtype (không dùng torch_dtype)
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        ).eval().to(device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._local_dir,
            trust_remote_code=True,
            local_files_only=True,
        )

        self._device = device
        self._ready = True
        logger.success(f"VinternEngine: model sẵn sàng trên {device}")

    def process(
        self,
        image: np.ndarray,
        lang: list[str] | None = None,
        doc_type: str = "unstructured",
    ) -> PageResult:
        if not self._ready:
            self.warm_up()

        h, w = image.shape[:2]
        t0 = time.perf_counter()

        prompt = _STRUCTURED_PROMPT if doc_type == "structured" else _OCR_PROMPT

        try:
            text = self._run_inference(image, prompt)
        except Exception as e:
            logger.error(f"VinternEngine inference lỗi: {e}")
            text = ""

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"VinternEngine: {elapsed_ms:.0f}ms, {len(text)} ký tự")

        return PageResult(
            page_number=1,
            width=w,
            height=h,
            text_blocks=self._text_to_blocks(text, w, h),
            tables=[],
            engine_used=self.name,
            processing_time_ms=elapsed_ms,
        )

    def stream_ocr(self, image: np.ndarray, doc_type: str = "unstructured"):
        """Yield partial OCR text token-by-token via TextIteratorStreamer."""
        import threading
        from transformers import TextIteratorStreamer

        if not self._ready:
            self.warm_up()

        pixel_values = self._load_image_tensor(image)
        prompt = _STRUCTURED_PROMPT if doc_type == "structured" else _OCR_PROMPT
        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=120.0,
        )
        gen_config = {"max_new_tokens": 2048, "do_sample": False, "streamer": streamer}

        def _chat():
            try:
                self._model.chat(self._tokenizer, pixel_values, prompt, gen_config)
            except Exception as exc:
                logger.debug(f"VinternEngine stream thread: {exc}")
            finally:
                try:
                    streamer.on_finalized_text("", stream_end=True)
                except Exception:
                    pass

        thread = threading.Thread(target=_chat, daemon=True)
        thread.start()

        partial = ""
        try:
            for token in streamer:
                partial += token
                yield partial
        except Exception as exc:
            logger.warning(f"VinternEngine stream bị ngắt: {exc}")

        thread.join(timeout=15)
        if not partial:
            logger.warning("VinternEngine: streamer không hoạt động, dùng blocking inference")
            yield self._run_inference(image, prompt)

    def _run_inference(self, image: np.ndarray, prompt: str) -> str:
        pixel_values = self._load_image_tensor(image)
        gen_config = {"max_new_tokens": 2048, "do_sample": False, "repetition_penalty": 1.05}
        response = self._model.chat(self._tokenizer, pixel_values, prompt, gen_config)
        return response.strip() if isinstance(response, str) else ""

    def _load_image_tensor(self, image: np.ndarray):
        """Chuyển BGR numpy → dynamic-tiled tensor theo format InternVL2."""
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        transform = _build_transform()
        patches = _dynamic_preprocess(pil, max_num=12)
        tensors = torch.stack([transform(p) for p in patches])
        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype
        return tensors.to(device=device, dtype=dtype)

    def _text_to_blocks(self, text: str, w: int, h: int) -> list[TextBlock]:
        """Chuyển raw text từ VLM thành list[TextBlock], mỗi đoạn/dòng = 1 block."""
        if not text:
            return []

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return [TextBlock(
                text=text.strip(), confidence=0.9,
                bbox=BoundingBox(x=0, y=0, w=w, h=h), block_type="paragraph",
            )]

        y_step = max(1, h // len(lines))
        blocks = []
        for i, line in enumerate(lines):
            if line.startswith("##"):
                block_type, line = "heading", line.lstrip("#").strip()
            elif re.match(r"^\|.+\|$", line):
                block_type = "paragraph"  # markdown table row
            else:
                block_type = "paragraph"

            blocks.append(TextBlock(
                text=line,
                confidence=0.9,
                bbox=BoundingBox(x=0, y=i * y_step, w=w, h=y_step),
                block_type=block_type,
            ))
        return blocks

    def release(self) -> None:
        self._model = None
        self._tokenizer = None
        self._ready = False
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.debug("VinternEngine: released")

    @property
    def is_ready(self) -> bool:
        return self._ready
