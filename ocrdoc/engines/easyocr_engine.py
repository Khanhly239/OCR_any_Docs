from __future__ import annotations

import gc
import time
from typing import ClassVar

import cv2
import numpy as np
from loguru import logger

from config.settings import Settings
from ocrdoc.core.models import BoundingBox, PageResult, TextBlock
from ocrdoc.core.vram_manager import VRAMManager
from ocrdoc.engines.base import BaseEngine


class EasyOCREngine(BaseEngine):
    """
    EasyOCR engine với Vietnamese + English support.
    Dùng character model train riêng cho tiếng Việt — xử lý đúng toàn bộ dấu
    tiếng Việt (ă, ơ, ư, ạ, ẩ, ắ, ề, ộ...) mà latin_PP-OCRv3 không có.
    """

    name: ClassVar[str] = "easyocr"

    def __init__(self, settings: Settings, vram_manager: VRAMManager):
        self._settings = settings
        self._vram = vram_manager
        self._reader = None
        self._ready = False

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def warm_up(self) -> None:
        if self._ready:
            return
        logger.info("EasyOCREngine: đang load models (vi + en)...")
        self._vram.free_vram()

        try:
            import easyocr
        except ImportError:
            raise ImportError(
                "easyocr chưa được cài. Chạy: pip install easyocr"
            )

        use_gpu = self._settings.paddle_device.lower() == "gpu"
        model_dir = str(self._settings.easyocr_model_dir)

        self._reader = easyocr.Reader(
            lang_list=["vi", "en"],
            gpu=use_gpu,
            model_storage_directory=model_dir,
            download_enabled=True,
            verbose=False,
        )

        self._ready = True
        logger.success("EasyOCREngine: sẵn sàng (Vietnamese + English)")

    def release(self) -> None:
        self._reader = None
        self._ready = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.debug("EasyOCREngine: released")

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Core processing ─────────────────────────────────────────────────────

    def process(self, image: np.ndarray, lang: list[str] | None = None, doc_type: str = "structured") -> PageResult:
        if not self._ready:
            self.warm_up()

        h, w = image.shape[:2]
        t0 = time.perf_counter()

        # EasyOCR nhận RGB; preprocessing pipeline trả về BGR
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            results = self._reader.readtext(
                rgb,
                detail=1,
                paragraph=False,
                batch_size=8,
            )
        except Exception as e:
            logger.error(f"EasyOCR readtext lỗi: {e}")
            results = []

        text_blocks: list[TextBlock] = []
        for item in results:
            if not item or len(item) < 3:
                continue
            bbox_points, text, conf = item[0], item[1], float(item[2])
            if not text or not text.strip():
                continue
            # EasyOCR bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] — cùng format với PaddleOCR
            text_blocks.append(TextBlock(
                text=text.strip(),
                confidence=conf,
                bbox=BoundingBox.from_paddle(bbox_points),
                block_type="paragraph",
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return PageResult(
            page_number=1,
            width=w,
            height=h,
            text_blocks=text_blocks,
            tables=[],
            engine_used=self.name,
            processing_time_ms=elapsed_ms,
        )
