from __future__ import annotations

import gc
import os
import tempfile
import time
from typing import ClassVar

import cv2
import numpy as np
from loguru import logger

from config.settings import Settings
from ocrdoc.core.models import BoundingBox, PageResult, TextBlock
from ocrdoc.core.vram_manager import VRAMManager
from ocrdoc.engines.base import BaseEngine


class OpenOCREngine(BaseEngine):
    """
    Hybrid engine: OpenOCR (PyTorch/SVTRv2) cho text detection
    + PaddleOCR vi_PP-OCRv3/v4 cho Vietnamese text recognition.

    Detection: OpenOCR outperforms PP-OCRv4 ~4.5% trên scene text benchmarks.
    Recognition: PaddleOCR vi_PP-OCRv3 rec model (Vietnamese-specific weights).
                 Khi vi_PP-OCRv4 có sẵn, PaddleOCR tự động dùng bản mới nhất.
    """

    name: ClassVar[str] = "openocr"

    def __init__(self, settings: Settings, vram_manager: VRAMManager):
        self._settings = settings
        self._vram = vram_manager
        self._det = None   # OpenOCR detector (PyTorch/SVTRv2)
        self._rec = None   # PaddleOCR recognition (vi_PP-OCRv3)
        self._ready = False

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def warm_up(self) -> None:
        if self._ready:
            return
        logger.info("OpenOCREngine: đang load models...")
        self._vram.free_vram()

        self._load_openocr_det()
        self._load_paddle_vi_rec()

        self._ready = True
        logger.success("OpenOCREngine: sẵn sàng (OpenOCR det + vi_PP-OCRv3 rec)")

    def _load_openocr_det(self) -> None:
        try:
            from openocr import OpenOCR
            mode = self._settings.openocr_mode
            self._det = OpenOCR(task="det", mode=mode)
            logger.success(f"  OpenOCR detection loaded (mode={mode})")
        except ImportError:
            raise ImportError(
                "openocr-python chưa được cài. Chạy: pip install openocr-python==0.1.5"
            )
        except Exception as e:
            raise RuntimeError(f"OpenOCR detection load thất bại: {e}") from e

    def _load_paddle_vi_rec(self) -> None:
        try:
            from paddleocr import PaddleOCR
            use_gpu = self._settings.paddle_device.lower() == "gpu"
            model_dir = self._settings.resolve_paddle_model_dir()
            # det=False: chỉ load rec + cls model, tiết kiệm VRAM (OpenOCR lo phần detection)
            self._rec = PaddleOCR(
                use_gpu=use_gpu,
                lang="vi",
                use_angle_cls=True,
                det=False,
                model_storage_directory=model_dir,
                show_log=False,
            )
            weights_label = self._settings.openocr_vi_weights.upper()
            logger.success(f"  PaddleOCR vi_PP-OCR{weights_label} recognition loaded")
        except Exception as e:
            raise RuntimeError(f"PaddleOCR vi rec load thất bại: {e}") from e

    def release(self) -> None:
        self._det = None
        self._rec = None
        self._ready = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            import paddle
            paddle.device.cuda.empty_cache()
        except Exception:
            pass
        logger.debug("OpenOCREngine: released")

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Core processing ─────────────────────────────────────────────────────

    def process(self, image: np.ndarray, lang: list[str] | None = None, doc_type: str = "structured") -> PageResult:
        if not self._ready:
            self.warm_up()

        h, w = image.shape[:2]
        t0 = time.perf_counter()

        # Step 1: OpenOCR detection → danh sách polygon bounding boxes
        det_boxes = self._detect(image)
        logger.debug(f"OpenOCR detected {len(det_boxes)} text regions")

        # Step 2: PaddleOCR vi_PP-OCRv3 recognition từng region
        text_blocks: list[TextBlock] = []
        for box in det_boxes:
            crop = self._crop_region(image, box)
            if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
                continue
            text, conf = self._recognize(crop)
            if not text or not text.strip():
                continue
            text_blocks.append(TextBlock(
                text=text.strip(),
                confidence=conf,
                bbox=BoundingBox.from_paddle(box.reshape(-1, 2).tolist()),
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

    # ── Detection ───────────────────────────────────────────────────────────

    def _detect(self, image: np.ndarray) -> list[np.ndarray]:
        """Gọi OpenOCR detector, trả về list polygon arrays (shape N×2)."""
        tmp_path = None
        try:
            # OpenOCR nhận file path; image là BGR từ preprocessing pipeline
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, image)

            raw = self._det(image_path=tmp_path)
            # OpenOCR trả về list[dict{'boxes': ndarray, 'elapse': float}]
            if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], dict):
                boxes = raw[0].get("boxes")
            elif isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
                boxes = raw[0]
            else:
                boxes = raw
            return self._parse_det_output(boxes)
        except Exception as e:
            logger.error(f"OpenOCR detection lỗi: {e}")
            return []
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _parse_det_output(self, results) -> list[np.ndarray]:
        """
        Normalize OpenOCR detection output thành list[np.ndarray shape (N,2)].

        OpenOCR trả về một trong các format:
          A) [ [[x,y],[x,y],[x,y],[x,y]], ... ]          ← list of polygons
          B) [[ [[x,y],...], [[x,y],...], ... ]]         ← batch wrap
        """
        if results is None or (hasattr(results, '__len__') and len(results) == 0):
            return []

        raw = results

        # Unwrap batch dimension nếu cần: [[box1, box2, ...]]
        if (
            isinstance(raw, (list, tuple))
            and raw
            and isinstance(raw[0], (list, tuple))
            and raw[0]
            and isinstance(raw[0][0], (list, tuple))
            and raw[0][0]
            and isinstance(raw[0][0][0], (list, tuple))
        ):
            raw = raw[0]

        boxes = []
        for item in raw:
            if item is None:
                continue
            try:
                pts = np.array(item, dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] >= 2:
                    boxes.append(pts)
            except Exception:
                continue
        return boxes

    # ── Crop & Recognition ──────────────────────────────────────────────────

    def _crop_region(self, image: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        """Perspective transform crop cho quadrilateral text box."""
        try:
            pts = box.reshape(-1, 2).astype(np.float32)

            if len(pts) == 4:
                # Quadrilateral → perspective warp để giữ text thẳng
                w_px = int(max(
                    np.linalg.norm(pts[0] - pts[1]),
                    np.linalg.norm(pts[2] - pts[3]),
                ))
                h_px = int(max(
                    np.linalg.norm(pts[0] - pts[3]),
                    np.linalg.norm(pts[1] - pts[2]),
                ))
                if w_px <= 0 or h_px <= 0:
                    return None
                dst = np.array(
                    [[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]],
                    dtype=np.float32,
                )
                M = cv2.getPerspectiveTransform(pts, dst)
                return cv2.warpPerspective(image, M, (w_px, h_px))
            else:
                # Fallback: axis-aligned crop
                x1 = max(0, int(pts[:, 0].min()))
                y1 = max(0, int(pts[:, 1].min()))
                x2 = min(image.shape[1], int(pts[:, 0].max()))
                y2 = min(image.shape[0], int(pts[:, 1].max()))
                return image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
        except Exception as e:
            logger.warning(f"Crop region lỗi: {e}")
            return None

    def _recognize(self, crop: np.ndarray) -> tuple[str, float]:
        """PaddleOCR vi_PP-OCRv3 recognition trên một crop BGR."""
        try:
            result = self._rec.ocr(crop, det=False, cls=True)
            if result and result[0]:
                text, conf = result[0][0][0], float(result[0][0][1])
                return text, conf
        except Exception as e:
            logger.warning(f"vi_PP-OCRv3 recognition lỗi: {e}")
        return "", 0.0
