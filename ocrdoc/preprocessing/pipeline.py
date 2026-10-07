from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from ocrdoc.preprocessing.deskew import Deskewer
from ocrdoc.preprocessing.denoise import Denoiser
from ocrdoc.preprocessing.enhance import ContrastEnhancer
from ocrdoc.preprocessing.pdf_converter import PDFConverter


class PreprocessingPipeline:
    def __init__(
        self,
        deskew: bool = True,
        denoise: bool = True,
        enhance: bool = True,
        dpi: int = 300,
        denoise_strength: str = "medium",
    ):
        self.do_deskew = deskew
        self.do_denoise = denoise
        self.do_enhance = enhance
        self._deskewer = Deskewer()
        self._denoiser = Denoiser(strength=denoise_strength)
        self._enhancer = ContrastEnhancer()
        self._pdf_converter = PDFConverter(dpi=dpi)

    def process(self, image: np.ndarray) -> np.ndarray:
        if self.do_deskew:
            image = self._deskewer.deskew(image)
        if self.do_denoise:
            image = self._denoiser.denoise(image)
        if self.do_enhance and self._enhancer.is_low_contrast(image):
            image = self._enhancer.enhance(image)
        return image

    def process_path(self, path: Path, max_pages: int = 100) -> list[np.ndarray]:
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            logger.info(f"Chuyển đổi PDF: {path.name}")
            raw_pages = self._pdf_converter.convert(path, max_pages=max_pages)
        else:
            pil = Image.open(path).convert("RGB")
            arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            raw_pages = [arr]

        logger.info(f"Preprocessing {len(raw_pages)} trang...")
        return [self.process(page) for page in raw_pages]

    def process_bytes(self, data: bytes, filename: str = "upload", max_pages: int = 100) -> list[np.ndarray]:
        """Xử lý từ bytes upload (API/Gradio)."""
        import tempfile, os
        suffix = Path(filename).suffix.lower() or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            return self.process_path(tmp_path, max_pages=max_pages)
        finally:
            os.unlink(tmp_path)
