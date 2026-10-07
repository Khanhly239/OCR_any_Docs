from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image


class PDFConversionError(RuntimeError):
    pass


class PDFConverter:
    def __init__(self, dpi: int = 300, fmt: str = "PNG"):
        self.dpi = dpi
        self.fmt = fmt

    def convert(self, pdf_path: Path, max_pages: int = 100) -> list[np.ndarray]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File không tồn tại: {pdf_path}")

        # Thử pdf2image trước, fallback sang PyMuPDF
        try:
            return self._convert_via_pdf2image(pdf_path, max_pages)
        except Exception as e:
            logger.warning(f"pdf2image thất bại ({e}), thử PyMuPDF...")
            return self._convert_via_pymupdf(pdf_path, max_pages)

    def _convert_via_pdf2image(self, pdf_path: Path, max_pages: int) -> list[np.ndarray]:
        from pdf2image import convert_from_path
        from pdf2image.exceptions import PDFInfoNotInstalledError

        poppler_path = self._get_poppler_path()
        try:
            pil_images = convert_from_path(
                str(pdf_path),
                dpi=self.dpi,
                fmt=self.fmt,
                first_page=1,
                last_page=max_pages,
                poppler_path=poppler_path or None,
            )
        except PDFInfoNotInstalledError:
            raise RuntimeError(
                "Poppler không tìm thấy. Cài đặt:\n"
                "  choco install poppler\n"
                "  hoặc đặt POPPLER_PATH trong .env"
            )
        return [self._pil_to_bgr(img) for img in pil_images]

    def _convert_via_pymupdf(self, pdf_path: Path, max_pages: int) -> list[np.ndarray]:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        images = []
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            images.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        doc.close()
        return images

    @staticmethod
    def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
        rgb = np.array(pil_img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _get_poppler_path() -> str:
        # 1. Từ biến môi trường
        env_path = os.environ.get("POPPLER_PATH", "")
        if env_path and Path(env_path).exists():
            return env_path
        # 2. Vị trí cài mặc định trên Windows qua Chocolatey
        default = Path(r"C:\ProgramData\chocolatey\lib\poppler\tools\Library\bin")
        if default.exists():
            return str(default)
        # 3. Để None để pdf2image tự tìm trong PATH
        return ""
