from __future__ import annotations

from abc import ABC, abstractmethod

from ocrdoc.core.models import OCRResult


class BaseExporter(ABC):
    @abstractmethod
    def export(self, result: OCRResult) -> str | bytes:
        """Chuyển OCRResult sang format output. Trả về str hoặc bytes."""

    @property
    @abstractmethod
    def mime_type(self) -> str:
        """MIME type của output."""

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Phần mở rộng file (không có dấu chấm)."""
