from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ocrdoc.core.models import PageResult


class BaseEngine(ABC):
    name: ClassVar[str] = "base"

    @abstractmethod
    def warm_up(self) -> None:
        """Load model weights vào memory. Gọi một lần khi khởi động."""

    @abstractmethod
    def process(
        self,
        image: np.ndarray,
        lang: list[str] | None = None,
        doc_type: str = "structured",
    ) -> PageResult:
        """Chạy OCR trên một trang ảnh đã preprocessing. Trả về PageResult."""

    @abstractmethod
    def release(self) -> None:
        """Giải phóng GPU memory. Gọi trước khi chuyển sang engine khác."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """True nếu warm_up() đã được gọi và model đang ở trong memory."""
