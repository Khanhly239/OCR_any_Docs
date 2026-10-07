from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Generator

from loguru import logger


class VRAMManager:
    """
    Singleton lock đảm bảo chỉ 1 engine sử dụng GPU tại một thời điểm.
    Tránh OOM khi chạy đồng thời PaddleOCR và Surya trên cùng một card.
    """

    _instance: "VRAMManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "VRAMManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._gpu_lock = threading.Lock()
                    inst._current_tenant: str | None = None
                    cls._instance = inst
        return cls._instance

    @contextmanager
    def acquire(self, engine_name: str, timeout_s: float = 120.0) -> Generator[None, None, None]:
        acquired = self._gpu_lock.acquire(timeout=timeout_s)
        if not acquired:
            raise TimeoutError(
                f"Engine '{engine_name}' không thể lấy GPU lock sau {timeout_s}s. "
                f"Engine đang giữ lock: {self._current_tenant}"
            )
        self._current_tenant = engine_name
        logger.debug(f"VRAM lock: '{engine_name}' đã lấy GPU")
        try:
            yield
        finally:
            self._current_tenant = None
            self._gpu_lock.release()
            logger.debug(f"VRAM lock: '{engine_name}' đã giải phóng GPU")

    def free_vram(self) -> None:
        """Gọi gc và empty_cache để giải phóng VRAM trước khi load engine mới."""
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("torch.cuda.empty_cache() done")
        except ImportError:
            pass

    def vram_free_mb(self) -> int:
        """Trả về VRAM còn trống (MB). Trả về -1 nếu không có GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                return free // (1024 * 1024)
        except Exception:
            pass
        return -1
