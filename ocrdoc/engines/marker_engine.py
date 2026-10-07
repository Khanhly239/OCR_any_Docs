"""
Marker engine — Document-to-Markdown converter (datalab-to/marker).
Chạy qua subprocess trong marker_venv riêng vì marker-pdf không tương thích
với transformers 5.x (dùng transformers.onnx đã bị xóa).

Setup (chạy 1 lần):
    scripts\\setup_marker_venv.ps1

VRAM: ~3.5-5GB. Tốt cho: tài liệu học thuật, báo cáo, bảng phức tạp.
"""
from __future__ import annotations

import gc
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from config.settings import Settings
from ocrdoc.core.models import BoundingBox, PageResult, TextBlock
from ocrdoc.core.vram_manager import VRAMManager
from ocrdoc.engines.base import BaseEngine

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
_VENV_PYTHON  = _SCRIPTS_DIR / "marker_venv" / "Scripts" / "python.exe"
_RUNNER       = _SCRIPTS_DIR / "marker_runner.py"


class MarkerEngine(BaseEngine):
    """
    Marker — chuyển ảnh/PDF thành Markdown qua pipeline Surya.
    Chạy trong subprocess (marker_venv) để tránh xung đột transformers.
    Output: markdown với heading, table, LaTeX, code block.
    """

    name: ClassVar[str] = "marker"

    def __init__(self, settings: Settings, vram_manager: VRAMManager):
        self._settings = settings
        self._vram = vram_manager
        self._proc: subprocess.Popen | None = None
        self._ready = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def warm_up(self) -> None:
        if self._ready:
            return

        if not _VENV_PYTHON.exists():
            raise RuntimeError(
                "Marker venv chưa tồn tại.\n"
                "Chạy lệnh sau để setup (1 lần):\n"
                "    scripts\\setup_marker_venv.ps1"
            )

        logger.info("MarkerEngine: khởi động worker subprocess...")
        self._proc = subprocess.Popen(
            [str(_VENV_PYTHON), "-u", str(_RUNNER)],  # -u: unbuffered stdout
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,   # capture để hiển thị lỗi thật
            text=True,
            encoding="utf-8",
        )

        # Đọc stdout theo dòng trong thread riêng để tránh deadlock
        import queue, threading

        q: queue.Queue[str | None] = queue.Queue()

        def _reader():
            try:
                for line in self._proc.stdout:
                    q.put(line.rstrip())
            except Exception:
                pass
            finally:
                q.put(None)

        threading.Thread(target=_reader, daemon=True).start()

        # Timeout 30 phút — lần đầu cần load nhiều models
        deadline = time.time() + 1800
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                self._proc.kill()
                raise RuntimeError("MarkerEngine: timeout chờ READY (>30 phút)")
            try:
                line = q.get(timeout=min(remaining, 5))
            except queue.Empty:
                if self._proc.poll() is not None:
                    err = self._proc.stderr.read(4000).strip()
                    raise RuntimeError(f"MarkerEngine crash:\n{err}")
                continue

            if line is None:
                err = self._proc.stderr.read(4000).strip()
                raise RuntimeError(f"MarkerEngine crash:\n{err}")

            if line == "READY":
                break
            if line.startswith("LOADING "):
                logger.info(f"MarkerEngine: {line[8:]}")
            elif line.startswith("ERROR"):
                err_line = q.get(timeout=10) or ""
                raise RuntimeError(f"Marker worker lỗi: {err_line}")

        self._ready = True
        logger.success("MarkerEngine: worker sẵn sàng")

    def release(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.write("QUIT\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
        self._proc = None
        self._ready = False
        gc.collect()
        logger.debug("MarkerEngine: released")

    @property
    def is_ready(self) -> bool:
        return self._ready and self._proc is not None and self._proc.poll() is None

    # ── Process ───────────────────────────────────────────────────────────────

    def process(
        self,
        image: np.ndarray,
        lang: list[str] | None = None,
        doc_type: str = "structured",
    ) -> PageResult:
        if not self.is_ready:
            self.warm_up()

        h, w = image.shape[:2]
        t0 = time.perf_counter()

        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tmp_path = None
        text = ""

        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                pil.save(tmp_path)

            # Gửi lệnh tới worker
            self._proc.stdin.write(f"PROCESS {tmp_path}\n")
            self._proc.stdin.flush()

            # Đọc phản hồi: "RESULT\n<json>\nEND\n"
            header = self._proc.stdout.readline().strip()
            if header != "RESULT":
                # Worker crash trong lúc inference — đọc stderr để biết lý do
                err = ""
                if self._proc.poll() is not None:
                    err = self._proc.stderr.read(2000).strip()
                self._ready = False  # đánh dấu cần restart lần sau
                raise RuntimeError(
                    f"Marker worker crash khi xử lý ảnh.\n{err}" if err
                    else "Marker worker crash khi xử lý ảnh (có thể do hết RAM/page file)"
                )

            json_parts = []
            for line in self._proc.stdout:
                if line.strip() == "END":
                    break
                json_parts.append(line)

            result = json.loads("".join(json_parts))
            text = result.get("text", "")
            if not result.get("ok"):
                logger.error(f"MarkerEngine lỗi từ worker: {result.get('error')}")

        except Exception as e:
            logger.error(f"MarkerEngine: {e}")
            text = ""
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"MarkerEngine: {elapsed_ms:.0f}ms, {len(text)} ký tự")

        return PageResult(
            page_number=1,
            width=w,
            height=h,
            text_blocks=self._text_to_blocks(text, w, h),
            tables=[],
            engine_used=self.name,
            processing_time_ms=elapsed_ms,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _text_to_blocks(self, text: str, w: int, h: int) -> list[TextBlock]:
        # Giữ toàn bộ markdown trong 1 block để result_to_outputs render đúng
        if not text:
            return []
        return [TextBlock(
            text=text,
            confidence=0.9,
            bbox=BoundingBox(x=0, y=0, w=w, h=h),
            block_type="paragraph",
        )]
