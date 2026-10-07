from __future__ import annotations

from fastapi import APIRouter

from api.schemas import HealthResponse, InfoResponse
from ocrdoc.core.vram_manager import VRAMManager

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    vram = VRAMManager()
    free_mb = vram.vram_free_mb()
    gpu_ok = free_mb >= 0

    return HealthResponse(
        status="ok",
        gpu_available=gpu_ok,
        vram_free_mb=max(free_mb, 0),
    )


@router.get("/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    try:
        import paddleocr
        paddle_ver = getattr(paddleocr, "__version__", "unknown")
    except ImportError:
        paddle_ver = "not installed"

    try:
        import chandra
        chandra_ver = getattr(chandra, "__version__", "unknown")
    except ImportError:
        chandra_ver = "not installed"

    return InfoResponse(
        engines=["paddle", "chandra", "easyocr", "openocr", "vintern"],
        languages=["vi", "en"],
        paddle_version=paddle_ver,
        chandra_version=chandra_ver,
    )
