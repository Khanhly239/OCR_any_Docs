from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routers import health as health_router
from api.routers import ocr as ocr_router
from config.settings import settings
from ocrdoc.orchestrator import OCROrchestrator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("OCRdoc server đang khởi động...")
    orchestrator = OCROrchestrator(settings)
    orchestrator.startup()
    app.state.orchestrator = orchestrator

    # Mount Gradio UI
    try:
        import gradio as gr
        from ui.gradio_app import build_ui
        gr.mount_gradio_app(app, build_ui(orchestrator), path="/ui")
        logger.info("Gradio UI: http://localhost:{}/ui", settings.api_port)
    except Exception as e:
        logger.warning(f"Gradio UI không khởi động được: {e}")

    yield

    logger.info("OCRdoc server đang tắt...")
    orchestrator.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="OCRdoc",
        version="1.0.0",
        description="Vietnamese + English local OCR API",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router.router, prefix="/api/v1")
    app.include_router(ocr_router.router, prefix="/api/v1")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=1,  # BẮT BUỘC: VRAMManager là singleton trong 1 process
        reload=False,
    )
