from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger

from api.dependencies import get_orchestrator
from api.schemas import OCRResponse
from ocrdoc.orchestrator import OCROrchestrator

router = APIRouter(tags=["ocr"])

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".pdf", ".bmp", ".webp"}


def _validate_file(filename: str) -> None:
    from pathlib import Path
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Định dạng file không được hỗ trợ: '{ext}'. Hỗ trợ: {ALLOWED_EXTENSIONS}",
        )


@router.post("/ocr/process", response_model=OCRResponse)
async def process_document(
    file: Annotated[UploadFile, File(description="Image (.png/.jpg/.tiff) hoặc PDF")],
    mode: Annotated[str, Form()] = "auto",
    formats: Annotated[str, Form()] = "json,txt",
    lang: Annotated[str, Form()] = "vi,en",
    orchestrator: OCROrchestrator = Depends(get_orchestrator),
) -> OCRResponse:
    _validate_file(file.filename or "upload")
    file_bytes = await file.read()
    fmt_list = [f.strip() for f in formats.split(",") if f.strip()]
    lang_list = [l.strip() for l in lang.split(",") if l.strip()]

    try:
        outputs = orchestrator.process(
            source=file_bytes,
            filename=file.filename or "upload",
            mode=mode,  # type: ignore
            output_formats=fmt_list,
            lang=lang_list,
        )
    except Exception as e:
        logger.exception(f"OCR thất bại: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Lấy thông tin từ JSON output nếu có
    json_out = outputs.get("json", "{}")
    try:
        meta = json.loads(json_out) if isinstance(json_out, str) else {}
    except Exception:
        meta = {}

    return OCRResponse(
        status="ok",
        doc_type=meta.get("doc_type", "unknown"),
        engine_used=",".join(
            {p.get("engine_used", "") for p in meta.get("pages", [])}
        ),
        processing_time_ms=meta.get("total_time_ms", 0.0),
        page_count=len(meta.get("pages", [])),
        results={k: str(v) for k, v in outputs.items()},
    )


@router.post("/ocr/process/stream")
async def process_stream(
    file: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()] = "auto",
    formats: Annotated[str, Form()] = "json,txt",
    lang: Annotated[str, Form()] = "vi,en",
    orchestrator: OCROrchestrator = Depends(get_orchestrator),
) -> StreamingResponse:
    """Server-Sent Events streaming cho PDF nhiều trang."""
    _validate_file(file.filename or "upload")
    file_bytes = await file.read()
    fmt_list = [f.strip() for f in formats.split(",") if f.strip()]
    lang_list = [l.strip() for l in lang.split(",") if l.strip()]

    def event_generator():
        try:
            for event in orchestrator.process_and_stream(
                source=file_bytes,
                filename=file.filename or "upload",
                mode=mode,  # type: ignore
                output_formats=fmt_list,
                lang=lang_list,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception(f"Stream OCR thất bại: {e}")
            yield f"data: {json.dumps({'event': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
