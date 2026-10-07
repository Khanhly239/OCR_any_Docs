from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class OCRRequest(BaseModel):
    mode: Literal["auto", "structured", "unstructured"] = "auto"
    formats: list[str] = Field(default=["json", "txt"])
    lang: list[str] = Field(default=["vi", "en"])


class OCRResponse(BaseModel):
    status: str = "ok"
    doc_type: str
    engine_used: str
    processing_time_ms: float
    page_count: int
    results: dict[str, str]  # format → content (xlsx là base64)
    warnings: list[str] = []


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    vram_free_mb: int


class InfoResponse(BaseModel):
    engines: list[str]
    languages: list[str]
    paddle_version: str
    chandra_version: str
