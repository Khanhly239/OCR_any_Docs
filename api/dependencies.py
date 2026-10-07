from __future__ import annotations

from fastapi import Request

from ocrdoc.orchestrator import OCROrchestrator


def get_orchestrator(request: Request) -> OCROrchestrator:
    return request.app.state.orchestrator
