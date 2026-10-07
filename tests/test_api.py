from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ocrdoc.core.models import BoundingBox, OCRResult, PageResult, TextBlock


@pytest.fixture
def mock_orchestrator():
    """Mock OCROrchestrator để test API mà không cần GPU."""
    orchestrator = MagicMock()
    orchestrator.process.return_value = {
        "json": '{"source_path":"test.png","doc_type":"structured","pages":[{"page_number":1,"width":800,"height":600,"text_blocks":[{"text":"Test","confidence":0.95,"bbox":{"x":0,"y":0,"w":100,"h":30},"block_type":"paragraph","language":null}],"tables":[],"engine_used":"paddle","processing_time_ms":500.0}],"total_time_ms":500.0,"metadata":{"page_count":1}}',
        "txt": "Test",
    }
    return orchestrator


@pytest.fixture
def client(mock_orchestrator):
    from api.main import create_app
    # Patch OCROrchestrator trong api.main để lifespan dùng mock thay vì real model
    with patch("api.main.OCROrchestrator", return_value=mock_orchestrator):
        app = create_app()
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "gpu_available" in data
        assert "vram_free_mb" in data

    def test_info_endpoint(self, client):
        response = client.get("/api/v1/info")
        assert response.status_code == 200
        data = response.json()
        assert "engines" in data
        assert "paddle" in data["engines"]
        assert "chandra" in data["engines"]


class TestOCREndpoint:
    def test_process_png(self, client, mock_orchestrator):
        # Tạo một ảnh PNG nhỏ để test
        img_bytes = _create_dummy_png()
        response = client.post(
            "/api/v1/ocr/process",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"mode": "auto", "formats": "json,txt"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_invalid_file_type_rejected(self, client):
        response = client.post(
            "/api/v1/ocr/process",
            files={"file": ("test.docx", b"fake content", "application/msword")},
        )
        assert response.status_code == 422

    def test_process_with_structured_mode(self, client, mock_orchestrator):
        img_bytes = _create_dummy_png()
        response = client.post(
            "/api/v1/ocr/process",
            files={"file": ("invoice.png", img_bytes, "image/png")},
            data={"mode": "structured", "formats": "json"},
        )
        assert response.status_code == 200
        mock_orchestrator.process.assert_called_once()
        call_kwargs = mock_orchestrator.process.call_args[1]
        assert call_kwargs.get("mode") == "structured"


def _create_dummy_png() -> bytes:
    """Tạo PNG 1x1 pixel để test upload."""
    import io
    from PIL import Image
    img = Image.new("RGB", (10, 10), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
