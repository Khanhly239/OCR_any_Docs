from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from ocrdoc.core.models import BoundingBox, OCRResult, PageResult, Table, TableCell, TextBlock

SAMPLES_DIR = Path(__file__).parent / "samples"


# ── Fixtures: synthetic images ───────────────────────────────────────────────

@pytest.fixture
def white_image() -> np.ndarray:
    return np.ones((300, 400, 3), dtype=np.uint8) * 255


@pytest.fixture
def invoice_image() -> np.ndarray:
    """Ảnh giả lập invoice với các đường kẻ ngang và dọc."""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    import cv2
    # Vẽ 5 đường kẻ ngang (giả lập rows của invoice)
    for y in [100, 200, 300, 400, 500]:
        cv2.line(img, (50, y), (750, y), (0, 0, 0), 2)
    # Vẽ 3 đường kẻ dọc (giả lập columns)
    for x in [50, 400, 750]:
        cv2.line(img, (x, 100), (x, 500), (0, 0, 0), 2)
    return img


@pytest.fixture
def skewed_image() -> np.ndarray:
    """Ảnh bị nghiêng 5°."""
    import cv2
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.line(img, (50, 200), (550, 200), (0, 0, 0), 2)
    cv2.line(img, (50, 250), (550, 250), (0, 0, 0), 2)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 5, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))


@pytest.fixture
def noisy_image() -> np.ndarray:
    """Ảnh có nhiều noise."""
    img = np.ones((300, 400, 3), dtype=np.uint8) * 200
    noise = np.random.randint(0, 60, img.shape, dtype=np.uint8)
    return np.clip(img.astype(int) - noise, 0, 255).astype(np.uint8)


# ── Fixtures: mock OCR result ─────────────────────────────────────────────────

@pytest.fixture
def sample_text_blocks() -> list[TextBlock]:
    return [
        TextBlock(
            text="Công ty TNHH ABC",
            confidence=0.98,
            bbox=BoundingBox(x=10, y=10, w=200, h=30),
            block_type="heading",
        ),
        TextBlock(
            text="Địa chỉ: 123 Nguyễn Huệ, TP.HCM",
            confidence=0.95,
            bbox=BoundingBox(x=10, y=50, w=300, h=25),
            block_type="paragraph",
        ),
        TextBlock(
            text="Số điện thoại: 0901234567",
            confidence=0.97,
            bbox=BoundingBox(x=10, y=80, w=200, h=25),
            block_type="paragraph",
        ),
    ]


@pytest.fixture
def sample_table() -> Table:
    cells = [
        TableCell(row=0, col=0, text="STT"),
        TableCell(row=0, col=1, text="Tên hàng"),
        TableCell(row=0, col=2, text="Số lượng"),
        TableCell(row=0, col=3, text="Đơn giá"),
        TableCell(row=1, col=0, text="1"),
        TableCell(row=1, col=1, text="Sản phẩm A"),
        TableCell(row=1, col=2, text="10"),
        TableCell(row=1, col=3, text="50,000"),
        TableCell(row=2, col=0, text="2"),
        TableCell(row=2, col=1, text="Sản phẩm B"),
        TableCell(row=2, col=2, text="5"),
        TableCell(row=2, col=3, text="120,000"),
    ]
    return Table(
        bbox=BoundingBox(x=0, y=100, w=600, h=200),
        rows=3,
        cols=4,
        cells=cells,
    )


@pytest.fixture
def sample_ocr_result(sample_text_blocks, sample_table) -> OCRResult:
    page = PageResult(
        page_number=1,
        width=800,
        height=600,
        text_blocks=sample_text_blocks,
        tables=[sample_table],
        engine_used="paddle",
        processing_time_ms=500.0,
    )
    return OCRResult(
        source_path="test.png",
        doc_type="structured",
        pages=[page],
        total_time_ms=500.0,
    )
