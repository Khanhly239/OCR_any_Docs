from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> "BoundingBox":
        return cls(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

    @classmethod
    def from_paddle(cls, points: list) -> "BoundingBox":
        """Chuyển từ PaddleOCR format [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]."""
        xs = [int(p[0]) for p in points]
        ys = [int(p[1]) for p in points]
        return cls(x=min(xs), y=min(ys), w=max(xs) - min(xs), h=max(ys) - min(ys))


class TextBlock(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    block_type: Literal["paragraph", "heading", "caption", "header", "footer"] = "paragraph"
    language: Optional[str] = None  # "vi" | "en" | None


class TableCell(BaseModel):
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    text: str
    bbox: Optional[BoundingBox] = None


class Table(BaseModel):
    bbox: BoundingBox
    rows: int
    cols: int
    cells: list[TableCell]

    def to_2d(self) -> list[list[str]]:
        """Chuyển sang ma trận 2D string để export."""
        grid: list[list[str]] = [[""] * self.cols for _ in range(self.rows)]
        for cell in self.cells:
            r, c = cell.row, cell.col
            if 0 <= r < self.rows and 0 <= c < self.cols:
                grid[r][c] = cell.text
        return grid


class PageResult(BaseModel):
    page_number: int
    width: int
    height: int
    text_blocks: list[TextBlock] = []
    tables: list[Table] = []
    engine_used: str
    processing_time_ms: float

    @property
    def full_text(self) -> str:
        """Toàn bộ text của trang theo reading order."""
        return "\n".join(b.text for b in self.text_blocks if b.text.strip())


class OCRResult(BaseModel):
    source_path: str
    doc_type: Literal["structured", "unstructured"]
    pages: list[PageResult] = []
    total_time_ms: float = 0.0
    metadata: dict = {}

    @property
    def full_text(self) -> str:
        parts = []
        for page in self.pages:
            parts.append(f"--- Trang {page.page_number} ---")
            parts.append(page.full_text)
        return "\n\n".join(parts)

    @property
    def all_tables(self) -> list[Table]:
        return [t for page in self.pages for t in page.tables]
