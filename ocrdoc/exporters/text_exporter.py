from __future__ import annotations

from ocrdoc.core.models import OCRResult, PageResult, Table
from ocrdoc.exporters.base_exporter import BaseExporter


class TextExporter(BaseExporter):
    def export(self, result: OCRResult) -> str:
        parts: list[str] = []
        for page in result.pages:
            if len(result.pages) > 1:
                parts.append(f"\n{'='*60}\nTRANG {page.page_number}\n{'='*60}\n")
            parts.append(self._render_page(page))
        return "\n\n".join(parts).strip()

    def _render_page(self, page: PageResult) -> str:
        items: list[tuple[int, str]] = []

        for block in page.text_blocks:
            if block.text.strip():
                items.append((block.bbox.y, block.text.strip()))

        for table in page.tables:
            items.append((table.bbox.y, self._render_table(table)))

        items.sort(key=lambda x: x[0])
        return "\n\n".join(content for _, content in items)

    def _render_table(self, table: Table) -> str:
        grid = table.to_2d()
        if not grid:
            return ""
        rows = []
        for row in grid:
            rows.append("\t".join(c.replace("\n", " ") for c in row))
        return "\n".join(rows)

    @property
    def mime_type(self) -> str:
        return "text/plain"

    @property
    def file_extension(self) -> str:
        return "txt"
