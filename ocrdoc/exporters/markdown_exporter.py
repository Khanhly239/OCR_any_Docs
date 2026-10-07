from __future__ import annotations

from ocrdoc.core.models import OCRResult, PageResult, Table, TextBlock
from ocrdoc.exporters.base_exporter import BaseExporter


class MarkdownExporter(BaseExporter):
    def export(self, result: OCRResult) -> str:
        parts: list[str] = []
        for page in result.pages:
            if len(result.pages) > 1:
                parts.append(f"\n---\n## Trang {page.page_number}\n")
            parts.append(self._render_page(page))
        return "\n".join(parts).strip()

    def _render_page(self, page: PageResult) -> str:
        # Gộp text_blocks và tables theo vị trí y để giữ thứ tự tự nhiên
        items: list[tuple[int, str]] = []  # (y, content)

        for block in page.text_blocks:
            rendered = self._render_block(block)
            if rendered:
                items.append((block.bbox.y, rendered))

        for table in page.tables:
            rendered = self._render_table(table)
            if rendered:
                items.append((table.bbox.y, rendered))

        items.sort(key=lambda x: x[0])
        return "\n\n".join(content for _, content in items)

    def _render_block(self, block: TextBlock) -> str:
        text = block.text.strip()
        if not text:
            return ""
        if block.block_type == "heading":
            return f"## {text}"
        elif block.block_type == "header":
            return f"**{text}**"
        elif block.block_type == "footer":
            return f"*{text}*"
        elif block.block_type == "caption":
            return f"*{text}*"
        return text

    def _render_table(self, table: Table) -> str:
        if not table.cells:
            return ""
        grid = table.to_2d()
        if not grid:
            return ""

        lines: list[str] = []
        for i, row in enumerate(grid):
            cells = [c.replace("|", "\\|").replace("\n", " ") for c in row]
            lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                # Separator sau header row
                lines.append("|" + "|".join(["---"] * len(row)) + "|")

        return "\n".join(lines)

    @property
    def mime_type(self) -> str:
        return "text/markdown"

    @property
    def file_extension(self) -> str:
        return "md"
