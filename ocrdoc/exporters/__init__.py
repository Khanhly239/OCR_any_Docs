from __future__ import annotations

from ocrdoc.exporters.base_exporter import BaseExporter
from ocrdoc.exporters.json_exporter import JSONExporter
from ocrdoc.exporters.markdown_exporter import MarkdownExporter
from ocrdoc.exporters.text_exporter import TextExporter
from ocrdoc.exporters.excel_exporter import ExcelExporter


def get_exporter(fmt: str) -> BaseExporter:
    exporters: dict[str, BaseExporter] = {
        "json": JSONExporter(),
        "markdown": MarkdownExporter(),
        "md": MarkdownExporter(),
        "txt": TextExporter(),
        "text": TextExporter(),
        "xlsx": ExcelExporter(),
        "excel": ExcelExporter(),
    }
    key = fmt.lower().strip()
    if key not in exporters:
        raise ValueError(f"Format không được hỗ trợ: '{fmt}'. Chọn: {list(exporters)}")
    return exporters[key]
