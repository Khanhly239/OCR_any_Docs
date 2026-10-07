from __future__ import annotations

import json

from ocrdoc.core.models import OCRResult
from ocrdoc.exporters.base_exporter import BaseExporter


class JSONExporter(BaseExporter):
    def __init__(self, indent: int = 2, include_bboxes: bool = True):
        self.indent = indent
        self.include_bboxes = include_bboxes

    def export(self, result: OCRResult) -> str:
        data = result.model_dump()
        if not self.include_bboxes:
            for page in data.get("pages", []):
                for block in page.get("text_blocks", []):
                    block.pop("bbox", None)
                for table in page.get("tables", []):
                    table.pop("bbox", None)
                    for cell in table.get("cells", []):
                        cell.pop("bbox", None)
        return json.dumps(data, ensure_ascii=False, indent=self.indent)

    @property
    def mime_type(self) -> str:
        return "application/json"

    @property
    def file_extension(self) -> str:
        return "json"
