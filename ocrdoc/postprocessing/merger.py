from __future__ import annotations

from ocrdoc.core.models import OCRResult, PageResult


class ResultMerger:
    """Gộp nhiều PageResult (từ multi-page PDF) thành OCRResult."""

    def merge(
        self,
        pages: list[PageResult],
        source_path: str,
        doc_type: str,
        total_time_ms: float,
    ) -> OCRResult:
        # Gán lại page_number theo thứ tự thực
        numbered_pages = []
        for i, page in enumerate(pages, start=1):
            numbered_pages.append(page.model_copy(update={"page_number": i}))

        return OCRResult(
            source_path=source_path,
            doc_type=doc_type,
            pages=numbered_pages,
            total_time_ms=total_time_ms,
            metadata={
                "page_count": len(numbered_pages),
                "total_tables": sum(len(p.tables) for p in numbered_pages),
                "total_text_blocks": sum(len(p.text_blocks) for p in numbered_pages),
                "engines_used": list({p.engine_used for p in numbered_pages}),
            },
        )
