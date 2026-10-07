from __future__ import annotations

from ocrdoc.core.models import TextBlock


class ReadingOrderSorter:
    """
    Sắp xếp TextBlocks theo reading order tự nhiên (trái-phải, trên-xuống).
    Được dùng khi engine không cung cấp thứ tự đọc (ví dụ PaddleOCR).
    """

    def __init__(self, column_gap_ratio: float = 0.05):
        # Ngưỡng để phát hiện layout 2 cột (tỉ lệ so với chiều rộng trang)
        self.column_gap_ratio = column_gap_ratio

    def sort(self, blocks: list[TextBlock], page_width: int = 0) -> list[TextBlock]:
        if not blocks:
            return blocks

        # Phát hiện layout 2 cột
        if page_width > 0 and self._is_two_column(blocks, page_width):
            return self._sort_two_column(blocks, page_width)

        # Layout 1 cột: sort theo y trước, x sau
        return sorted(blocks, key=lambda b: (b.bbox.y, b.bbox.x))

    def _is_two_column(self, blocks: list[TextBlock], page_width: int) -> bool:
        """Phát hiện layout 2 cột bằng cách tìm khoảng trống dọc ở giữa trang."""
        mid = page_width // 2
        gap = int(page_width * self.column_gap_ratio)

        # Đếm blocks ở trái và phải mid
        left = sum(1 for b in blocks if b.bbox.x2 < mid - gap)
        right = sum(1 for b in blocks if b.bbox.x > mid + gap)
        total = len(blocks)

        # 2 cột nếu cả 2 phía đều có ít nhất 20% số blocks
        return total > 4 and left / total > 0.2 and right / total > 0.2

    def _sort_two_column(self, blocks: list[TextBlock], page_width: int) -> list[TextBlock]:
        mid = page_width // 2
        left_col = [b for b in blocks if b.bbox.x + b.bbox.w // 2 < mid]
        right_col = [b for b in blocks if b.bbox.x + b.bbox.w // 2 >= mid]

        left_sorted = sorted(left_col, key=lambda b: (b.bbox.y, b.bbox.x))
        right_sorted = sorted(right_col, key=lambda b: (b.bbox.y, b.bbox.x))

        return left_sorted + right_sorted
