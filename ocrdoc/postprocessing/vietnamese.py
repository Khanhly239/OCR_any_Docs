from __future__ import annotations

import unicodedata
import regex

from loguru import logger


# Bảng các ký tự tiếng Việt hợp lệ (NFC)
_VI_VOWELS_WITH_TONE = frozenset(
    "àáâãăặắằẳẵấầẩẫảạ"
    "èéêếềểễẹẻẽ"
    "ìíịỉĩ"
    "òóôõơởớờởỡọỏố"  # noqa
    "ùúũưựứừửữụủ"
    "ỳýỵỷỹ"
    "đĐ"
    "ÀÁÂÃĂẶẮẰẲẴẤẦẨẪẢẠ"
    "ÈÉÊẾỀỂỄẸẺẼ"
    "ÌÍỊỈĨ"
    "ÒÓÔÕƠỞỚỜỞỠỌỎỐ"
    "ÙÚŨƯỰỨỪỬỮỤỦ"
    "ỲÝỴỶỸ"
)

# Các lỗi OCR phổ biến (PaddleOCR & Surya) cho tiếng Việt
_COMMON_CORRECTIONS: dict[str, str] = {
    # Dấu đặc thù bị mất
    "đ": "đ",   # đ (Latin small letter d with stroke) — đôi khi bị drop
    "d̀": "đ",  # d + combining grave → nhầm với đ
}

# Regex phát hiện hai dấu thanh liên tiếp trên cùng một âm tiết
_DOUBLE_TONE_PATTERN = regex.compile(
    r"[̣̀́̃̉]{2,}",  # hai combining tone marks
    regex.UNICODE,
)


class VietnameseNormalizer:
    def normalize(self, text: str) -> str:
        if not text:
            return text

        # 1. NFC normalization — quan trọng nhất
        text = unicodedata.normalize("NFC", text)

        # 2. Sửa lỗi phổ biến
        for wrong, correct in _COMMON_CORRECTIONS.items():
            text = text.replace(wrong, correct)

        # 3. Loại bỏ double tone marks (giữ cái đầu tiên)
        text = _DOUBLE_TONE_PATTERN.sub(lambda m: m.group()[0], text)

        # 4. Chuẩn hóa khoảng trắng
        text = regex.sub(r"[ \t]+", " ", text)
        text = regex.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def normalize_block(self, text: str) -> str:
        """Normalize một TextBlock, giữ nguyên newlines trong block."""
        lines = text.split("\n")
        return "\n".join(self.normalize(line) for line in lines)
