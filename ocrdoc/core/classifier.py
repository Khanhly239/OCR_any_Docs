from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from loguru import logger


class DocumentClassifier:
    """
    Phân loại tài liệu dựa trên đặc trưng hình ảnh (không dùng ML, chạy CPU <50ms).
    "structured"   → forms, invoices, tables (nhiều đường kẻ và ô chữ nhật)
    "unstructured" → report, luận văn, scan tự do
    """

    def __init__(
        self,
        line_ratio_threshold: float = 0.12,
        rect_count_threshold: int = 8,
    ):
        self.line_ratio_threshold = line_ratio_threshold
        self.rect_count_threshold = rect_count_threshold

    def classify(self, image: np.ndarray) -> Literal["structured", "unstructured"]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h_ratio, v_ratio = self._line_density(gray)
        rect_count = self._rectangle_count(gray)

        is_structured = (
            h_ratio > self.line_ratio_threshold
            or v_ratio > self.line_ratio_threshold
            or rect_count >= self.rect_count_threshold
        )
        result: Literal["structured", "unstructured"] = "structured" if is_structured else "unstructured"
        logger.debug(
            f"Classifier: h_lines={h_ratio:.3f}, v_lines={v_ratio:.3f}, "
            f"rects={rect_count} → {result}"
        )
        return result

    def _line_density(self, gray: np.ndarray) -> tuple[float, float]:
        h, w = gray.shape
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        lines = cv2.HoughLinesP(
            binary, 1, np.pi / 180,
            threshold=60, minLineLength=max(50, w // 8), maxLineGap=20,
        )
        if lines is None:
            return 0.0, 0.0

        h_count = v_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 10 or angle > 170:
                h_count += 1
            elif 80 < angle < 100:
                v_count += 1

        total = max(len(lines), 1)
        return h_count / total, v_count / total

    def _rectangle_count(self, gray: np.ndarray) -> int:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # Morphological close để nối các đường đứt đoạn
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        img_area = gray.shape[0] * gray.shape[1]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Bỏ qua contour quá nhỏ (noise) hoặc quá lớn (toàn trang)
            if area < 500 or area > img_area * 0.5:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                count += 1
        return count
