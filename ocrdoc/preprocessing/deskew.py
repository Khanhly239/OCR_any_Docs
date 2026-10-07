import cv2
import numpy as np
from loguru import logger


class Deskewer:
    def __init__(self, max_angle_deg: float = 15.0):
        self.max_angle_deg = max_angle_deg

    def estimate_angle(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        lines = cv2.HoughLinesP(
            binary, 1, np.pi / 180,
            threshold=80, minLineLength=100, maxLineGap=10
        )
        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Chỉ lấy các đường gần ngang (±max_angle_deg)
            if abs(angle) <= self.max_angle_deg:
                angles.append(angle)

        if not angles:
            return 0.0
        return float(np.median(angles))

    def deskew(self, image: np.ndarray) -> np.ndarray:
        angle = self.estimate_angle(image)
        if abs(angle) < 0.3:
            return image

        logger.debug(f"Deskew: xoay {angle:.2f}°")
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Tính kích thước ảnh sau khi xoay để không cắt xén nội dung
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(
            image, M, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if image.ndim == 3 else 255,
        )
        return rotated
