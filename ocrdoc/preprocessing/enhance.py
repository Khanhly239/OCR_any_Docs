import cv2
import numpy as np


class ContrastEnhancer:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return self._enhance_gray(image)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def _enhance_gray(self, gray: np.ndarray) -> np.ndarray:
        enhanced = self.clahe.apply(gray)
        # Nếu ảnh có nền đen (trung bình tối), dùng Otsu threshold
        mean_val = np.mean(enhanced)
        if mean_val < 127:
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            return thresh
        return enhanced

    def is_low_contrast(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return float(gray.std()) < 30.0
