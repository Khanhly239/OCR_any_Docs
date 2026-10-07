from typing import Literal
import cv2
import numpy as np


class Denoiser:
    def __init__(self, strength: Literal["light", "medium", "strong"] = "medium"):
        self.strength = strength

    def denoise(self, image: np.ndarray) -> np.ndarray:
        if self.strength == "light":
            return cv2.fastNlMeansDenoisingColored(image, None, h=5, hColor=5,
                                                    templateWindowSize=7, searchWindowSize=21)
        elif self.strength == "medium":
            denoised = cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10,
                                                        templateWindowSize=7, searchWindowSize=21)
            return cv2.bilateralFilter(denoised, d=5, sigmaColor=50, sigmaSpace=50)
        else:  # strong — dành cho scan chất lượng thấp
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Morphological opening để loại noise nhỏ
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            # Median blur
            blurred = cv2.medianBlur(opened, 3)
            return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
