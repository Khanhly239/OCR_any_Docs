from __future__ import annotations

import cv2
import numpy as np
import pytest

from ocrdoc.preprocessing.deskew import Deskewer
from ocrdoc.preprocessing.denoise import Denoiser
from ocrdoc.preprocessing.enhance import ContrastEnhancer


class TestDeskewer:
    def test_no_rotation_on_straight_image(self, white_image):
        deskewer = Deskewer()
        result = deskewer.deskew(white_image)
        assert result.shape[0] >= white_image.shape[0]

    def test_estimates_angle_close_to_zero_for_horizontal_lines(self):
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.line(img, (50, 200), (550, 200), (0, 0, 0), 2)
        cv2.line(img, (50, 250), (550, 250), (0, 0, 0), 2)
        deskewer = Deskewer()
        angle = deskewer.estimate_angle(img)
        assert abs(angle) < 2.0, f"Góc ước tính ({angle}°) nên gần 0"

    def test_deskew_skewed_image(self, skewed_image):
        deskewer = Deskewer()
        result = deskewer.deskew(skewed_image)
        # Ảnh sau deskew phải có angle gần 0
        corrected_angle = deskewer.estimate_angle(result)
        assert abs(corrected_angle) < 2.0, f"Angle sau deskew: {corrected_angle}°"

    def test_no_change_on_tiny_angle(self):
        img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        deskewer = Deskewer()
        # Ảnh không có lines → angle = 0 → không transform
        result = deskewer.deskew(img)
        assert result.shape == img.shape


class TestDenoiser:
    def test_light_denoising(self, noisy_image):
        denoiser = Denoiser(strength="light")
        result = denoiser.denoise(noisy_image)
        assert result.shape == noisy_image.shape
        assert result.dtype == np.uint8

    def test_medium_denoising(self, noisy_image):
        denoiser = Denoiser(strength="medium")
        result = denoiser.denoise(noisy_image)
        assert result.shape == noisy_image.shape

    def test_strong_denoising_returns_3_channel(self, noisy_image):
        denoiser = Denoiser(strength="strong")
        result = denoiser.denoise(noisy_image)
        assert result.ndim == 3

    def test_denoising_reduces_std(self, noisy_image):
        denoiser = Denoiser(strength="medium")
        result = denoiser.denoise(noisy_image)
        # Noise reduction → std deviation giảm
        original_std = float(np.std(noisy_image.astype(float)))
        result_std = float(np.std(result.astype(float)))
        assert result_std <= original_std


class TestContrastEnhancer:
    def test_enhance_color_image(self, white_image):
        enhancer = ContrastEnhancer()
        result = enhancer.enhance(white_image)
        assert result.shape == white_image.shape

    def test_enhance_grayscale(self):
        gray = np.ones((200, 300), dtype=np.uint8) * 50  # Dark
        enhancer = ContrastEnhancer()
        result = enhancer._enhance_gray(gray)
        assert result.ndim == 2

    def test_is_low_contrast_dark(self):
        dark = np.ones((200, 300, 3), dtype=np.uint8) * 30
        enhancer = ContrastEnhancer()
        assert enhancer.is_low_contrast(dark)

    def test_is_not_low_contrast_normal(self, invoice_image):
        enhancer = ContrastEnhancer()
        # Invoice image có contrast bình thường (lines đen trên nền trắng)
        assert not enhancer.is_low_contrast(invoice_image)
