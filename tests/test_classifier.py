from __future__ import annotations

import numpy as np
import pytest

from ocrdoc.core.classifier import DocumentClassifier


class TestDocumentClassifier:
    def test_structured_invoice_image(self, invoice_image):
        classifier = DocumentClassifier()
        result = classifier.classify(invoice_image)
        assert result == "structured", f"Invoice image nên là 'structured', nhận được '{result}'"

    def test_unstructured_plain_image(self, white_image):
        classifier = DocumentClassifier()
        result = classifier.classify(white_image)
        assert result == "unstructured", "Ảnh trắng nên là 'unstructured'"

    def test_structured_detection_with_rectangles(self):
        import cv2
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        # Vẽ nhiều ô chữ nhật (giả lập form fields)
        for i in range(5):
            x, y = 50 + i * 100, 100
            cv2.rectangle(img, (x, y), (x + 80, y + 40), (0, 0, 0), 2)
            cv2.rectangle(img, (x, y + 60), (x + 80, y + 100), (0, 0, 0), 2)

        classifier = DocumentClassifier()
        result = classifier.classify(img)
        assert result == "structured"

    def test_classify_returns_valid_type(self, invoice_image):
        classifier = DocumentClassifier()
        result = classifier.classify(invoice_image)
        assert result in ("structured", "unstructured")

    def test_line_density_horizontal_dominant(self):
        import cv2
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        # Vẽ nhiều đường ngang
        for y in range(50, 400, 30):
            cv2.line(img, (10, y), (590, y), (0, 0, 0), 1)

        classifier = DocumentClassifier()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h_ratio, v_ratio = classifier._line_density(gray)
        assert h_ratio > 0.1, f"h_ratio={h_ratio} nên > 0.1 với nhiều đường ngang"
