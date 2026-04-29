from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel


class ImagePanel(QLabel):
    def __init__(self, title: str = "Image") -> None:
        super().__init__()
        self._base_text = title
        self._pixmap: QPixmap | None = None

        self.setAlignment(Qt.AlignCenter)
        self.setText(title)
        self.setMinimumSize(320, 240)
        self.setStyleSheet(
            """
            QLabel {
                border: 1px solid #AAAAAA;
                background-color: #F8F8F8;
                color: #444444;
                font-size: 14px;
            }
            """
        )

    def set_image(self, image: np.ndarray) -> None:
        if image is None:
            return

        if image.ndim == 2:
            h, w = image.shape
            qimg = QImage(image.data, w, h, image.strides[0], QImage.Format_Grayscale8)
        elif image.ndim == 3 and image.shape[2] == 3:
            h, w, _ = image.shape
            qimg = QImage(image.data, w, h, image.strides[0], QImage.Format_RGB888)
        else:
            raise ValueError("Unsupported image format")

        qimg = qimg.copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._refresh_pixmap()

    def clear_panel(self) -> None:
        self._pixmap = None
        self.clear()
        self.setText(self._base_text)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
