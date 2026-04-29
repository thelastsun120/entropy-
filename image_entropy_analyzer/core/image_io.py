from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path) -> np.ndarray | None:
    """Read image with Unicode-safe fallback for Windows-like paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_image_as_rgb_and_gray(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        image_bgr = imread_unicode(path)

    if image_bgr is None:
        raise ValueError("图像读取失败，请检查文件格式")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return image_rgb, gray.astype(np.uint8)
