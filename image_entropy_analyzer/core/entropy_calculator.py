from __future__ import annotations

import numpy as np
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte


def calculate_histogram(gray_image: np.ndarray) -> np.ndarray:
    gray_u8 = np.asarray(gray_image, dtype=np.uint8)
    return np.bincount(gray_u8.flatten(), minlength=256)


def calculate_global_entropy(gray_image: np.ndarray) -> float:
    hist = calculate_histogram(gray_image)
    prob = hist / gray_image.size
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)


def calculate_local_entropy(gray_image: np.ndarray, radius: int = 5) -> np.ndarray:
    gray_u8 = img_as_ubyte(np.asarray(gray_image, dtype=np.uint8))
    footprint = disk(radius)
    return rank_entropy(gray_u8, footprint)
