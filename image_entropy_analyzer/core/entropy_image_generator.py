from __future__ import annotations

import cv2
import numpy as np

from core.entropy_calculator import calculate_global_entropy


def _blend_with_noise(gray_image: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.integers(0, 256, size=gray_image.shape, dtype=np.uint8)
    out = (1.0 - strength) * gray_image.astype(np.float32) + strength * noise.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _reduce_entropy(gray_image: np.ndarray, strength: float) -> np.ndarray:
    # strength: 0~1, bigger means lower entropy
    k = int(1 + round(strength * 10)) * 2 + 1
    blurred = cv2.GaussianBlur(gray_image, (k, k), 0)
    levels = int(np.clip(round(256 - strength * 224), 16, 256))
    step = max(1, 256 // levels)
    quantized = (blurred // step) * step
    return quantized.astype(np.uint8)


def generate_image_with_target_entropy(
    gray_image: np.ndarray,
    target_entropy: float,
    max_iter: int = 18,
    tolerance: float = 0.03,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    src = np.asarray(gray_image, dtype=np.uint8)
    base_entropy = calculate_global_entropy(src)
    rng = np.random.default_rng(seed)

    if abs(target_entropy - base_entropy) <= tolerance:
        return src.copy(), base_entropy

    if target_entropy > base_entropy:
        left, right = 0.0, 1.0
        best_img, best_entropy = src.copy(), base_entropy
        for _ in range(max_iter):
            mid = (left + right) / 2.0
            candidate = _blend_with_noise(src, mid, rng)
            cand_entropy = calculate_global_entropy(candidate)
            if abs(cand_entropy - target_entropy) < abs(best_entropy - target_entropy):
                best_img, best_entropy = candidate, cand_entropy
            if cand_entropy < target_entropy:
                left = mid
            else:
                right = mid
        return best_img, best_entropy

    left, right = 0.0, 1.0
    best_img, best_entropy = src.copy(), base_entropy
    for _ in range(max_iter):
        mid = (left + right) / 2.0
        candidate = _reduce_entropy(src, mid)
        cand_entropy = calculate_global_entropy(candidate)
        if abs(cand_entropy - target_entropy) < abs(best_entropy - target_entropy):
            best_img, best_entropy = candidate, cand_entropy
        if cand_entropy > target_entropy:
            left = mid
        else:
            right = mid
    return best_img, best_entropy
