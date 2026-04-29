from __future__ import annotations

import numpy as np


def add_gaussian_noise(gray_image: np.ndarray, mean: float = 0.0, sigma: float = 20.0) -> np.ndarray:
    if sigma <= 0:
        return gray_image.copy()

    noise = np.random.normal(mean, sigma, gray_image.shape)
    noisy = gray_image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(gray_image: np.ndarray, amount: float = 0.02) -> np.ndarray:
    if amount <= 0:
        return gray_image.copy()

    noisy = gray_image.copy()
    num_pixels = gray_image.size

    num_salt = int(num_pixels * amount * 0.5)
    num_pepper = int(num_pixels * amount * 0.5)

    if num_salt > 0:
        coords_salt = (
            np.random.randint(0, gray_image.shape[0], num_salt),
            np.random.randint(0, gray_image.shape[1], num_salt),
        )
        noisy[coords_salt] = 255

    if num_pepper > 0:
        coords_pepper = (
            np.random.randint(0, gray_image.shape[0], num_pepper),
            np.random.randint(0, gray_image.shape[1], num_pepper),
        )
        noisy[coords_pepper] = 0

    return noisy
