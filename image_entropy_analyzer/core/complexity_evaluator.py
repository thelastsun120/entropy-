"""图像复杂度评估：融合熵特征与结构特征。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


EPS = 1e-12


@dataclass
class ComplexityFeatures:
    global_entropy: float
    local_entropy_mean: float
    sobel_energy: float
    laplacian_variance: float
    edge_ratio: float
    structure_score: float
    final_score: float
    noise_suppressed: bool


def _to_gray(image: np.ndarray) -> np.ndarray:
    """将输入图像统一为 [0, 255] 的灰度浮点图。"""
    arr = np.asarray(image)
    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            gray = arr[..., 0].astype(np.float32)
        else:
            # ITU-R BT.601
            gray = (
                0.299 * arr[..., 0]
                + 0.587 * arr[..., 1]
                + 0.114 * arr[..., 2]
            ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if gray.max(initial=0.0) <= 1.0:
        gray = gray * 255.0
    return np.clip(gray, 0.0, 255.0)


def _shannon_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=False)
    prob = hist.astype(np.float64)
    prob /= prob.sum() + EPS
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob + EPS)).sum())


def _local_entropy_mean(gray: np.ndarray, patch_size: int = 16) -> float:
    h, w = gray.shape
    if h < patch_size or w < patch_size:
        return _shannon_entropy(gray)

    entropies = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = gray[y : y + patch_size, x : x + patch_size]
            entropies.append(_shannon_entropy(patch))

    return float(np.mean(entropies)) if entropies else _shannon_entropy(gray)


def _sobel_energy(gray: np.ndarray) -> float:
    gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    gy = gx.T
    px = np.pad(gray, 1, mode="reflect")

    grad_x = np.zeros_like(gray, dtype=np.float32)
    grad_y = np.zeros_like(gray, dtype=np.float32)
    for i in range(3):
        for j in range(3):
            grad_x += gx[i, j] * px[i : i + gray.shape[0], j : j + gray.shape[1]]
            grad_y += gy[i, j] * px[i : i + gray.shape[0], j : j + gray.shape[1]]

    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return float(np.mean(grad_mag**2))


def _laplacian_variance(gray: np.ndarray) -> float:
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    padded = np.pad(gray, 1, mode="reflect")
    lap = np.zeros_like(gray, dtype=np.float32)
    for i in range(3):
        for j in range(3):
            lap += kernel[i, j] * padded[i : i + gray.shape[0], j : j + gray.shape[1]]
    return float(np.var(lap))


def _edge_ratio(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx**2 + gy**2)
    threshold = np.percentile(grad, 75)
    return float((grad > threshold).mean())


def _normalize(value: float, scale: float) -> float:
    # 饱和归一化，避免单指标过大主导评分
    return float(value / (value + scale + EPS))


def _build_result_text(feat: ComplexityFeatures) -> str:
    lines = [
        f"全局熵: {feat.global_entropy:.3f}",
        f"局部熵均值: {feat.local_entropy_mean:.3f}",
        f"结构分数: {feat.structure_score:.3f}",
        f"Sobel梯度能量: {feat.sobel_energy:.3f}",
        f"Laplacian方差: {feat.laplacian_variance:.3f}",
        f"边缘像素占比: {feat.edge_ratio:.3f}",
        f"综合复杂度得分: {feat.final_score:.3f}",
    ]
    if feat.noise_suppressed:
        lines.append("提示: 随机噪声占比高（高熵但结构一致性较弱），已进行阈值抑制。")
    return "\n".join(lines)


def evaluate_complexity(image: np.ndarray) -> Dict[str, Any]:
    """评估图像复杂度并输出可解释指标。"""
    gray = _to_gray(image)

    global_entropy = _shannon_entropy(gray)
    local_entropy_mean = _local_entropy_mean(gray)

    sobel_energy = _sobel_energy(gray)
    lap_var = _laplacian_variance(gray)
    edge_ratio = _edge_ratio(gray)

    sobel_norm = _normalize(sobel_energy, 5000.0)
    lap_norm = _normalize(lap_var, 3000.0)
    edge_norm = np.clip(edge_ratio / 0.35, 0.0, 1.0)
    structure_score = float(0.5 * sobel_norm + 0.3 * lap_norm + 0.2 * edge_norm)

    # 融合评分（用户指定权重）
    global_entropy_norm = np.clip(global_entropy / 8.0, 0.0, 1.0)
    local_entropy_norm = np.clip(local_entropy_mean / 8.0, 0.0, 1.0)
    score = float(
        0.6 * global_entropy_norm + 0.2 * local_entropy_norm + 0.2 * structure_score
    )

    # 对纯噪声倾向图像进行抑制：高熵 + 低结构分数 + 局部熵非常均匀
    noise_suppressed = False
    if global_entropy_norm > 0.90 and structure_score < 0.35 and local_entropy_norm > 0.88:
        score *= 0.85
        noise_suppressed = True

    features = ComplexityFeatures(
        global_entropy=global_entropy,
        local_entropy_mean=local_entropy_mean,
        sobel_energy=sobel_energy,
        laplacian_variance=lap_var,
        edge_ratio=edge_ratio,
        structure_score=structure_score,
        final_score=score,
        noise_suppressed=noise_suppressed,
    )

    return {
        "score": score,
        "global_entropy": global_entropy,
        "local_entropy_mean": local_entropy_mean,
        "structure_score": structure_score,
        "sobel_energy": sobel_energy,
        "laplacian_variance": lap_var,
        "edge_ratio": edge_ratio,
        "noise_suppressed": noise_suppressed,
        "result_text": _build_result_text(features),
    }
