from __future__ import annotations


def evaluate_complexity(
    global_entropy: float,
    local_entropy_mean: float | None = None,
    local_entropy_std: float | None = None,
) -> tuple[str, str]:
    score = global_entropy
    if local_entropy_mean is not None:
        score = 0.7 * global_entropy + 0.3 * local_entropy_mean

    if score < 4.0:
        level = "低复杂度"
        description = "图像灰度分布较集中，纹理和细节较少。"
    elif score < 6.5:
        level = "中等复杂度"
        description = "图像包含一定纹理和灰度变化，视觉信息量适中。"
    else:
        level = "高复杂度"
        description = "图像灰度分布较分散，纹理、边缘或噪声较丰富。"

    if local_entropy_mean is not None:
        description += f" 局部熵均值为 {local_entropy_mean:.3f}。"
    if local_entropy_std is not None:
        description += f" 局部熵标准差为 {local_entropy_std:.3f}。"

    return level, description
