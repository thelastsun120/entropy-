from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class LLMComplexityResult:
    level: str
    description: str
    model: str


def _build_prompt(
    global_entropy: float,
    local_entropy_mean: float | None,
    local_entropy_std: float | None,
    noisy_entropy: float | None,
    entropy_delta: float | None,
) -> str:
    return (
        "你是图像复杂度分析助手。请根据以下统计量判断视觉复杂度等级，仅返回 JSON："
        '{"level":"低复杂度|中等复杂度|高复杂度","description":"简洁中文结论"}。'
        f"全局熵={global_entropy:.4f} bit；"
        f"局部熵均值={local_entropy_mean if local_entropy_mean is not None else 'N/A'}；"
        f"局部熵标准差={local_entropy_std if local_entropy_std is not None else 'N/A'}；"
        f"加噪后熵={noisy_entropy if noisy_entropy is not None else 'N/A'}；"
        f"熵变化={entropy_delta if entropy_delta is not None else 'N/A'}。"
        "规则参考：<4低，4-6.5中，>=6.5高；若噪声导致熵升高，要提示可能是随机性而非有效信息。"
    )


def evaluate_complexity_with_llm(
    global_entropy: float,
    local_entropy_mean: float | None = None,
    local_entropy_std: float | None = None,
    noisy_entropy: float | None = None,
    entropy_delta: float | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> LLMComplexityResult:
    effective_api_key = api_key or os.getenv("OPENAI_API_KEY")
    model = model_name or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    effective_base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    if not effective_api_key:
        raise RuntimeError("未配置 API Key，无法使用大语言模型评级。")

    prompt = _build_prompt(global_entropy, local_entropy_mean, local_entropy_std, noisy_entropy, entropy_delta)

    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "complexity_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["level", "description"],
                    "additionalProperties": False,
                },
            }
        },
    }

    req = Request(
        f"{effective_base_url.rstrip('/')}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {effective_api_key}",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM 请求失败: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"LLM 网络错误: {exc.reason}") from exc

    data = json.loads(raw)
    output_text = data.get("output_text", "").strip()
    if not output_text:
        raise RuntimeError("LLM 未返回有效文本。")

    result = json.loads(output_text)
    level = result.get("level", "中等复杂度")
    description = result.get("description", "模型未返回说明。")
    return LLMComplexityResult(level=level, description=description, model=model)
