"""Gemini 文本/多模态结构化分析工具。

用法：
    from src.tools.gemini_text import analyze_text, analyze_multimodal

    result = analyze_text(prompt="...", schema=MyModel)
    result = analyze_multimodal(contents=[...], schema=MyModel)
"""

import logging
import time
from typing import TypeVar

from google.genai.types import GenerateContentConfig, Part
from pydantic import BaseModel

from src.tools.models.registry import get_genai_client

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# 模型 fallback 映射
_MODEL_FALLBACKS: dict[str, list[str]] = {
    "gemini-3-pro-preview": ["gemini-3-flash-preview"],
    "gemini-3-flash-preview": ["gemini-3-pro-preview"],
}


def _try_model(
    model: str,
    contents,
    system_instruction: str,
    schema: type[T],
    max_retries: int,
    retry_delay: float,
    func_name: str,
) -> T:
    """用指定模型尝试生成，失败时抛出最后的异常。"""
    last_error: Exception | None = None
    delay = retry_delay
    for attempt in range(1, max_retries + 1):
        client = get_genai_client(timeout=180_000)
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=GenerateContentConfig(
                    system_instruction=system_instruction or None,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            return schema.model_validate_json(resp.text)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(f"{func_name} [{model}] 重试 {attempt}/{max_retries}: {e}")
                time.sleep(delay)
                delay = min(delay * 2, 30)
    raise last_error  # type: ignore[misc]


def analyze_text(
    prompt: str,
    schema: type[T],
    model: str = "gemini-3-flash-preview",
    system_instruction: str = "",
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> T:
    """纯文本结构化分析，返回 Pydantic 对象。主模型失败后自动 fallback。"""
    models_to_try = [model] + _MODEL_FALLBACKS.get(model, [])
    last_error: Exception | None = None
    for m in models_to_try:
        try:
            return _try_model(
                model=m, contents=prompt, system_instruction=system_instruction,
                schema=schema, max_retries=max_retries, retry_delay=retry_delay,
                func_name="analyze_text",
            )
        except Exception as e:
            last_error = e
            if m != models_to_try[-1]:
                logger.warning(f"analyze_text [{m}] 全部重试失败，fallback 到下一个模型: {e}")

    raise RuntimeError(f"analyze_text 所有模型均失败: {last_error}") from last_error


def analyze_multimodal(
    contents: list,
    schema: type[T],
    model: str = "gemini-3-flash-preview",
    system_instruction: str = "",
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> T:
    """多模态结构化分析（文本+图片），返回 Pydantic 对象。主模型失败后自动 fallback。"""
    models_to_try = [model] + _MODEL_FALLBACKS.get(model, [])
    last_error: Exception | None = None
    for m in models_to_try:
        try:
            return _try_model(
                model=m, contents=contents, system_instruction=system_instruction,
                schema=schema, max_retries=max_retries, retry_delay=retry_delay,
                func_name="analyze_multimodal",
            )
        except Exception as e:
            last_error = e
            if m != models_to_try[-1]:
                logger.warning(f"analyze_multimodal [{m}] 全部重试失败，fallback 到下一个模型: {e}")

    raise RuntimeError(f"analyze_multimodal 所有模型均失败: {last_error}") from last_error
