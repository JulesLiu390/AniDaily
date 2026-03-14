"""Gemini 图片生成工具。

用法：
    from src.tools.gemini_image import generate_image

    path = generate_image(prompt="一只可爱的猫咪", output_path="output/cat.jpg")
"""

import logging
import time
from pathlib import Path

from google.genai.types import GenerateContentConfig, Part

from src.tools.models.registry import get_genai_client

logger = logging.getLogger(__name__)

IMAGE_MODEL = "gemini-3.1-flash-image-preview"


def _extract_image_from_response(resp) -> bytes | None:
    """从 Gemini 响应中提取第一张图片的 bytes。"""
    if not resp.candidates:
        return None
    for part in resp.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            return part.inline_data.data
    return None


def generate_image(
    prompt: str,
    output_path: str | Path,
    reference_images: list[str | Path] | None = None,
    model: str = IMAGE_MODEL,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Path:
    """使用 Gemini 生成图片。

    Args:
        prompt: 图片描述提示词。
        output_path: 输出图片路径。
        reference_images: 参考图片路径列表（可选），会和 prompt 一起发送。
        model: 图像生成模型 ID。
        max_retries: 最大重试次数。
        retry_delay: 重试间隔（秒），每次翻倍。

    Returns:
        生成的图片路径。
    """
    output_path = Path(output_path)

    # 构建 contents
    contents: list = []
    if reference_images:
        for img in reference_images:
            p = Path(img)
            if not p.exists():
                raise FileNotFoundError(f"参考图片不存在: {p}")
            mime = _guess_mime(p)
            contents.append(Part.from_bytes(data=p.read_bytes(), mime_type=mime))
    contents.append(prompt)

    last_error: Exception | None = None
    delay = retry_delay
    for attempt in range(1, max_retries + 1):
        client = get_genai_client()
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            image_bytes = _extract_image_from_response(resp)
            if image_bytes is None:
                raise RuntimeError("Gemini 未返回图片")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_bytes)
            logger.info(f"图片已保存: {output_path}")
            return output_path
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(f"图片生成重试 {attempt}/{max_retries}: {e}")
                time.sleep(delay)
                delay = min(delay * 2, 30)

    raise RuntimeError(f"图片生成失败，已重试 {max_retries} 次: {last_error}") from last_error


def edit_image(
    image_path: str | Path,
    prompt: str,
    output_path: str | Path | None = None,
    reference_images: list[str | Path] | None = None,
    model: str = IMAGE_MODEL,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Path:
    """使用 Gemini 编辑已有图片。

    Args:
        image_path: 待编辑的图片路径。
        prompt: 编辑指令。
        output_path: 输出路径，默认在原图旁生成 _edited 后缀文件。
        reference_images: 参考图片路径列表（可选）。
        model: 图像编辑模型 ID。
        max_retries: 最大重试次数。
        retry_delay: 重试间隔（秒），每次翻倍。

    Returns:
        编辑后的图片路径。
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_edited{image_path.suffix}"
    output_path = Path(output_path)

    # 构建 contents: 原图 + 参考图 + prompt
    contents: list = [
        Part.from_bytes(data=image_path.read_bytes(), mime_type=_guess_mime(image_path))
    ]
    if reference_images:
        for img in reference_images:
            p = Path(img)
            if not p.exists():
                raise FileNotFoundError(f"参考图片不存在: {p}")
            contents.append(Part.from_bytes(data=p.read_bytes(), mime_type=_guess_mime(p)))
    contents.append(prompt)

    last_error: Exception | None = None
    delay = retry_delay
    for attempt in range(1, max_retries + 1):
        client = get_genai_client()
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            image_bytes = _extract_image_from_response(resp)
            if image_bytes is None:
                raise RuntimeError("Gemini 未返回图片")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_bytes)
            logger.info(f"编辑后图片已保存: {output_path}")
            return output_path
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(f"图片编辑重试 {attempt}/{max_retries}: {e}")
                time.sleep(delay)
                delay = min(delay * 2, 30)

    raise RuntimeError(f"图片编辑失败，已重试 {max_retries} 次: {last_error}") from last_error


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
