"""测试条漫生成：4个角色 + 1个场景 → 一页条漫。

流程：
    1. Gemini Flash 分析每张角色图 → 精确外貌描述
    2. 角色图 + 场景图 + 精确描述 → Gemini Image 生成条漫
"""

import logging
from pathlib import Path

from google.genai.types import GenerateContentConfig, Part
from pydantic import BaseModel

from src.tools.gemini_text import analyze_multimodal
from src.tools.models.registry import get_genai_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("tests/test_comic_strip/output")
IMAGE_MODEL = "gemini-3.1-flash-image-preview"

# 素材路径
CHARACTER_PATHS = [
    Path("output/stylized/person_0.png"),
    Path("output/stylized/person_1.png"),
    Path("output/stylized/person_2.png"),
    Path("output/stylized/person_3.png"),
]
SCENE_PATH = Path("output/scenes/stylized/scene_1.png")


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")


def generate_comic():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("生成条漫...")
    contents: list = []

    # 角色图 + 场景图
    for char_path in CHARACTER_PATHS:
        contents.append(
            Part.from_bytes(data=char_path.read_bytes(), mime_type=_guess_mime(char_path))
        )
    contents.append(
        Part.from_bytes(data=SCENE_PATH.read_bytes(), mime_type=_guess_mime(SCENE_PATH))
    )

    contents.append(
        "Image 1: Character A\n"
        "Image 2: Character B\n"
        "Image 3: Character C\n"
        "Image 4: Character D\n"
        "Image 5: Scene\n\n"
        "Generate a vertical comic strip (条漫) with 4-6 panels.\n\n"
        "STRICT: The comic contains ONLY these 4 characters (A, B, C, D) shown above. "
        "Do NOT invent or add ANY other characters. "
        "Each character's appearance must match their reference image exactly.\n\n"
        "Scene: Chinese New Year family dinner at this home.\n"
        "- Manga/comic layout, black panel borders, 9:16 vertical\n"
        "- Dialogue bubbles in Chinese\n"
        "- Vary camera angles\n"
        "- Every character appears in at least 2 panels\n"
        "Output a single vertical comic strip image."
    )

    client = get_genai_client()
    resp = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=contents,
        config=GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    # 提取图片
    image_bytes = None
    if resp.candidates:
        for part in resp.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                image_bytes = part.inline_data.data
                break
            if part.text:
                logger.info(f"模型文字回复: {part.text}")

    if image_bytes is None:
        logger.error("Gemini 未返回图片")
        return

    output_path = OUTPUT_DIR / "comic_strip.png"
    output_path.write_bytes(image_bytes)
    logger.info(f"条漫已保存: {output_path}")


if __name__ == "__main__":
    generate_comic()
