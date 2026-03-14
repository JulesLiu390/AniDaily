"""Tool 4: 生成单格条漫。

输入角色图 + 场景图（可选）+ 分镜描述 → 生成一格条漫画面。
"""

import logging
from pathlib import Path

from google.genai.types import GenerateContentConfig, Part

from mcp.server.fastmcp import FastMCP

from src.tools.models.registry import get_genai_client

logger = logging.getLogger(__name__)

IMAGE_MODEL = "gemini-3.1-flash-image-preview"


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")


def register(mcp: FastMCP) -> None:
    """注册单格条漫生成工具。"""

    @mcp.tool()
    def generate_panel(
        character_paths: list[str],
        description: str,
        output_path: str,
        scene_path: str | None = None,
        dialogue: list[str] | None = None,
        camera: str = "medium shot",
    ) -> dict:
        """生成单格条漫画面。

        Args:
            character_paths: 角色图片路径列表（只画这些角色）。
            description: 画面描述，如 "远景，四人围坐餐桌"。
            output_path: 输出图片路径。
            scene_path: 场景/背景参考图路径（可选）。
            dialogue: 对话列表，如 ["A: 新年快乐！", "B: 干杯！"]。
            camera: 镜头类型，如 "wide shot", "close-up", "medium shot"。

        Returns:
            生成结果，包含输出路径。
        """
        contents: list = []

        # 添加角色图
        char_labels = []
        for i, cp in enumerate(character_paths):
            p = Path(cp)
            if not p.exists():
                return {"error": f"角色图不存在: {cp}"}
            contents.append(
                Part.from_bytes(data=p.read_bytes(), mime_type=_guess_mime(p))
            )
            label = chr(ord("A") + i)
            char_labels.append(f"Image {i + 1}: Character {label}")

        # 添加场景图
        scene_label = ""
        if scene_path:
            sp = Path(scene_path)
            if not sp.exists():
                return {"error": f"场景图不存在: {scene_path}"}
            contents.append(
                Part.from_bytes(data=sp.read_bytes(), mime_type=_guess_mime(sp))
            )
            scene_label = f"Image {len(character_paths) + 1}: Scene/Background\n"

        # 构建 prompt
        labels_text = "\n".join(char_labels)
        dialogue_text = ""
        if dialogue:
            dialogue_text = "\nDialogue bubbles:\n" + "\n".join(f"- {d}" for d in dialogue)

        prompt = (
            f"{labels_text}\n{scene_label}\n"
            f"Generate a single comic panel.\n\n"
            f"STRICT: ONLY draw the characters shown above. "
            f"Do NOT add any extra characters.\n"
            f"Each character's appearance must match their reference image exactly.\n\n"
            f"Panel description: {description}\n"
            f"Camera: {camera}\n"
            f"{dialogue_text}\n\n"
            f"- Manga/comic art style\n"
            f"- Dialogue bubbles in Chinese\n"
            f"Output a single comic panel image."
        )
        contents.append(prompt)

        try:
            client = get_genai_client()
            resp = client.models.generate_content(
                model=IMAGE_MODEL,
                contents=contents,
                config=GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )

            image_bytes = None
            text_response = None
            if resp.candidates:
                for part in resp.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                        image_bytes = part.inline_data.data
                    if part.text:
                        text_response = part.text

            if image_bytes is None:
                return {"error": "Gemini 未返回图片", "text": text_response}

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(image_bytes)

            result = {
                "output_path": str(out),
                "characters": character_paths,
                "description": description,
            }
            if text_response:
                result["text"] = text_response
            return result

        except Exception as e:
            return {"error": str(e)}
