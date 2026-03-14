"""Tool 2: 编辑已有素材。

对已生成的角色图/场景图进行二次编辑（换服装、调整细节等）。
"""

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.tools.gemini_image import edit_image, generate_image

logger = logging.getLogger(__name__)


def register(mcp: FastMCP) -> None:
    """注册素材编辑工具。"""

    @mcp.tool()
    def edit_asset(
        image_path: str,
        prompt: str,
        output_path: str | None = None,
    ) -> dict:
        """编辑已有图片素材（角色图、场景图等）。

        Args:
            image_path: 待编辑的图片路径。
            prompt: 编辑指令，如 "换成红色外套"、"去掉背景中的人"。
            output_path: 输出路径，不指定则在原图旁生成 _edited 文件。

        Returns:
            编辑结果，包含输出路径。
        """
        try:
            result_path = edit_image(
                image_path=image_path,
                prompt=prompt,
                output_path=output_path,
            )
            return {
                "output_path": str(result_path),
                "source_image": image_path,
                "edit_prompt": prompt,
            }
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def generate_asset(
        prompt: str,
        output_path: str,
        reference_images: list[str] | None = None,
    ) -> dict:
        """凭空生成新素材（新角色、新场景等）。

        Args:
            prompt: 生成描述，如 "穿白大褂的医生，动画风格，白色背景，全身"。
            output_path: 输出图片路径。
            reference_images: 参考图片路径列表（可选，用于风格参考）。

        Returns:
            生成结果，包含输出路径。
        """
        try:
            result_path = generate_image(
                prompt=prompt,
                output_path=output_path,
                reference_images=reference_images,
            )
            return {
                "output_path": str(result_path),
                "prompt": prompt,
            }
        except Exception as e:
            return {"error": str(e)}
