"""Tool 1: 人脸检测 + 角色风格化。

detect - 输入图片，返回检测到的人脸列表（bbox、age、gender、裁剪图路径）
stylize - 输入人脸裁剪图 + 原图，返回风格化角色图路径
"""

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.tools.person_detector import detect_faces, crop_faces
from src.tools.face_stylizer import stylize_face

logger = logging.getLogger(__name__)


def register(mcp: FastMCP) -> None:
    """注册人脸检测和风格化工具。"""

    @mcp.tool()
    def detect_faces_in_image(
        image_path: str,
        confidence: float = 0.5,
        output_dir: str | None = None,
    ) -> dict:
        """检测图片中的所有人脸。

        Args:
            image_path: 图片路径。
            confidence: 检测置信度阈值（0-1）。
            output_dir: 裁剪输出目录，不指定则自动生成在图片同级目录。

        Returns:
            检测结果，包含人脸列表（bbox、age、gender、裁剪图路径）。
        """
        img_path = Path(image_path)
        if not img_path.exists():
            return {"error": f"图片不存在: {image_path}"}

        faces = detect_faces(img_path, confidence=confidence)
        if not faces:
            return {"faces": [], "count": 0, "message": "未检测到人脸"}

        # 裁剪人脸
        if output_dir:
            out_dir = Path(output_dir)
        else:
            out_dir = img_path.parent / f"{img_path.stem}_faces"

        crop_paths = crop_faces(img_path, output_dir=out_dir)

        face_list = []
        for i, face in enumerate(faces):
            face_info = {
                "index": i,
                "bbox": {
                    "x1": round(face.x1, 1),
                    "y1": round(face.y1, 1),
                    "x2": round(face.x2, 1),
                    "y2": round(face.y2, 1),
                },
                "width": round(face.width, 1),
                "height": round(face.height, 1),
                "confidence": round(face.confidence, 3),
                "age": face.age,
                "gender": face.gender,
            }
            if i < len(crop_paths):
                face_info["crop_path"] = str(crop_paths[i])
            face_list.append(face_info)

        return {
            "faces": face_list,
            "count": len(face_list),
            "source_image": str(img_path),
        }

    @mcp.tool()
    def stylize_character(
        face_path: str,
        output_path: str,
        original_image_path: str | None = None,
        prompt: str | None = None,
    ) -> dict:
        """将人脸照片风格化为动画角色形象（全身，9:16）。

        Args:
            face_path: 裁剪的人脸图片路径。
            output_path: 输出风格化图片路径。
            original_image_path: 原始完整图片路径（提供服装、体型参考）。
            prompt: 自定义风格化提示词，None 使用默认动画风格。

        Returns:
            风格化结果，包含输出路径。
        """
        try:
            result_path = stylize_face(
                face_path=face_path,
                output_path=output_path,
                original_image_path=original_image_path,
                prompt=prompt,
            )
            return {
                "output_path": str(result_path),
                "face_path": face_path,
                "original_image_path": original_image_path,
            }
        except Exception as e:
            return {"error": str(e)}
