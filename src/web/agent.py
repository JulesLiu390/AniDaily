"""对话 Agent - 基于 Gemini function calling 编排 AniDaily tools。"""

import json
import logging
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

from google.genai.types import (
    Content,
    FunctionCallingConfig,
    FunctionCallingConfigMode,
    FunctionDeclaration,
    GenerateContentConfig,
    Part,
    Schema,
    ToolConfig,
    Type,
)

from src.tools.face_stylizer import stylize_face
from src.tools.gemini_image import edit_image, generate_image
from src.tools.models.registry import get_genai_client
from src.tools.person_detector import crop_faces, detect_faces

logger = logging.getLogger(__name__)

MODEL = "gemini-3-flash-preview"
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ========== Tool 定义 ==========

TOOL_DECLARATIONS = [
    FunctionDeclaration(
        name="detect_faces_in_image",
        description="检测图片中的所有人脸，返回人脸列表（bbox、age、gender）并裁剪保存。",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "image_path": Schema(type=Type.STRING, description="图片路径"),
                "output_dir": Schema(type=Type.STRING, description="裁剪输出目录（可选）"),
            },
            required=["image_path"],
        ),
    ),
    FunctionDeclaration(
        name="stylize_character",
        description="将人脸照片风格化为全身动画角色形象（9:16）。输入人脸裁剪图和原图。",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "face_path": Schema(type=Type.STRING, description="裁剪的人脸图片路径"),
                "output_path": Schema(type=Type.STRING, description="输出路径"),
                "original_image_path": Schema(type=Type.STRING, description="原始完整图片路径（可选）"),
                "prompt": Schema(type=Type.STRING, description="自定义风格化提示词（可选）"),
            },
            required=["face_path", "output_path"],
        ),
    ),
    FunctionDeclaration(
        name="edit_asset",
        description="编辑已有图片素材（换服装、调整细节、去人等）。",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "image_path": Schema(type=Type.STRING, description="待编辑的图片路径"),
                "prompt": Schema(type=Type.STRING, description="编辑指令"),
                "output_path": Schema(type=Type.STRING, description="输出路径（可选）"),
            },
            required=["image_path", "prompt"],
        ),
    ),
    FunctionDeclaration(
        name="generate_asset",
        description="凭空生成新素材（新角色、新场景等）。",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "prompt": Schema(type=Type.STRING, description="生成描述"),
                "output_path": Schema(type=Type.STRING, description="输出路径"),
                "reference_images": Schema(
                    type=Type.ARRAY,
                    items=Schema(type=Type.STRING),
                    description="参考图片路径列表（可选）",
                ),
            },
            required=["prompt", "output_path"],
        ),
    ),
    FunctionDeclaration(
        name="generate_panel",
        description="生成单格条漫画面。输入角色图+场景图+描述。",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "character_paths": Schema(
                    type=Type.ARRAY,
                    items=Schema(type=Type.STRING),
                    description="角色图片路径列表",
                ),
                "description": Schema(type=Type.STRING, description="画面描述"),
                "output_path": Schema(type=Type.STRING, description="输出路径"),
                "scene_path": Schema(type=Type.STRING, description="场景图路径（可选）"),
                "dialogue": Schema(
                    type=Type.ARRAY,
                    items=Schema(type=Type.STRING),
                    description="对话列表，如 ['A: 你好', 'B: 嗨']",
                ),
                "camera": Schema(type=Type.STRING, description="镜头类型：wide/medium/close-up"),
            },
            required=["character_paths", "description", "output_path"],
        ),
    ),
    FunctionDeclaration(
        name="read_script",
        description="读取剧本 md 文件内容。",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "file_path": Schema(type=Type.STRING, description="md 文件路径"),
            },
            required=["file_path"],
        ),
    ),
    FunctionDeclaration(
        name="write_script",
        description="写入/覆盖剧本 md 文件。",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "file_path": Schema(type=Type.STRING, description="md 文件路径"),
                "content": Schema(type=Type.STRING, description="完整文件内容"),
            },
            required=["file_path", "content"],
        ),
    ),
    FunctionDeclaration(
        name="update_script",
        description="替换剧本文件中的指定文本。",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "file_path": Schema(type=Type.STRING, description="md 文件路径"),
                "old_text": Schema(type=Type.STRING, description="要替换的原文本"),
                "new_text": Schema(type=Type.STRING, description="替换后的新文本"),
            },
            required=["file_path", "old_text", "new_text"],
        ),
    ),
    FunctionDeclaration(
        name="list_files",
        description="列出指定目录下的文件和子目录（类似 tree 命令）。用于查看项目中已有的素材文件。",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "directory": Schema(type=Type.STRING, description="要列出的目录路径"),
                "max_depth": Schema(type=Type.INTEGER, description="最大递归深度（默认2）"),
            },
            required=["directory"],
        ),
    ),
]


# ========== Tool 执行 ==========

def _execute_tool(name: str, args: dict) -> dict:
    """执行指定 tool 并返回结果。"""
    logger.info(f"执行 tool: {name}({json.dumps(args, ensure_ascii=False)[:200]})")

    if name == "detect_faces_in_image":
        img_path = Path(args["image_path"])
        if not img_path.exists():
            return {"error": f"图片不存在: {args['image_path']}"}
        faces = detect_faces(img_path, confidence=0.5)
        out_dir = Path(args.get("output_dir") or str(img_path.parent / f"{img_path.stem}_faces"))
        crop_paths = crop_faces(img_path, output_dir=out_dir)
        face_list = []
        for i, face in enumerate(faces):
            info = {
                "index": i,
                "width": round(face.width, 1),
                "height": round(face.height, 1),
                "confidence": round(face.confidence, 3),
                "age": face.age,
                "gender": face.gender,
            }
            if i < len(crop_paths):
                info["crop_path"] = str(crop_paths[i])
            face_list.append(info)
        return {
            "faces": face_list,
            "count": len(face_list),
            "original_image": str(img_path),
            "crop_directory": str(out_dir),
        }

    elif name == "stylize_character":
        try:
            result_path = stylize_face(
                face_path=args["face_path"],
                output_path=args["output_path"],
                original_image_path=args.get("original_image_path"),
                prompt=args.get("prompt"),
            )
            return {"output_path": str(result_path)}
        except Exception as e:
            return {"error": str(e)}

    elif name == "edit_asset":
        try:
            result_path = edit_image(
                image_path=args["image_path"],
                prompt=args["prompt"],
                output_path=args.get("output_path"),
            )
            return {"output_path": str(result_path)}
        except Exception as e:
            return {"error": str(e)}

    elif name == "generate_asset":
        try:
            result_path = generate_image(
                prompt=args["prompt"],
                output_path=args["output_path"],
                reference_images=args.get("reference_images"),
            )
            return {"output_path": str(result_path)}
        except Exception as e:
            return {"error": str(e)}

    elif name == "generate_panel":
        from src.mcp_tools.generate_panel import _guess_mime
        from google.genai.types import GenerateContentConfig as GCC

        contents: list = []
        char_labels = []
        for i, cp in enumerate(args["character_paths"]):
            p = Path(cp)
            if not p.exists():
                return {"error": f"角色图不存在: {cp}"}
            contents.append(Part.from_bytes(data=p.read_bytes(), mime_type=_guess_mime(p)))
            char_labels.append(f"Image {i + 1}: Character {chr(ord('A') + i)}")

        scene_label = ""
        if args.get("scene_path"):
            sp = Path(args["scene_path"])
            if not sp.exists():
                return {"error": f"场景图不存在: {args['scene_path']}"}
            contents.append(Part.from_bytes(data=sp.read_bytes(), mime_type=_guess_mime(sp)))
            scene_label = f"Image {len(args['character_paths']) + 1}: Scene\n"

        dialogue_text = ""
        if args.get("dialogue"):
            dialogue_text = "\nDialogue:\n" + "\n".join(f"- {d}" for d in args["dialogue"])

        camera = args.get("camera", "medium shot")
        prompt = (
            f"{chr(10).join(char_labels)}\n{scene_label}\n"
            f"Generate a single comic panel.\n"
            f"STRICT: ONLY draw the characters shown above. Do NOT add extra characters.\n"
            f"Panel: {args['description']}\nCamera: {camera}\n{dialogue_text}\n"
            f"Manga style, dialogue bubbles in Chinese.\n"
            f"Output a single comic panel image."
        )
        contents.append(prompt)

        try:
            client = get_genai_client()
            resp = client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=contents,
                config=GCC(response_modalities=["IMAGE", "TEXT"]),
            )
            image_bytes = None
            if resp.candidates:
                for part in resp.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                        image_bytes = part.inline_data.data
            if not image_bytes:
                return {"error": "未返回图片"}
            out = Path(args["output_path"])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(image_bytes)
            return {"output_path": str(out)}
        except Exception as e:
            return {"error": str(e)}

    elif name == "read_script":
        p = Path(args["file_path"])
        if not p.exists():
            return {"error": f"文件不存在: {args['file_path']}"}
        return {"content": p.read_text(encoding="utf-8")}

    elif name == "write_script":
        p = Path(args["file_path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"], encoding="utf-8")
        return {"message": "写入成功", "file_path": str(p)}

    elif name == "update_script":
        p = Path(args["file_path"])
        if not p.exists():
            return {"error": f"文件不存在: {args['file_path']}"}
        content = p.read_text(encoding="utf-8")
        if args["old_text"] not in content:
            return {"error": "未找到要替换的文本"}
        new_content = content.replace(args["old_text"], args["new_text"])
        p.write_text(new_content, encoding="utf-8")
        return {"message": "替换成功"}

    elif name == "list_files":
        directory = Path(args["directory"])
        if not directory.exists():
            return {"error": f"目录不存在: {args['directory']}"}
        if not directory.is_dir():
            return {"error": f"不是目录: {args['directory']}"}
        max_depth = int(args.get("max_depth", 2))
        files: list[str] = []

        def _walk(d: Path, depth: int, prefix: str = ""):
            if depth > max_depth:
                return
            try:
                entries = sorted(d.iterdir())
            except PermissionError:
                return
            for entry in entries:
                files.append(f"{prefix}{entry.name}{'/' if entry.is_dir() else ''}")
                if entry.is_dir():
                    _walk(entry, depth + 1, prefix + "  ")

        _walk(directory, 1)
        return {"directory": str(directory), "files": files, "count": len(files)}

    else:
        return {"error": f"未知 tool: {name}"}


# ========== Agent ==========

SYSTEM_INSTRUCTION_TEMPLATE = (
    "你是 AniDaily 动画条漫生成助手。你可以：\n"
    "1. 检测图片中的人脸并风格化为动画角色\n"
    "2. 编辑已有素材或凭空生成新素材\n"
    "3. 分析场景\n"
    "4. 生成单格条漫\n"
    "5. 读写剧本 md 文件\n\n"
    "交互规则：\n"
    "- 当用户发送图片时，不要自动执行任何工具。先确认用户的意图（例如：检测人脸？风格化角色？编辑素材？），"
    "然后再执行对应操作。\n"
    "- 只有当用户明确要求执行某个操作时，才调用对应工具。\n"
    "- 工具返回的文件路径必须在后续操作中直接使用，不要猜测路径。\n"
    "- 如果不确定文件位置，先使用 list_files 工具查看。\n\n"
    "{project_context}"
    "回复使用中文。"
)


class Agent:
    """对话 Agent，维护多轮对话历史，通过 Gemini function calling 调用 tools。"""

    def __init__(self, project_dir: Path | None = None):
        self.history: list[Content] = []
        self.project_dir = project_dir

    def _build_system_instruction(self) -> str:
        if self.project_dir:
            out = self.project_dir / "output"
            project_context = (
                f"当前项目目录: {self.project_dir}\n"
                f"用户上传的临时文件在 {self.project_dir / 'tmp'} 下。\n"
                f"输出文件必须放在对应子目录下：\n"
                f"  - 风格化角色: {out / 'stylized'}/  (例: {out / 'stylized' / 'character_1.png'})\n"
                f"  - 人脸裁剪: {out / 'faces'}/\n"
                f"  - 场景: {out / 'scenes' / 'stylized'}/\n"
                f"  - 去人场景: {out / 'scenes' / 'no_people'}/\n"
                f"  - 条漫: {out / 'panels'}/\n"
                f"  - 剧本: {out / 'scripts'}/\n"
            )
        else:
            project_context = "所有文件路径基于项目根目录。输出文件默认放在 output/ 下。\n"
        return SYSTEM_INSTRUCTION_TEMPLATE.format(project_context=project_context)

    def _build_config(self) -> GenerateContentConfig:
        return GenerateContentConfig(
            system_instruction=self._build_system_instruction(),
            tools=[{"function_declarations": TOOL_DECLARATIONS}],
            tool_config=ToolConfig(
                function_calling_config=FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.AUTO,
                ),
            ),
        )

    def chat_stream(
        self, message: str, image_paths: list[str] | None = None
    ) -> Generator[dict, None, None]:
        """处理一轮对话，流式返回事件。

        事件类型:
        - text_delta: {"event": "text_delta", "delta": str}
        - tool_start:  {"event": "tool_start", "tool": str, "args": dict, "index": int}
        - tool_end:    {"event": "tool_end", "tool": str, "result": dict, "duration_ms": int, "index": int, "images": list|None}
        - done:        {"event": "done"}
        """
        # 构建用户消息
        user_parts: list = []
        if image_paths:
            for img_path in image_paths:
                p = Path(img_path)
                if p.exists():
                    suffix = p.suffix.lower()
                    mime = {
                        ".png": "image/png", ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg", ".webp": "image/webp",
                    }.get(suffix, "application/octet-stream")
                    user_parts.append(Part.from_text(text=f"[图片路径: {p}]"))
                    user_parts.append(Part.from_bytes(data=p.read_bytes(), mime_type=mime))
        user_parts.append(Part.from_text(text=message))
        self.history.append(Content(role="user", parts=user_parts))

        config = self._build_config()
        tool_index = 0
        max_rounds = 10

        for _ in range(max_rounds):
            client = get_genai_client(timeout=180_000)

            # 流式调用 Gemini
            accumulated_text = ""
            accumulated_parts: list[Part] = []
            function_call_parts: list = []

            stream = client.models.generate_content_stream(
                model=MODEL,
                contents=self.history,
                config=config,
            )

            for chunk in stream:
                if not chunk.candidates:
                    continue
                for part in chunk.candidates[0].content.parts:
                    if part.function_call is not None:
                        function_call_parts.append(part)
                    elif part.text:
                        accumulated_text += part.text
                        yield {"event": "text_delta", "delta": part.text}

            # 构建完整的 content 并加入历史
            all_parts = []
            if accumulated_text:
                all_parts.append(Part.from_text(text=accumulated_text))
            all_parts.extend(function_call_parts)

            if all_parts:
                self.history.append(Content(role="model", parts=all_parts))

            if not function_call_parts:
                # 没有 function call，结束
                break

            # 执行 function calls
            function_response_parts = []
            for fc_part in function_call_parts:
                fc = fc_part.function_call
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                yield {"event": "tool_start", "tool": tool_name, "args": tool_args, "index": tool_index}

                t0 = time.time()
                result = _execute_tool(tool_name, tool_args)
                duration_ms = round((time.time() - t0) * 1000)

                # 收集生成的图片
                images = None
                if "output_path" in result:
                    out_path = Path(result["output_path"])
                    if out_path.exists():
                        rel = out_path.relative_to(PROJECT_ROOT)
                        images = [{"path": str(out_path), "url": f"/files/{rel}", "tool": tool_name}]

                yield {
                    "event": "tool_end",
                    "tool": tool_name,
                    "result": result,
                    "duration_ms": duration_ms,
                    "index": tool_index,
                    "images": images,
                }

                tool_index += 1

                function_response_parts.append(
                    Part.from_function_response(name=tool_name, response=result)
                )

            self.history.append(Content(role="user", parts=function_response_parts))

        yield {"event": "done"}
