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
from src.tools.models.registry import get_genai_client, get_key_count
from src.tools.person_detector import crop_faces

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
                "character_name": Schema(type=Type.STRING, description="角色特征简短描述，用于文件命名，如 yellow_jacket_boy、pink_dress_girl"),
                "original_image_path": Schema(type=Type.STRING, description="原始完整图片路径（可选）"),
                "prompt": Schema(type=Type.STRING, description="自定义风格化提示词（可选）"),
            },
            required=["face_path", "character_name"],
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
        description="凭空生成新素材（新角色、新场景等）。生成角色时请输出到 stylized 目录，会自动应用角色设计约束（9:16全身白底彩色动画风）。",
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
        name="generate_comic_strip",
        description=(
            "生成一条竖向条漫（4-6格）。每次调用生成一条，可多次调用生成多条。"
            "必须传入角色素材图片路径（stylized目录下）和本条的剧本内容。"
            "输出一张 9:16 竖向条漫图片，包含分格、对话气泡和黑色分格边框。"
        ),
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "character_paths": Schema(
                    type=Type.ARRAY,
                    items=Schema(type=Type.STRING),
                    description="角色图片路径列表（必须是 stylized 目录下的角色图）",
                ),
                "character_names": Schema(
                    type=Type.ARRAY,
                    items=Schema(type=Type.STRING),
                    description="角色名称列表，与 character_paths 一一对应，如 ['Jules', 'Peize']",
                ),
                "script": Schema(type=Type.STRING, description="本条条漫的剧本（4-6格的分格描述、对话、镜头）"),
                "strip_index": Schema(type=Type.INTEGER, description="条漫编号，从1开始"),
                "output_path": Schema(type=Type.STRING, description="输出路径"),
                "scene_path": Schema(type=Type.STRING, description="场景/背景参考图路径（可选）"),
            },
            required=["character_paths", "character_names", "script", "strip_index", "output_path"],
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
        name="describe_image",
        description="用 VLM 分析图片，返回素材命名（snake_case英文）和中文描述。",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "image_path": Schema(type=Type.STRING, description="图片路径"),
            },
            required=["image_path"],
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
    FunctionDeclaration(
        name="select_characters",
        description=(
            "让用户从已有素材中选择角色。展示所有已风格化角色和人脸素材，"
            "自动预选最匹配用户描述的角色。用户可以自由调整选择后确认。"
            "在生成条漫前必须调用此工具让用户确认角色。"
        ),
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "preselected": Schema(
                    type=Type.ARRAY,
                    items=Schema(
                        type=Type.OBJECT,
                        properties={
                            "path": Schema(type=Type.STRING, description="预选角色的文件路径"),
                            "label": Schema(type=Type.STRING, description="角色在剧本中的名字，如 Jules、Peize"),
                        },
                    ),
                    description="根据用户描述预选的角色列表（path + label）",
                ),
            },
            required=["preselected"],
        ),
    ),
    FunctionDeclaration(
        name="propose_plan",
        description=(
            "向用户展示任务计划。当用户的请求涉及 2 个以上步骤时，"
            "先调用此工具让用户确认计划，用户可以跳过某些步骤或修改计划。"
            "用户确认后再按计划逐步执行。简单请求（单步操作）不需要调用。"
        ),
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "steps": Schema(
                    type=Type.ARRAY,
                    items=Schema(
                        type=Type.OBJECT,
                        properties={
                            "id": Schema(type=Type.INTEGER, description="步骤序号，从1开始"),
                            "label": Schema(type=Type.STRING, description="步骤简短描述"),
                            "tool": Schema(type=Type.STRING, description="对应的工具名（可选，用于前端进度追踪）"),
                            "needs_confirm": Schema(type=Type.BOOLEAN, description="是否需要用户交互确认（如选择人脸、选择角色）"),
                            "depends_on": Schema(
                                type=Type.ARRAY,
                                items=Schema(type=Type.INTEGER),
                                description="依赖的步骤 id 列表。该步骤会等所有依赖完成后才执行。无依赖的步骤可并发执行。",
                            ),
                        },
                        required=["id", "label"],
                    ),
                    description="任务步骤列表",
                ),
            },
            required=["steps"],
        ),
    ),
    FunctionDeclaration(
        name="select_faces",
        description=(
            "让用户从已检测的人脸中选择要风格化的人脸。"
            "当用户要求风格化角色时，必须先调用此工具让用户勾选要风格化哪些人脸。"
            "不要在人脸检测后自动调用，只在用户明确要求风格化时调用。"
        ),
        parameters=Schema(
            type=Type.OBJECT,
            properties={},
        ),
    ),
    FunctionDeclaration(
        name="rename_asset",
        description=(
            "重命名素材：同时更新显示名称和文件名（保留扩展名）。"
            "当用户给角色或素材起名、备注、改名时调用此工具。"
            "文件名默认与 name 相同，也可通过 filename 参数单独指定。"
        ),
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "file_path": Schema(type=Type.STRING, description="素材文件路径"),
                "name": Schema(type=Type.STRING, description="新的显示名称"),
                "filename": Schema(type=Type.STRING, description="新的文件名（不含扩展名，可选，默认用 name）"),
                "description": Schema(type=Type.STRING, description="新的描述（可选，不传则保留原描述）"),
            },
            required=["file_path", "name"],
        ),
    ),
]


# ========== Tool 执行 ==========

def _load_assets_json(directory: Path) -> dict:
    """读取目录下的 assets.json。"""
    p = directory / "assets.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_asset_meta(file_path: Path, name: str, description: str, source_face: str | None = None) -> None:
    """将素材的 name/description 写入所在目录的 assets.json。"""
    directory = file_path.parent
    data = _load_assets_json(directory)
    entry: dict[str, str] = {"name": name, "description": description}
    if source_face:
        entry["source_face"] = source_face
    data[file_path.name] = entry
    (directory / "assets.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


DESCRIBE_MODEL = "gemini-2.5-flash-lite"
DESCRIBE_PROMPT = (
    'Analyze this image and respond with a JSON object containing exactly two fields:\n'
    '1. "name": a short snake_case name for this asset (max 4 words), suitable as a filename. '
    'Focus on the most distinctive visual features: clothing color/type, hair, accessories, scene type.\n'
    '   Examples: yellow_jacket_boy, pink_dress_girl, sunset_beach_scene, dark_alley\n'
    '2. "description": a concise one-sentence description of what is in the image (in Chinese).\n\n'
    'Output ONLY valid JSON, no markdown, no extra text.\n'
    'Example: {"name": "yellow_jacket_boy", "description": "穿黄色夹克的年轻男性，短发，双手交叉"}'
)


def _describe_image(image_path: Path) -> dict:
    """用 VLM 生成图片的名称和描述。

    Returns:
        {"name": "snake_case_name", "description": "中文描述"}
    """
    try:
        from google.genai.types import GenerateContentConfig, Part

        client = get_genai_client(timeout=30_000)
        suffix = image_path.suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(suffix, "image/jpeg")
        img_part = Part.from_bytes(data=image_path.read_bytes(), mime_type=mime)
        resp = client.models.generate_content(
            model=DESCRIBE_MODEL,
            contents=[img_part, DESCRIBE_PROMPT],
            config=GenerateContentConfig(temperature=0.0),
        )
        text = resp.text.strip()
        # 去掉可能的 markdown 代码块
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        name = data.get("name", "asset")
        name = name.lower().replace(" ", "_").replace("-", "_")
        name = "".join(c for c in name if c.isalnum() or c == "_").strip("_")
        return {
            "name": name[:40] if name else "asset",
            "description": data.get("description", ""),
        }
    except Exception as e:
        logger.warning(f"VLM 描述失败: {e}")
    return {"name": "asset", "description": ""}


def _find_stylized_dir(face_path: Path) -> Path:
    """从人脸路径推断项目的 stylized 输出目录。"""
    # 人脸通常在 projects/{name}/output/faces/ 下
    # 往上找到 output 目录，然后定位 stylized
    current = face_path.parent
    for _ in range(5):
        if current.name == "output":
            d = current / "stylized"
            d.mkdir(parents=True, exist_ok=True)
            return d
        if current.name == "faces" and current.parent.name == "output":
            d = current.parent / "stylized"
            d.mkdir(parents=True, exist_ok=True)
            return d
        current = current.parent
    # fallback: 同目录
    return face_path.parent


def _resolve_path(p: str, project_dir: Path | None) -> str:
    """将相对路径解析为绝对路径（基于项目目录）。"""
    path = Path(p)
    if not path.is_absolute() and project_dir:
        path = project_dir / p
    return str(path)


def _execute_tool(name: str, args: dict, project_dir: Path | None = None, lang: str = "zh") -> dict:
    """执行指定 tool 并返回结果。"""
    # 自动将常见路径参数解析为绝对路径
    PATH_KEYS = [
        "image_path", "face_path", "output_path", "output_dir", "original_image_path",
        "file_path", "directory", "scene_path",
    ]
    for key in PATH_KEYS:
        if key in args and args[key] and not Path(args[key]).is_absolute():
            args[key] = _resolve_path(args[key], project_dir)
    # 列表类型路径参数
    for key in ["character_paths", "reference_images"]:
        if key in args and isinstance(args[key], list):
            args[key] = [_resolve_path(p, project_dir) if not Path(p).is_absolute() else p for p in args[key]]
    # 对象数组中的 path 字段（如 preselected: [{path, label}]）
    if "preselected" in args and isinstance(args["preselected"], list):
        for item in args["preselected"]:
            if isinstance(item, dict) and "path" in item and item["path"] and not Path(item["path"]).is_absolute():
                item["path"] = _resolve_path(item["path"], project_dir)

    logger.info(f"执行 tool: {name}({json.dumps(args, ensure_ascii=False)[:200]})")

    if name == "detect_faces_in_image":
        from concurrent.futures import ThreadPoolExecutor, as_completed

        img_path = Path(args["image_path"])
        if not img_path.exists():
            return {"error": f"图片不存在: {args['image_path']}"}
        out_dir = Path(args.get("output_dir") or str(img_path.parent / f"{img_path.stem}_faces"))
        crop_result = crop_faces(img_path, output_dir=out_dir)

        # 并发调用 VLM 描述每个人脸
        descriptions: dict[int, dict] = {}
        if crop_result.cropped_paths:
            with ThreadPoolExecutor(max_workers=len(crop_result.cropped_paths)) as pool:
                futures = {
                    pool.submit(_describe_image, cp): i
                    for i, cp in enumerate(crop_result.cropped_paths)
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    descriptions[idx] = fut.result()

        face_list = []
        for i, face in enumerate(crop_result.faces_kept):
            crop_path = crop_result.cropped_paths[i]
            desc = descriptions.get(i, {"name": "person", "description": ""})
            _save_asset_meta(crop_path, desc["name"], desc["description"])
            info = {
                "index": i,
                "name": desc["name"],
                "description": desc["description"],
                "width": round(face.width, 1),
                "height": round(face.height, 1),
                "confidence": round(face.confidence, 3),
                "age": face.age,
                "gender": face.gender,
                "crop_path": str(crop_path),
            }
            face_list.append(info)
        return {
            "faces": face_list,
            "count": len(face_list),
            "skipped_small": crop_result.skipped_small,
            "skipped_blurry": crop_result.skipped_blurry,
            "original_image": str(img_path),
            "crop_directory": str(out_dir),
        }

    elif name == "stylize_character":
        try:
            import uuid as _uuid
            char_name = args.get("character_name", "character").replace(" ", "_")
            uid = _uuid.uuid4().hex[:8]
            # 输出到项目的 stylized 目录
            face_p = Path(args["face_path"])
            # 向上找 output/stylized 目录
            stylized_dir = _find_stylized_dir(face_p)
            out_path = stylized_dir / f"{char_name}_{uid}.png"
            result_path = stylize_face(
                face_path=args["face_path"],
                output_path=out_path,
                original_image_path=args.get("original_image_path"),
                prompt=args.get("prompt"),
            )
            # VLM 描述风格化后的角色
            desc = _describe_image(result_path)
            face_filename = Path(args["face_path"]).name
            _save_asset_meta(result_path, desc["name"], desc["description"], source_face=face_filename)
            return {"output_path": str(result_path), "character_name": desc["name"], "description": desc["description"], "source_face": face_filename}
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
            from src.tools.face_stylizer import DEFAULT_PROMPT as CHARACTER_DESIGN_PROMPT

            out_path = Path(args["output_path"])
            prompt = args["prompt"]

            # 如果输出到 stylized 目录，自动注入角色设计约束（9:16、全身、白色背景、彩色）
            is_character = "stylized" in out_path.parts
            if is_character:
                prompt = (
                    CHARACTER_DESIGN_PROMPT.replace(
                        "Transform this person's photo into an anime/illustration style FULL-BODY character design.",
                        "Generate an anime/illustration style FULL-BODY character design from the description below.",
                    )
                    + f"\n\nCharacter description: {prompt}"
                )

            result_path = generate_image(
                prompt=prompt,
                output_path=args["output_path"],
                reference_images=args.get("reference_images"),
            )
            # VLM 描述生成的素材
            desc = _describe_image(result_path)
            # 如果参考图来自 faces 目录，记录 source_face
            source_face = None
            ref_images = args.get("reference_images") or []
            for ref in ref_images:
                ref_p = Path(ref)
                if "faces" in ref_p.parts:
                    source_face = ref_p.name
                    break
            _save_asset_meta(result_path, desc["name"], desc["description"], source_face=source_face)
            result_dict = {
                "output_path": str(result_path),
                "name": desc["name"],
                "description": desc["description"],
            }
            if source_face:
                result_dict["source_face"] = source_face
            return result_dict
        except Exception as e:
            return {"error": str(e)}

    elif name == "generate_comic_strip":
        from src.mcp_tools.generate_panel import _guess_mime
        from google.genai.types import GenerateContentConfig as GCC

        char_paths = args["character_paths"]
        char_names = args.get("character_names", [])
        script = args["script"]
        strip_index = int(args.get("strip_index", 1))

        contents: list = []
        char_labels = []
        for i, cp in enumerate(char_paths):
            p = Path(cp)
            if not p.exists():
                return {"error": f"角色图不存在: {cp}"}
            contents.append(Part.from_bytes(data=p.read_bytes(), mime_type=_guess_mime(p)))
            name_label = char_names[i] if i < len(char_names) else chr(ord("A") + i)
            char_labels.append(f"Image {i + 1}: Character {name_label}")

        scene_label = ""
        if args.get("scene_path"):
            sp = Path(args["scene_path"])
            if not sp.exists():
                return {"error": f"场景图不存在: {args['scene_path']}"}
            contents.append(Part.from_bytes(data=sp.read_bytes(), mime_type=_guess_mime(sp)))
            scene_label = f"Image {len(char_paths) + 1}: Scene/Background\n"

        prompt = (
            f"{chr(10).join(char_labels)}\n{scene_label}\n"
            f"Generate a vertical comic strip (条漫) with 4-6 panels.\n"
            f"This is strip #{strip_index}. Script for this strip:\n\n"
            f"{script}\n\n"
            f"STRICT RULES:\n"
            f"1. ONLY draw the characters shown above. Do NOT invent or add ANY other characters.\n"
            f"2. Each character's appearance MUST match their reference image exactly.\n"
            f"3. Every named character in the script should appear in at least 2 panels.\n\n"
            f"FORMAT:\n"
            f"- Vertical layout, 9:16 aspect ratio\n"
            f"- Black panel borders separating each panel\n"
            f"- Manga/comic art style\n"
            f"- Dialogue bubbles in {'English' if lang == 'en' else 'Chinese'}\n"
            f"- Vary camera angles across panels (wide, medium, close-up)\n\n"
            f"Output a single vertical comic strip image."
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
            text_response = None
            if resp.candidates:
                for part in resp.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                        image_bytes = part.inline_data.data
                    if part.text:
                        text_response = part.text
            if not image_bytes:
                return {"error": "未返回图片", "text": text_response}
            out = Path(args["output_path"])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(image_bytes)
            result_dict: dict = {
                "output_path": str(out),
                "strip_index": strip_index,
                "characters": char_paths,
            }
            if text_response:
                result_dict["text"] = text_response
            return result_dict
        except Exception as e:
            return {"error": str(e)}

    elif name == "describe_image":
        img_path = Path(args["image_path"])
        if not img_path.exists():
            return {"error": f"图片不存在: {args['image_path']}"}
        desc = _describe_image(img_path)
        _save_asset_meta(img_path, desc["name"], desc["description"])
        return {"name": desc["name"], "description": desc["description"], "image_path": str(img_path)}

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
        assets_meta: dict[str, dict] = {}

        def _walk(d: Path, depth: int, prefix: str = ""):
            if depth > max_depth:
                return
            # 读取该目录的 assets.json
            meta = _load_assets_json(d)
            if meta:
                rel = str(d.relative_to(directory)) if d != directory else "."
                assets_meta[rel] = meta
            try:
                entries = sorted(d.iterdir())
            except PermissionError:
                return
            for entry in entries:
                if entry.name == "assets.json":
                    continue  # 不列出 assets.json 本身
                files.append(f"{prefix}{entry.name}{'/' if entry.is_dir() else ''}")
                if entry.is_dir():
                    _walk(entry, depth + 1, prefix + "  ")

        _walk(directory, 1)
        result: dict = {"directory": str(directory), "files": files, "count": len(files)}
        if assets_meta:
            result["assets"] = assets_meta
        return result

    elif name == "select_characters":
        preselected = args.get("preselected", [])
        preselected_paths = {item["path"] for item in preselected}
        preselected_labels = {item["path"]: item.get("label", "") for item in preselected}

        all_options: list[dict] = []

        # 从 preselected 路径推断项目目录
        project_dir = None
        for item in preselected:
            p = Path(item["path"])
            current = p.parent
            for _ in range(5):
                if current.name == "output":
                    project_dir = current.parent
                    break
                current = current.parent
            if project_dir:
                break

        SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}
        if project_dir:
            for category, subdir in [
                ("characters", project_dir / "output" / "stylized"),
                ("faces", project_dir / "output" / "faces"),
            ]:
                if not subdir.exists():
                    continue
                meta = _load_assets_json(subdir)
                for f in sorted(subdir.iterdir()):
                    if not f.is_file() or f.suffix.lower() not in SUPPORTED:
                        continue
                    file_meta = meta.get(f.name, {})
                    rel = f.relative_to(PROJECT_ROOT)
                    _mtime = int(f.stat().st_mtime)
                    option: dict[str, Any] = {
                        "path": str(f),
                        "url": f"/files/{rel}?v={_mtime}",
                        "filename": f.name,
                        "category": category,
                        "name": file_meta.get("name", f.stem),
                        "description": file_meta.get("description", ""),
                        "selected": str(f) in preselected_paths,
                        "label": preselected_labels.get(str(f), ""),
                    }
                    if file_meta.get("source_face"):
                        option["source_face"] = file_meta["source_face"]
                    all_options.append(option)

        return {
            "type": "character_select",
            "options": all_options,
            "preselected_count": len(preselected),
        }

    elif name == "propose_plan":
        steps = args.get("steps", [])
        return {
            "type": "task_plan",
            "steps": steps,
        }

    elif name == "rename_asset":
        file_path = Path(args["file_path"])
        if not file_path.exists():
            return {"error": f"文件不存在: {args['file_path']}"}
        directory = file_path.parent
        old_filename = file_path.name
        data = _load_assets_json(directory)
        entry = data.pop(old_filename, {})
        entry["name"] = args["name"]
        if "description" in args and args["description"]:
            entry["description"] = args["description"]

        # Rename file on disk
        new_stem = args.get("filename") or args["name"]
        new_filename = new_stem + file_path.suffix
        new_path = directory / new_filename
        # Avoid conflict
        if new_path.exists() and new_path != file_path:
            import uuid as _uuid
            new_filename = f"{new_stem}_{_uuid.uuid4().hex[:6]}{file_path.suffix}"
            new_path = directory / new_filename
        file_path.rename(new_path)

        data[new_filename] = entry
        (directory / "assets.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Update source_face references in other directories
        if project_dir:
            for sub in (project_dir / "output").rglob("assets.json"):
                if sub.parent == directory:
                    continue
                try:
                    sub_data = json.loads(sub.read_text(encoding="utf-8"))
                    changed = False
                    for k, v in sub_data.items():
                        if isinstance(v, dict) and v.get("source_face") == old_filename:
                            v["source_face"] = new_filename
                            changed = True
                    if changed:
                        sub.write_text(json.dumps(sub_data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass

        return {"message": f"已重命名为「{args['name']}」", "file_path": str(new_path), "name": args["name"], "old_filename": old_filename, "new_filename": new_filename}

    elif name == "select_faces":
        # 返回项目中已有的人脸供用户选择风格化
        faces_dir = project_dir / "output" / "faces" if project_dir else None
        face_list: list[dict] = []
        if faces_dir and faces_dir.exists():
            meta = _load_assets_json(faces_dir)
            SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}
            for f in sorted(faces_dir.iterdir()):
                if not f.is_file() or f.suffix.lower() not in SUPPORTED:
                    continue
                file_meta = meta.get(f.name, {})
                rel = f.relative_to(PROJECT_ROOT)
                face_list.append({
                    "index": len(face_list),
                    "name": file_meta.get("name", f.stem),
                    "description": file_meta.get("description", ""),
                    "age": None,
                    "gender": None,
                    "crop_path": str(f),
                    "crop_url": f"/files/{rel}?v={int(f.stat().st_mtime)}",
                })
        return {
            "type": "face_select",
            "faces": face_list,
            "count": len(face_list),
        }

    else:
        return {"error": f"未知 tool: {name}"}


# ========== Agent ==========

SYSTEM_INSTRUCTION_TEMPLATE = (
    "你是 AniDaily 动画条漫生成助手。你可以：\n"
    "1. 检测图片中的人脸并风格化为动画角色\n"
    "2. 编辑已有素材或凭空生成新素材\n"
    "3. 分析场景\n"
    "4. 生成竖向条漫（4-6格）\n"
    "5. 读写剧本 md 文件\n\n"
    "交互规则：\n"
    "- **任务计划**：当用户的请求涉及 2 个以上步骤时（如「检测人脸并风格化」「生成条漫」等复合任务），"
    "**必须先调用 propose_plan** 列出步骤让用户确认。简单单步操作不需要计划。"
    "每个步骤写清 label，可选填 tool（对应工具名）、needs_confirm（需要用户交互确认后才执行的步骤）、"
    "depends_on（依赖的步骤 id 列表，无依赖的步骤会并发执行）。\n"
    "**depends_on 规则**：如果步骤 B 需要步骤 A 的输出结果（如文件路径），则 B 必须 depends_on: [A.id]。"
    "同类型的独立操作（如生成多个不同角色）互相不依赖，系统会自动并发执行。\n"
    "- **needs_confirm 标记规则**：以下类型的步骤**必须**设置 needs_confirm=true：\n"
    "  - 选择类：选择人脸、选择角色\n"
    "  - 生成类：风格化角色(stylize_character)、生成条漫(generate_comic_strip)、生成素材(generate_asset)、编辑素材(edit_asset)\n"
    "  - 只有读取类(read_script)、列出文件(list_files)、重命名(rename_asset)、写入(write_script/update_script)等轻量操作不需要标记\n"
    "  - 简单说：**凡是耗时较长或产出用户需要审阅的内容的步骤，都要标 needs_confirm=true**\n"
    "- **严禁用文字询问确认**：绝对不要在文字中写「确认后开始执行」「准备好了吗」「您看这样可以吗」"
    "「接下来继续吗」「是否继续」等确认语句。"
    "所有确认必须通过 propose_plan 工具实现，前端会展示可点击的按钮。\n"
    "- **分步执行**：用户确认计划后，系统会自动逐步发送步骤指令给你。"
    "当你收到「继续执行计划步骤 N: xxx」的指令时，执行对应的工具调用即可，每次只执行一个步骤。"
    "不要自行决定执行多个步骤，不要在文字中询问是否继续。\n"
    "- 当用户发送图片时，不要自动执行任何工具。先确认用户的意图（例如：检测人脸？风格化角色？编辑素材？），"
    "然后再执行对应操作。\n"
    "- 只有当用户明确要求执行某个操作时，才调用对应工具。\n"
    "- 当用户给角色或素材起名、备注、改名时（如「这个人叫xxx」「把他命名为xxx」），"
    "立即调用 rename_asset，它会同时重命名显示名称和文件名。"
    "如果能从上下文推断出要重命名哪个文件，直接执行，不要反复确认。\n"
    "- 人脸检测完成后，如果检测到人脸，**必须先向用户展示检测结果**（每个人脸的编号、年龄、性别等信息），"
    "然后询问用户是否要对这些人脸进行风格化，以及要风格化哪些人（例如「全部」或「只要第1和第3个」）。"
    "**绝对不要在检测后自动调用 stylize_character**，必须等用户确认。\n"
    "- 当用户要求风格化角色时，**必须先调用 select_faces** 让用户在前端交互式选择要风格化哪些人脸，"
    "用户确认后再对选中的人脸逐个调用 stylize_character。\n"
    "- 工具返回的文件路径必须在后续操作中直接使用，不要猜测路径。\n"
    "- 如果不确定文件位置，先使用 list_files 工具查看。\n"
    "- **生成条漫的工作流**：\n"
    "  1. 先用 list_files 查看 output/stylized/ 目录，了解已有角色。\n"
    "  2. 调用 select_characters，根据用户描述预选最匹配的角色（preselected 里填 path+label）。\n"
    "     用户会在前端交互式选择确认。\n"
    "  3. 用户确认角色后，写剧本（write_script），按条划分（每条4-6格）。\n"
    "  4. 把剧本展示给用户，询问：要生成几条条漫？默认1条。\n"
    "  5. 用户确认后，对每条调用一次 generate_comic_strip，传入用户确认的角色路径。\n"
    "  6. 绝对不要用 generate_asset 生成条漫，generate_asset 不支持角色参考图。\n"
    "- **风格设定**：项目根目录有 style.md，记录画风、语言、排版等偏好。"
    "每次生成角色、条漫或素材前，必须先用 read_script 读取 style.md 并遵守其中的设定。\n\n"
    "{project_context}"
    "{lang_instruction}"
)


class Agent:
    """对话 Agent，维护多轮对话历史，通过 Gemini function calling 调用 tools。"""

    def __init__(self, project_dir: Path | None = None, lang: str = "zh", history: list[Content] | None = None):
        self.history: list[Content] = history or []
        self.project_dir = project_dir
        self.lang = lang
        # Plan execution state
        self.active_plan: list[dict] | None = None
        self.plan_paused: bool = False
        self.plan_auto: bool = True

    # Tools that produce interactive UI cards — always pause even in auto mode
    INTERACTIVE_TOOLS = {"select_faces", "select_characters"}

    # ---- Plan management (DAG-based) ----

    def plan_confirm(self, steps: list[dict], auto_execute: bool = True) -> None:
        """User confirmed a plan. Store enabled steps and prepare for execution."""
        self.active_plan = [
            {**s, "status": "pending"} for s in steps if s.get("status") != "skipped"
        ]
        self.plan_paused = False
        self.plan_auto = auto_execute

    def plan_continue(self, prompt: str | None = None) -> None:
        """User clicked continue at a gate."""
        self.plan_paused = False

    def plan_cancel(self) -> None:
        """User cancelled the plan."""
        if self.active_plan:
            for s in self.active_plan:
                if s["status"] in ("pending", "active"):
                    s["status"] = "skipped"
        self.plan_paused = False

    def _plan_step_by_id(self, step_id: int) -> dict | None:
        if not self.active_plan:
            return None
        for s in self.active_plan:
            if s["id"] == step_id:
                return s
        return None

    def _plan_done_ids(self) -> set[int]:
        """IDs of all completed steps."""
        if not self.active_plan:
            return set()
        return {s["id"] for s in self.active_plan if s["status"] == "done"}

    def _plan_runnable(self) -> list[dict]:
        """Return all pending steps whose dependencies are satisfied."""
        if not self.active_plan:
            return []
        done = self._plan_done_ids()
        runnable = []
        for s in self.active_plan:
            if s["status"] != "pending":
                continue
            deps = s.get("depends_on") or []
            if all(d in done for d in deps):
                runnable.append(s)
        return runnable

    def _plan_has_pending(self) -> bool:
        return bool(self.active_plan and any(s["status"] in ("pending", "active") for s in self.active_plan))

    def _plan_should_gate(self, step: dict) -> bool:
        """Check if a step should gate (pause for user)."""
        if not step.get("needs_confirm"):
            return False
        if self.plan_auto:
            return step.get("tool") in self.INTERACTIVE_TOOLS
        return True

    def _build_system_instruction(self) -> str:
        if self.project_dir:
            out = self.project_dir / "output"
            project_context = (
                f"当前项目目录: {self.project_dir}\n"
                f"用户上传的原始图片在 {self.project_dir / 'input'}/ 下。\n"
                f"输出文件必须放在对应子目录下：\n"
                f"  - 风格化角色: {out / 'stylized'}/\n"
                f"  - 人脸裁剪: {out / 'faces'}/\n"
                f"  - 场景: {out / 'scenes' / 'stylized'}/\n"
                f"  - 去人场景: {out / 'scenes' / 'no_people'}/\n"
                f"  - 条漫: {out / 'panels'}/\n"
                f"  - 剧本: {out / 'scripts'}/\n\n"
            )
            # 自动注入风格设定
            style_path = self.project_dir / "style.md"
            if style_path.exists():
                style_content = style_path.read_text(encoding="utf-8").strip()
                project_context += f"漫画风格设定（用户可编辑，生成内容时必须遵守）：\n{style_content}\n\n"

            # 自动注入各分类已有素材
            asset_summary = self._build_asset_summary()
            if asset_summary:
                project_context += f"当前项目已有素材：\n{asset_summary}\n"
        else:
            project_context = "所有文件路径基于项目根目录。输出文件默认放在 output/ 下。\n"
        lang_instruction = "Reply in English." if self.lang == "en" else "回复使用中文。"
        return SYSTEM_INSTRUCTION_TEMPLATE.format(project_context=project_context, lang_instruction=lang_instruction)

    def _build_asset_summary(self) -> str:
        """读取项目各分类的 assets.json，生成素材摘要注入到 system instruction。"""
        if not self.project_dir:
            return ""
        out = self.project_dir / "output"
        categories = {
            "原始图片": self.project_dir / "input",
            "风格化角色": out / "stylized",
            "人脸": out / "faces",
            "场景": out / "scenes" / "stylized",
            "去人场景": out / "scenes" / "no_people",
            "条漫": out / "panels",
            "剧本": out / "scripts",
        }
        lines: list[str] = []
        for label, dir_path in categories.items():
            if not dir_path.exists():
                continue
            meta = _load_assets_json(dir_path)
            # 统计目录中的实际文件（排除 assets.json）
            files = [f for f in dir_path.iterdir() if f.is_file() and f.name != "assets.json"]
            if not files:
                continue
            if meta:
                items = []
                for f in files:
                    info = meta.get(f.name)
                    if info:
                        items.append(f"    - {f.name}: {info.get('name', '')} ({info.get('description', '')})")
                    else:
                        items.append(f"    - {f.name}")
                lines.append(f"  [{label}] ({len(files)} 个):")
                lines.extend(items)
            else:
                lines.append(f"  [{label}] ({len(files)} 个): {', '.join(f.name for f in files[:5])}")
                if len(files) > 5:
                    lines.append(f"    ... 等共 {len(files)} 个文件")
        return "\n".join(lines)

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

    # ---- Single AI round (shared by normal chat and plan steps) ----

    def _run_ai_round(
        self, message: str, config: GenerateContentConfig,
        tool_index_start: int, image_paths: list[str] | None = None,
    ) -> Generator[dict, None, int]:
        """Run one AI round: send message, stream response, execute tool calls.

        Yields SSE events. Returns the next tool_index after this round.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Build user message
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

        tool_index = tool_index_start
        max_rounds = 15

        for _ in range(max_rounds):
            client = get_genai_client(timeout=180_000)
            accumulated_text = ""
            function_call_parts: list = []

            stream = client.models.generate_content_stream(
                model=MODEL, contents=self.history, config=config,
            )
            for chunk in stream:
                if not chunk.candidates:
                    continue
                content = chunk.candidates[0].content
                if content is None or content.parts is None:
                    continue
                for part in content.parts:
                    if part.function_call is not None:
                        function_call_parts.append(part)
                    elif part.text:
                        accumulated_text += part.text
                        yield {"event": "text_delta", "delta": part.text}

            all_parts = []
            if accumulated_text:
                all_parts.append(Part.from_text(text=accumulated_text))
            all_parts.extend(function_call_parts)
            if all_parts:
                self.history.append(Content(role="model", parts=all_parts))

            if not function_call_parts:
                break

            # Execute function calls (concurrent if multiple)
            tasks: list[dict] = []
            for fc_part in function_call_parts:
                fc = fc_part.function_call
                idx = tool_index
                tool_index += 1
                yield {"event": "tool_start", "tool": fc.name, "args": dict(fc.args) if fc.args else {}, "index": idx}
                tasks.append({"name": fc.name, "args": dict(fc.args) if fc.args else {}, "index": idx})

            if len(tasks) == 1:
                task = tasks[0]
                t0 = time.time()
                result = _execute_tool(task["name"], task["args"], project_dir=self.project_dir, lang=self.lang)
                duration_ms = round((time.time() - t0) * 1000)
                images = self._collect_tool_images(task["name"], result)
                yield {"event": "tool_end", "tool": task["name"], "result": result,
                       "duration_ms": duration_ms, "index": task["index"], "images": images}
                completed_results = [(task, result)]
            else:
                def _run_tool(t: dict) -> tuple[dict, dict, int]:
                    t0 = time.time()
                    res = _execute_tool(t["name"], t["args"], project_dir=self.project_dir, lang=self.lang)
                    ms = round((time.time() - t0) * 1000)
                    return t, res, ms

                max_workers = min(len(tasks), get_key_count())
                completed_results = []
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(_run_tool, t): t for t in tasks}
                    for future in as_completed(futures):
                        task, result, duration_ms = future.result()
                        images = self._collect_tool_images(task["name"], result)
                        yield {"event": "tool_end", "tool": task["name"], "result": result,
                               "duration_ms": duration_ms, "index": task["index"], "images": images}
                        completed_results.append((task, result))

            # Add function responses to history in original order
            by_index = {t["index"]: r for t, r in completed_results}
            function_response_parts = [
                Part.from_function_response(name=t["name"], response=by_index[t["index"]])
                for t in tasks
            ]
            self.history.append(Content(role="user", parts=function_response_parts))

            # If propose_plan was called, stop
            if any(t["name"] == "propose_plan" for t in tasks):
                break

        return tool_index

    # ---- Main chat_stream ----

    def chat_stream(
        self, message: str, image_paths: list[str] | None = None
    ) -> Generator[dict, None, None]:
        """处理一轮对话，流式返回事件。"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # If plan is paused and user sends a message, mark active steps done
        if self.active_plan and self.plan_paused:
            self.plan_paused = False
            for s in self.active_plan:
                if s["status"] == "active":
                    s["status"] = "done"
                    yield {"event": "step_done", "step_id": s["id"], "cursor": 0}

        config = self._build_config()
        tool_index = 0

        # Run initial AI round with the user's message
        tool_index = yield from self._run_ai_round(message, config, tool_index, image_paths)

        # ---- Plan execution loop (DAG-based) ----
        if self.active_plan and not self.plan_paused:
            yield from self._execute_plan(config, tool_index)
            return

        yield {"event": "done"}

    def _execute_plan(self, config: GenerateContentConfig, tool_index: int) -> Generator[dict, None, None]:
        """Execute plan steps respecting dependencies. Concurrent where possible."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import queue as queue_module

        max_iterations = 30  # safety limit

        for _ in range(max_iterations):
            if not self._plan_has_pending():
                yield {"event": "plan_done"}
                yield {"event": "done"}
                return

            runnable = self._plan_runnable()
            if not runnable:
                # No steps can run (shouldn't happen if deps are correct)
                yield {"event": "plan_done"}
                yield {"event": "done"}
                return

            # Separate gate steps from auto steps
            gate_steps = [s for s in runnable if self._plan_should_gate(s)]
            auto_steps = [s for s in runnable if not self._plan_should_gate(s)]

            # If there are gate steps, pause BEFORE executing them
            if gate_steps and not auto_steps:
                # All runnable steps are gates — pause at first one
                self.plan_paused = True
                yield {"event": "plan_gate", "step": gate_steps[0], "cursor": 0}
                yield {"event": "done"}
                return

            if gate_steps and auto_steps:
                # Mix of gates and auto — run auto steps first, gate after
                pass  # auto_steps will run below, gates will be picked up next iteration

            if not auto_steps:
                break

            # Execute auto steps concurrently
            # Each step gets its own AI call in a thread
            if len(auto_steps) == 1:
                # Single step — run in main thread (no overhead)
                step = auto_steps[0]
                step["status"] = "active"
                yield {"event": "step_start", "step_id": step["id"], "cursor": 0}

                instruction = f"继续执行计划步骤 {step['id']}: {step['label']}"
                tool_index = yield from self._run_ai_round(instruction, config, tool_index)

                step["status"] = "done"
                yield {"event": "step_done", "step_id": step["id"], "cursor": 0}
            else:
                # Multiple independent steps — run concurrently
                for s in auto_steps:
                    s["status"] = "active"
                    yield {"event": "step_start", "step_id": s["id"], "cursor": 0}

                # Each concurrent step runs a full AI round in its own thread
                # We collect events via a queue since generators can't yield from threads
                event_queue: queue_module.Queue[dict | None] = queue_module.Queue()

                def _run_step(step: dict) -> dict:
                    """Run one plan step's AI round, collecting events into queue."""
                    instruction = f"继续执行计划步骤 {step['id']}: {step['label']}"
                    events: list[dict] = []
                    # Run a non-streaming AI call for concurrent steps
                    client = get_genai_client(timeout=180_000)
                    try:
                        resp = client.models.generate_content(
                            model=MODEL, contents=self.history + [
                                Content(role="user", parts=[Part.from_text(text=instruction)])
                            ], config=config,
                        )
                    except Exception as e:
                        return {"step": step, "error": str(e), "tool_results": [], "events": []}

                    # Extract function calls from response
                    tool_results = []
                    text_parts = []
                    if resp.candidates:
                        for part in resp.candidates[0].content.parts:
                            if part.function_call is not None:
                                fc = part.function_call
                                tool_name = fc.name
                                tool_args = dict(fc.args) if fc.args else {}
                                t0 = time.time()
                                result = _execute_tool(tool_name, tool_args, project_dir=self.project_dir, lang=self.lang)
                                duration_ms = round((time.time() - t0) * 1000)
                                images = self._collect_tool_images(tool_name, result)
                                events.append({"event": "tool_start", "tool": tool_name, "args": tool_args, "index": -1})
                                events.append({"event": "tool_end", "tool": tool_name, "result": result,
                                               "duration_ms": duration_ms, "index": -1, "images": images})
                                tool_results.append((tool_name, result, part))
                            elif part.text:
                                text_parts.append(part.text)
                                events.append({"event": "text_delta", "delta": part.text})

                    return {"step": step, "events": events, "tool_results": tool_results,
                            "text_parts": text_parts, "response": resp}

                max_workers = min(len(auto_steps), get_key_count())
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(_run_step, s): s for s in auto_steps}
                    for future in as_completed(futures):
                        result = future.result()
                        step = result["step"]

                        # Assign tool_index to events
                        for evt in result["events"]:
                            if evt.get("index") == -1:
                                evt["index"] = tool_index
                                tool_index += 1
                            yield evt

                        # Update history with this step's conversation
                        if result.get("response") and result["response"].candidates:
                            # Add model response
                            self.history.append(result["response"].candidates[0].content)
                            # Add function responses
                            if result["tool_results"]:
                                fr_parts = [
                                    Part.from_function_response(name=tn, response=tr)
                                    for tn, tr, _ in result["tool_results"]
                                ]
                                self.history.append(Content(role="user", parts=fr_parts))

                        step["status"] = "done"
                        yield {"event": "step_done", "step_id": step["id"], "cursor": 0}

        # If we exit the loop, plan is done or stuck
        if not self._plan_has_pending():
            yield {"event": "plan_done"}
        yield {"event": "done"}

    @staticmethod
    def _collect_tool_images(tool_name: str, result: dict) -> list[dict] | None:
        """Extract generated image info from tool result with mtime cache-busting."""
        images = None
        if "output_path" in result:
            out_path = Path(result["output_path"])
            if out_path.exists():
                rel = out_path.relative_to(PROJECT_ROOT)
                mtime = int(out_path.stat().st_mtime)
                images = [{"path": str(out_path), "url": f"/files/{rel}?v={mtime}", "tool": tool_name}]
        if "faces" in result and isinstance(result["faces"], list):
            face_images = []
            for face_info in result["faces"]:
                cp = face_info.get("crop_path")
                if cp:
                    cp_path = Path(cp)
                    if cp_path.exists():
                        rel = cp_path.relative_to(PROJECT_ROOT)
                        mtime = int(cp_path.stat().st_mtime)
                        face_images.append({"path": cp, "url": f"/files/{rel}?v={mtime}", "tool": tool_name})
            if face_images:
                images = (images or []) + face_images
        return images
