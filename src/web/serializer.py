"""Gemini Content / Part JSON serializer for SQLite persistence."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from google.genai.types import Content, FunctionCall, FunctionResponse, Part

logger = logging.getLogger(__name__)

# Pattern used by the agent to mark image file paths before inline_data parts.
_IMAGE_PATH_RE = re.compile(r"^\[图片路径: (.+)\]$")


def _guess_mime(path: Path) -> str:
    """Return a MIME type for common image extensions."""
    suffix = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")


# ---------------------------------------------------------------------------
# Part  <-->  dict
# ---------------------------------------------------------------------------

def part_to_dict(part: Part) -> dict[str, Any] | None:
    """Serialize a single Part to a JSON-safe dict.

    Returns None for inline_data parts (raw bytes are skipped; the preceding
    ``[图片路径: …]`` text part is enough to reconstruct it on restore).
    """
    if part.text is not None:
        return {"type": "text", "text": part.text}

    if part.function_call is not None:
        fc: FunctionCall = part.function_call
        return {
            "type": "function_call",
            "name": fc.name,
            "args": fc.args or {},
        }

    if part.function_response is not None:
        fr: FunctionResponse = part.function_response
        return {
            "type": "function_response",
            "name": fr.name,
            "response": fr.response or {},
        }

    if part.inline_data is not None:
        # Skip raw bytes – the text marker is preserved separately.
        return None

    # Unsupported part type (thought, executable_code, etc.) – skip.
    logger.debug("Skipping unsupported part type during serialization: %s", part)
    return None


def dict_to_part(d: dict[str, Any]) -> Part | None:
    """Deserialize a dict back into a Part.

    For text parts that match the ``[图片路径: …]`` pattern, we attempt to
    re-read the image file.  If the file still exists the text Part **and**
    a new inline_data Part are returned (as a 2-element list via the caller).
    If the file is gone, the text Part is returned as-is.

    Returns None only when the dict cannot be interpreted.
    """
    ptype = d.get("type")

    if ptype == "text":
        text: str = d.get("text", "")
        return Part.from_text(text=text)

    if ptype == "function_call":
        return Part.from_function_call(
            name=d["name"],
            args=d.get("args", {}),
        )

    if ptype == "function_response":
        return Part.from_function_response(
            name=d["name"],
            response=d.get("response", {}),
        )

    logger.warning("Unknown part type during deserialization: %s", ptype)
    return None


def _maybe_restore_image(text: str) -> Part | None:
    """If *text* is an image-path marker and the file exists, return an
    inline_data Part with the file bytes.  Otherwise return None.
    """
    m = _IMAGE_PATH_RE.match(text)
    if m is None:
        return None
    path = Path(m.group(1))
    if not path.exists():
        return None
    mime = _guess_mime(path)
    return Part.from_bytes(data=path.read_bytes(), mime_type=mime)


# ---------------------------------------------------------------------------
# Content  <-->  dict
# ---------------------------------------------------------------------------

def content_to_dict(content: Content) -> dict[str, Any]:
    """Serialize a Content message to a JSON-safe dict."""
    parts_list: list[dict[str, Any]] = []
    for part in content.parts or []:
        d = part_to_dict(part)
        if d is not None:
            parts_list.append(d)
    return {
        "role": content.role or "user",
        "parts": parts_list,
    }


def dict_to_content(d: dict[str, Any]) -> Content:
    """Deserialize a dict back into a Content message.

    Image-path marker texts are detected automatically.  If the referenced
    file still exists on disk, both the text marker Part and a fresh
    inline_data Part are inserted (mirroring the original structure).
    """
    role: str = d.get("role", "user")
    raw_parts: list[dict[str, Any]] = d.get("parts", [])

    parts: list[Part] = []
    for pd in raw_parts:
        part = dict_to_part(pd)
        if part is None:
            continue
        parts.append(part)

        # If this is the image-path marker, try to restore the inline_data.
        if pd.get("type") == "text":
            img_part = _maybe_restore_image(pd["text"])
            if img_part is not None:
                parts.append(img_part)

    return Content(role=role, parts=parts)
