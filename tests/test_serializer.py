"""Tests for src.web.serializer – Gemini Content/Part JSON round-trips."""

from google.genai.types import Content, Part

from src.web.serializer import (
    content_to_dict,
    dict_to_content,
    dict_to_part,
    part_to_dict,
)


# ── Text ────────────────────────────────────────────────────────────────

def test_text_round_trip():
    """A simple text Part survives serialization ↔ deserialization."""
    original = Part.from_text(text="你好，世界")
    d = part_to_dict(original)
    assert d == {"type": "text", "text": "你好，世界"}

    restored = dict_to_part(d)
    assert restored is not None
    assert restored.text == "你好，世界"


# ── Function call ───────────────────────────────────────────────────────

def test_function_call_round_trip():
    """A function_call Part with args round-trips correctly."""
    args = {"image_path": "/tmp/photo.jpg", "output_dir": "/tmp/faces"}
    original = Part.from_function_call(name="detect_faces_in_image", args=args)
    d = part_to_dict(original)

    assert d == {
        "type": "function_call",
        "name": "detect_faces_in_image",
        "args": args,
    }

    restored = dict_to_part(d)
    assert restored is not None
    assert restored.function_call is not None
    assert restored.function_call.name == "detect_faces_in_image"
    assert restored.function_call.args == args


def test_function_call_empty_args():
    """A function_call with no args serializes with an empty dict."""
    original = Part.from_function_call(name="list_tools", args={})
    d = part_to_dict(original)
    assert d["args"] == {}

    restored = dict_to_part(d)
    assert restored.function_call.args == {}


# ── Function response ──────────────────────────────────────────────────

def test_function_response_round_trip():
    """A function_response Part with a response dict round-trips."""
    response = {
        "faces": [
            {"bbox": [10, 20, 100, 120], "age": 25, "gender": "male"},
        ],
        "count": 1,
    }
    original = Part.from_function_response(
        name="detect_faces_in_image", response=response,
    )
    d = part_to_dict(original)

    assert d == {
        "type": "function_response",
        "name": "detect_faces_in_image",
        "response": response,
    }

    restored = dict_to_part(d)
    assert restored is not None
    assert restored.function_response is not None
    assert restored.function_response.name == "detect_faces_in_image"
    assert restored.function_response.response == response


def test_function_response_empty_response():
    """A function_response with an empty response dict."""
    original = Part.from_function_response(name="noop", response={})
    d = part_to_dict(original)
    assert d["response"] == {}

    restored = dict_to_part(d)
    assert restored.function_response.response == {}


# ── Content-level round-trip ────────────────────────────────────────────

def test_content_text_round_trip():
    """A Content with a single text part."""
    content = Content(
        role="user",
        parts=[Part.from_text(text="生成一张图")],
    )
    d = content_to_dict(content)
    assert d["role"] == "user"
    assert len(d["parts"]) == 1
    assert d["parts"][0] == {"type": "text", "text": "生成一张图"}

    restored = dict_to_content(d)
    assert restored.role == "user"
    assert len(restored.parts) == 1
    assert restored.parts[0].text == "生成一张图"


def test_content_mixed_parts():
    """A Content with text + function_call parts together (model turn)."""
    content = Content(
        role="model",
        parts=[
            Part.from_text(text="好的，我来帮你检测人脸。"),
            Part.from_function_call(
                name="detect_faces_in_image",
                args={"image_path": "/tmp/photo.jpg"},
            ),
        ],
    )
    d = content_to_dict(content)
    assert d["role"] == "model"
    assert len(d["parts"]) == 2
    assert d["parts"][0]["type"] == "text"
    assert d["parts"][1]["type"] == "function_call"

    restored = dict_to_content(d)
    assert restored.role == "model"
    assert len(restored.parts) == 2
    assert restored.parts[0].text == "好的，我来帮你检测人脸。"
    assert restored.parts[1].function_call.name == "detect_faces_in_image"


def test_content_function_response_parts():
    """A Content with multiple function_response parts (user turn)."""
    content = Content(
        role="user",
        parts=[
            Part.from_function_response(
                name="detect_faces_in_image",
                response={"count": 2},
            ),
            Part.from_function_response(
                name="stylize_character",
                response={"output": "/tmp/stylized.png"},
            ),
        ],
    )
    d = content_to_dict(content)
    assert len(d["parts"]) == 2

    restored = dict_to_content(d)
    assert len(restored.parts) == 2
    assert restored.parts[0].function_response.name == "detect_faces_in_image"
    assert restored.parts[1].function_response.name == "stylize_character"


# ── inline_data / image skipping ────────────────────────────────────────

def test_inline_data_skipped():
    """inline_data parts are dropped during serialization."""
    img_part = Part.from_bytes(data=b"\x89PNG fake", mime_type="image/png")
    d = part_to_dict(img_part)
    assert d is None


def test_content_with_inline_data_preserves_text_marker():
    """When Content has [图片路径: ...] + inline_data, only the text marker
    is kept (inline_data is dropped). On restore without the file, only the
    text marker Part comes back.
    """
    content = Content(
        role="user",
        parts=[
            Part.from_text(text="[图片路径: /nonexistent/img.png]"),
            Part.from_bytes(data=b"\x89PNG", mime_type="image/png"),
            Part.from_text(text="请帮我分析这张图"),
        ],
    )
    d = content_to_dict(content)
    # inline_data part should be gone
    assert len(d["parts"]) == 2
    assert d["parts"][0] == {"type": "text", "text": "[图片路径: /nonexistent/img.png]"}
    assert d["parts"][1] == {"type": "text", "text": "请帮我分析这张图"}

    # Restore – file doesn't exist, so no inline_data re-injected
    restored = dict_to_content(d)
    assert len(restored.parts) == 2
    assert restored.parts[0].text == "[图片路径: /nonexistent/img.png]"
    assert restored.parts[1].text == "请帮我分析这张图"


def test_content_with_real_image_file(tmp_path):
    """When the image file exists on disk, dict_to_content re-injects the
    inline_data Part after the text marker.
    """
    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"\x89PNG_FAKE_DATA")

    d = {
        "role": "user",
        "parts": [
            {"type": "text", "text": f"[图片路径: {img_file}]"},
            {"type": "text", "text": "分析一下"},
        ],
    }

    restored = dict_to_content(d)
    # Should be 3 parts: text marker, inline_data, "分析一下"
    assert len(restored.parts) == 3
    assert restored.parts[0].text == f"[图片路径: {img_file}]"
    assert restored.parts[1].inline_data is not None
    assert restored.parts[1].inline_data.data == b"\x89PNG_FAKE_DATA"
    assert restored.parts[1].inline_data.mime_type == "image/png"
    assert restored.parts[2].text == "分析一下"
