"""最小测试：用 Gemini 对话式编辑改变场景角度。

场景1 (原图) -> edit_image("Show this exact scene from a reverse angle") -> 场景2

运行: uv run python tests/test_scene_angle.py
"""

from pathlib import Path

from src.tools.gemini_image import edit_image

# 用现有的场景图做测试
INPUT = Path("projects/jojo test/output/storyboards/scenes/clip_1_scene.png")
OUTPUT_DIR = Path("tests/test_scene_angle_output")

# 测试 Pro 模型
PRO_MODEL = "gemini-3-pro-image-preview"


def test_reverse_angle():
    """测试：同一场景，反向视角。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "scene_reverse.png"

    result = edit_image(
        image_path=INPUT,
        prompt=(
            "Show this exact same scene from the REVERSE angle — "
            "as if the camera turned 180 degrees to face the opposite direction. "
            "Keep the same art style, lighting, and time of day. "
            "The scene should look like the same location but viewed from the other end of the street."
        ),
        output_path=out,
    )

    assert result.exists(), "输出文件不存在"
    assert result.stat().st_size > 10000, "输出文件太小"
    print(f"✓ 反向视角: {result} ({result.stat().st_size // 1024}KB)")


def test_low_angle():
    """测试：同一场景，低角度仰视。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "scene_low_angle.png"

    result = edit_image(
        image_path=INPUT,
        prompt=(
            "Show this exact same scene from a LOW ANGLE perspective — "
            "camera placed near ground level, looking upward. "
            "The buildings should tower above, sky more prominent. "
            "Keep the same art style, lighting, and location."
        ),
        output_path=out,
    )

    assert result.exists(), "输出文件不存在"
    assert result.stat().st_size > 10000, "输出文件太小"
    print(f"✓ 低角度仰视: {result} ({result.stat().st_size // 1024}KB)")


def test_side_angle():
    """测试：同一场景，侧面视角。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "scene_side.png"

    result = edit_image(
        image_path=INPUT,
        prompt=(
            "Show this exact same scene from a SIDE ANGLE — "
            "camera positioned to the right side, showing the street from a 90-degree side view. "
            "We should see one row of buildings in the foreground and the street extending to the left. "
            "Keep the same art style, lighting, and location."
        ),
        output_path=out,
    )

    assert result.exists(), "输出文件不存在"
    assert result.stat().st_size > 10000, "输出文件太小"
    print(f"✓ 侧面视角: {result} ({result.stat().st_size // 1024}KB)")


def test_pro_reverse_angle():
    """测试 Pro 模型：反向视角。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "pro_scene_reverse.png"

    result = edit_image(
        image_path=INPUT,
        prompt=(
            "Show this exact same scene from the REVERSE angle — "
            "as if the camera turned 180 degrees to face the opposite direction. "
            "Keep the same art style, lighting, and time of day. "
            "The scene should look like the same location but viewed from the other end of the street."
        ),
        output_path=out,
        model=PRO_MODEL,
    )

    assert result.exists()
    print(f"✓ Pro 反向视角: {result} ({result.stat().st_size // 1024}KB)")


def test_pro_low_angle():
    """测试 Pro 模型：低角度仰视。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "pro_scene_low_angle.png"

    result = edit_image(
        image_path=INPUT,
        prompt=(
            "Show this exact same scene from a LOW ANGLE perspective — "
            "camera placed near ground level, looking upward. "
            "The buildings should tower above, sky more prominent. "
            "Keep the same art style, lighting, and location."
        ),
        output_path=out,
        model=PRO_MODEL,
    )

    assert result.exists()
    print(f"✓ Pro 低角度仰视: {result} ({result.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    if not INPUT.exists():
        print(f"跳过: 测试输入图不存在 {INPUT}")
        exit(0)

    print(f"输入场景: {INPUT}")
    print("--- Pro 模型测试 ---")
    test_pro_reverse_angle()
    test_pro_low_angle()
    print("---")
    print(f"所有输出在: {OUTPUT_DIR}/")
    print("请目视对比原图和输出图，确认角度是否变化。")
