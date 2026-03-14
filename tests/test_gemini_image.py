"""测试 Gemini 纯 prompt 生成图片。"""

from pathlib import Path

from src.tools.gemini_image import generate_image


def test_gemini_generate_image():
    output_dir = Path(__file__).parent / "test_gemini_image_output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "test_image.jpg"
    result = generate_image(
        prompt="A cute anime cat sitting on a windowsill watching the sunset, watercolor style, warm colors, cozy atmosphere",
        output_path=output_path,
    )
    assert result.exists(), f"图片未生成: {result}"
    print(f"生成的图片: {result} ({result.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    test_gemini_generate_image()
