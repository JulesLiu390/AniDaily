"""测试人脸风格化 - 京阿尼画风。

从 test_face_from_image 的 output 中读取人脸和原图。
"""

from pathlib import Path

from src.tools.face_stylizer import stylize_face

FACE_FROM_IMAGE_DIR = Path(__file__).parent / "test_face_from_image"
TEST_DIR = Path(__file__).parent / "test_face_stylizer"
OUTPUT_DIR = TEST_DIR / "output"

KYOANI_PROMPT = (
    "Transform this person's photo into a Kyoto Animation (京アニ) style anime character illustration. "
    "Requirements:\n"
    "- Kyoto Animation signature art style: soft lighting, delicate facial features, detailed expressive eyes with reflections, smooth hair with natural flow\n"
    "- Keep the person's face features, hairstyle, and overall appearance recognizable\n"
    "- Generate ONLY this one person as a single character portrait, full upper body\n"
    "- Preserve this person's clothing, body type from the original scene photo\n"
    "- Pure white background (#FFFFFF)\n"
    "- Clean linework, soft pastel color palette, gentle shading\n"
    "- High detail on eyes, hair highlights, and skin tones\n"
    "- The result should look like an official Kyoto Animation character design\n"
    "Output a single stylized portrait image."
)


def test_face_stylizer():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 遍历 test_face_from_image/output 下的每个图片目录
    output_base = FACE_FROM_IMAGE_DIR / "output"
    if not output_base.exists():
        print(f"请先运行 test_face_from_image 生成人脸数据: {output_base}")
        return

    for img_dir in sorted(output_base.iterdir()):
        if not img_dir.is_dir():
            continue

        cropped_dir = img_dir / "cropped"
        if not cropped_dir.exists():
            continue

        # 找原图
        original = FACE_FROM_IMAGE_DIR / "input" / f"{img_dir.name}.png"
        if not original.exists():
            original = FACE_FROM_IMAGE_DIR / "input" / f"{img_dir.name}.jpg"
        if not original.exists():
            print(f"跳过 {img_dir.name}: 找不到原图")
            continue

        # 只取 face 0-2（按数字排序）
        import re
        all_faces = list(cropped_dir.glob("*_face_*.jpg"))
        all_faces.sort(key=lambda p: int(re.search(r"_face_(\d+)", p.stem).group(1)))
        faces = all_faces[:3]
        print(f"\n原图: {original.name}, 风格化 {len(faces)} 张人脸")

        for face_path in faces:
            output_path = OUTPUT_DIR / f"{face_path.stem}_kyoani.jpg"
            print(f"  {face_path.name} -> {output_path.name}")
            result = stylize_face(
                face_path=face_path,
                output_path=output_path,
                original_image_path=original,
                prompt=KYOANI_PROMPT,
            )
            print(f"    完成: {result.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    test_face_stylizer()
