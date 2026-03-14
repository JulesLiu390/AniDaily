"""测试 InsightFace 人脸检测 - 从 input 文件夹读取图片。

用法：
    1. 将图片放入 tests/test_face_from_image/input/
    2. uv run python -m tests.test_face_from_image
"""

from pathlib import Path

from src.tools.person_detector import detect_faces, crop_faces, compare_faces

TEST_DIR = Path(__file__).parent / "test_face_from_image"
INPUT_DIR = TEST_DIR / "input"
OUTPUT_DIR = TEST_DIR / "output"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def test_face_from_image():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = [f for f in sorted(INPUT_DIR.iterdir()) if f.suffix.lower() in SUPPORTED_EXTS]
    if not images:
        print(f"请将图片放入 {INPUT_DIR}/")
        return

    for image_path in images:
        print(f"\n{'='*50}")
        print(f"处理: {image_path.name}")
        print(f"{'='*50}")

        img_output_dir = OUTPUT_DIR / image_path.stem
        img_output_dir.mkdir(parents=True, exist_ok=True)

        # 检测人脸
        faces = detect_faces(
            image_path,
            save_annotated=True,
            output_path=img_output_dir / f"{image_path.stem}_annotated.jpg",
        )
        print(f"检测到 {len(faces)} 张人脸:")
        for i, f in enumerate(faces):
            print(f"  [{i}] confidence={f.confidence:.2f} age={f.age} gender={f.gender} box=({f.x1:.0f},{f.y1:.0f},{f.x2:.0f},{f.y2:.0f})")

        # 裁剪人脸
        cropped = crop_faces(image_path, img_output_dir / "cropped")
        print(f"裁剪了 {len(cropped)} 张人脸:")
        for path in cropped:
            print(f"  {path.name} ({path.stat().st_size / 1024:.0f} KB)")

        # 人脸相似度比对
        if len(faces) >= 2:
            print("\n人脸相似度比对:")
            for i in range(len(faces)):
                for j in range(i + 1, len(faces)):
                    if faces[i].embedding is not None and faces[j].embedding is not None:
                        sim = compare_faces(faces[i].embedding, faces[j].embedding)
                        print(f"  face[{i}] vs face[{j}]: {sim:.4f}")


if __name__ == "__main__":
    test_face_from_image()
