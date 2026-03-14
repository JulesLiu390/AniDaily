"""测试 InsightFace 人脸检测。"""

from pathlib import Path

from src.tools.gemini_image import generate_image
from src.tools.person_detector import detect_faces, crop_faces


def test_detect_faces():
    output_dir = Path(__file__).parent / "test_person_detector_output"
    output_dir.mkdir(exist_ok=True)

    # 先用 Gemini 生成一张有人物的图片
    image_path = output_dir / "scene.jpg"
    if not image_path.exists():
        generate_image(
            prompt="A realistic photo of three young people standing together in a park, front-facing, clear faces, natural lighting",
            output_path=image_path,
        )

    # 检测人脸
    faces = detect_faces(
        image_path,
        save_annotated=True,
        output_path=output_dir / "scene_annotated.jpg",
    )
    print(f"检测到 {len(faces)} 张人脸:")
    for i, f in enumerate(faces):
        print(f"  [{i}] confidence={f.confidence:.2f} age={f.age} gender={f.gender} box=({f.x1:.0f},{f.y1:.0f},{f.x2:.0f},{f.y2:.0f})")
        if f.embedding is not None:
            print(f"       embedding shape={f.embedding.shape}")

    # 裁剪人脸
    cropped = crop_faces(image_path, output_dir / "cropped")
    print(f"裁剪了 {len(cropped)} 张人脸:")
    for path in cropped:
        print(f"  {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    test_detect_faces()
