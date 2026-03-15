"""测试：条漫 4 等分 → 16:9 分镜帧。

把一张 9:16 条漫竖图均匀切成 4 格，每格左右裁到 16:9。

运行: uv run python tests/test_comic_split.py
"""

from pathlib import Path

from PIL import Image

INPUT = Path("projects/jojo test/output/panels/nobita_vs_shizuka_p1.png")
OUTPUT_DIR = Path("tests/test_comic_split_output")


def split_comic_to_frames(
    comic_path: Path,
    output_dir: Path,
    num_panels: int = 4,
    target_ratio: float = 16 / 9,
) -> list[Path]:
    """把条漫均匀切成 num_panels 格，每格左右居中裁到 target_ratio。"""
    img = Image.open(comic_path)
    w, h = img.size
    panel_h = h // num_panels

    results = []
    for i in range(num_panels):
        top = i * panel_h
        bottom = top + panel_h

        # 左右居中裁到 16:9
        target_w = int(panel_h * target_ratio)
        if target_w > w:
            target_w = w  # 宽度不够就不裁
        left = (w - target_w) // 2
        right = left + target_w

        panel = img.crop((left, top, right, bottom))
        out_path = output_dir / f"frame_{i + 1}.png"
        panel.save(out_path)
        results.append(out_path)

    return results


if __name__ == "__main__":
    if not INPUT.exists():
        print(f"跳过: {INPUT} 不存在")
        exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img = Image.open(INPUT)
    print(f"输入: {INPUT} ({img.size[0]}x{img.size[1]})")
    print("---")

    frames = split_comic_to_frames(INPUT, OUTPUT_DIR)

    for f in frames:
        frame_img = Image.open(f)
        ratio = frame_img.size[0] / frame_img.size[1]
        print(f"  {f.name}: {frame_img.size[0]}x{frame_img.size[1]} (ratio={ratio:.2f}, 16:9={16/9:.2f})")

    print(f"---\n输出: {OUTPUT_DIR}/")
