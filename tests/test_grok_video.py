"""测试 Grok Video 3 纯 prompt 生成视频。"""

from pathlib import Path

from src.tools.video_generator import generate_video


def test_grok_generate_video():
    output_dir = Path(__file__).parent / "test_grok_video_output"
    output_dir.mkdir(exist_ok=True)
    generate_video(
        prompt="A cute anime girl walking through a cherry blossom garden in spring, gentle breeze, petals falling, anime style, soft lighting",
        mode="grok",
        aspect_ratio="16:9",
        output_dir=output_dir,
    )
    print(f"视频输出目录: {output_dir}")
    videos = list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.webm"))
    assert len(videos) > 0, "未找到生成的视频文件"
    print(f"生成的视频: {videos[0]} ({videos[0].stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    test_grok_generate_video()
