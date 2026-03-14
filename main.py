"""AniDaily 主流程。

用法：
    uv run python main.py

输入结构：
    input/
    ├── 2024-01-15/          ← 事件文件夹（日期/名称）
    │   └── photo_001.png
    ├── 2024-02-10_2024-02-11/
    │   ├── photo_003.png
    │   └── photo_004.png
    └── ...

流程：
    1. 扫描 input/ 下所有事件子文件夹
    2. 汇总所有图片 → InsightFace 检测 + DBSCAN 聚类
    3. 每个人物选最佳配对图，裁剪所有人脸
    4. 风格化生成角色形象
    5. 场景去重 → 去人 → 动画化
    6. (TODO) 动画分镜
"""

import logging
from pathlib import Path

from src.tools.face_matcher import match_and_crop
from src.tools.face_stylizer import stylize_face
from src.tools.scene_analyzer import analyze_scenes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(input_dir: Path) -> dict[str, list[Path]]:
    """扫描 input 下的事件子文件夹，返回 {事件名: [图片路径]} 映射。"""
    events: dict[str, list[Path]] = {}

    for item in sorted(input_dir.iterdir()):
        if not item.is_dir():
            continue
        images = sorted(
            f for f in item.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        )
        if images:
            events[item.name] = images

    return events


def main():
    # 1. 扫描事件
    events = collect_images(INPUT_DIR)
    if not events:
        logger.error(f"请将图片放入 {INPUT_DIR}/<事件名>/ 子文件夹中")
        return

    all_images: list[Path] = []
    logger.info(f"找到 {len(events)} 个事件:")
    for event_name, images in events.items():
        logger.info(f"  {event_name}: {len(images)} 张图片")
        all_images.extend(images)

    logger.info(f"共 {len(all_images)} 张图片")

    # 2-3. 检测人脸 → 聚类 → 选最佳配对 → 裁剪
    faces_dir = OUTPUT_DIR / "faces"
    persons = match_and_crop(all_images, output_dir=faces_dir)

    if not persons:
        logger.warning("未检测到任何人物")
        return

    logger.info(f"\n{'='*50}")
    logger.info(f"共识别出 {len(persons)} 个独立人物:")
    for p in persons:
        event_names = set()
        for occ in p.occurrences:
            event_names.add(occ.image_path.parent.name)
        logger.info(
            f"  人物 {p.person_id}: 出现 {p.appearance_count} 次, "
            f"跨 {len(event_names)} 个事件, "
            f"最佳图 {p.best_image_path.name}, "
            f"人脸 {p.best_face_crop_path}"
        )

    # 4. 风格化：人脸 + 原图 → 动画角色形象
    stylized_dir = OUTPUT_DIR / "stylized"
    stylized_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*50}")
    logger.info("开始生成动画角色形象...")

    for p in persons:
        if p.best_face_crop_path is None:
            logger.warning(f"人物 {p.person_id}: 无裁剪人脸，跳过风格化")
            continue

        output_path = stylized_dir / f"person_{p.person_id}.png"
        try:
            stylize_face(
                face_path=p.best_face_crop_path,
                output_path=output_path,
                original_image_path=p.best_image_path,
            )
            logger.info(f"人物 {p.person_id}: 风格化完成 → {output_path}")
        except Exception as e:
            logger.error(f"人物 {p.person_id}: 风格化失败: {e}")

    # 5. 场景分析：去重 → 去人 → 动画化
    scenes_dir = OUTPUT_DIR / "scenes"
    logger.info(f"\n{'='*50}")
    logger.info("开始场景分析...")

    analysis, stylized_scene_paths = analyze_scenes(all_images, output_dir=scenes_dir)

    logger.info(f"\n{'='*50}")
    logger.info(f"场景分析完成: {len(analysis.scenes)} 个独立场景")
    for s in analysis.scenes:
        logger.info(
            f"  场景 {s.scene_id}: {s.description} "
            f"({s.location_type}, {s.time_of_day}, {s.mood})"
        )
    logger.info(f"动画化场景: {len(stylized_scene_paths)} 张")
    logger.info(f"场景 JSON: {scenes_dir / 'scenes.json'}")


if __name__ == "__main__":
    main()
