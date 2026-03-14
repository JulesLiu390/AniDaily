"""人脸/人物检测工具 - 使用 InsightFace 检测图片/视频中的人脸。

用法：
    from src.tools.person_detector import detect_faces, detect_faces_in_video

    # 图片检测
    faces = detect_faces("input/photo.jpg")

    # 视频检测
    results = detect_faces_in_video("input/video.mp4")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# 全局模型缓存
_app = None


def _get_app():
    """懒加载 InsightFace FaceAnalysis 模型。"""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


@dataclass
class FaceBox:
    """检测到的人脸。"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    age: int | None = None
    gender: str | None = None  # "M" or "F"
    embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class FrameFaces:
    """单帧的人脸检测结果。"""
    frame_index: int
    faces: list[FaceBox]


def detect_faces(
    image: str | Path,
    confidence: float = 0.5,
    save_annotated: bool = False,
    output_path: str | Path | None = None,
) -> list[FaceBox]:
    """检测图片中的人脸。

    Args:
        image: 图片路径。
        confidence: 置信度阈值。
        save_annotated: 是否保存标注后的图片。
        output_path: 标注图片输出路径（save_annotated=True 时有效）。

    Returns:
        检测到的人脸列表。
    """
    image = Path(image)
    if not image.exists():
        raise FileNotFoundError(f"图片不存在: {image}")

    img = cv2.imread(str(image))
    if img is None:
        raise RuntimeError(f"无法读取图片: {image}")

    app = _get_app()
    raw_faces = app.get(img)

    faces = _extract_faces(raw_faces, confidence)
    logger.info(f"检测到 {len(faces)} 张人脸: {image.name}")

    if save_annotated and faces:
        _save_annotated(img, faces, image, output_path)

    return faces


def detect_faces_in_video(
    video: str | Path,
    confidence: float = 0.5,
    sample_fps: float | None = None,
) -> list[FrameFaces]:
    """检测视频中每帧的人脸。

    Args:
        video: 视频路径。
        confidence: 置信度阈值。
        sample_fps: 采样帧率。None 表示处理所有帧。
            例如设为 1.0 则每秒采样 1 帧。

    Returns:
        每帧的人脸检测结果列表。
    """
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(f"视频不存在: {video}")

    app = _get_app()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(video_fps / sample_fps) if sample_fps else 1

    all_results: list[FrameFaces] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            raw_faces = app.get(frame)
            faces = _extract_faces(raw_faces, confidence)
            if faces:
                all_results.append(FrameFaces(frame_index=frame_idx, faces=faces))

        frame_idx += 1

    cap.release()
    total_faces = sum(len(ff.faces) for ff in all_results)
    logger.info(f"视频共 {frame_idx} 帧，{len(all_results)} 帧检测到人脸，共 {total_faces} 个检测框")

    return all_results


def crop_faces(
    image: str | Path,
    output_dir: str | Path,
    confidence: float = 0.5,
    padding: float = 0.3,
) -> list[Path]:
    """检测并裁剪出图片中的所有人脸。

    Args:
        image: 图片路径。
        output_dir: 裁剪图片输出目录。
        confidence: 置信度阈值。
        padding: 边界框扩展比例（0.3 = 30%，人脸裁剪建议大一些）。

    Returns:
        裁剪后的图片路径列表。
    """
    image = Path(image)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    faces = detect_faces(image, confidence=confidence)
    if not faces:
        return []

    img = Image.open(image)
    w, h = img.size
    cropped_paths: list[Path] = []

    for i, face in enumerate(faces):
        pad_x = face.width * padding
        pad_y = face.height * padding
        x1 = max(0, int(face.x1 - pad_x))
        y1 = max(0, int(face.y1 - pad_y))
        x2 = min(w, int(face.x2 + pad_x))
        y2 = min(h, int(face.y2 + pad_y))

        cropped = img.crop((x1, y1, x2, y2))
        if cropped.mode == "RGBA":
            cropped = cropped.convert("RGB")
        out_path = output_dir / f"{image.stem}_face_{i}.jpg"
        cropped.save(out_path, quality=95)
        cropped_paths.append(out_path)

    logger.info(f"裁剪了 {len(cropped_paths)} 张人脸: {output_dir}")
    return cropped_paths


def compare_faces(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """比较两张人脸的相似度（余弦相似度）。

    Args:
        embedding1: 第一张人脸的 embedding。
        embedding2: 第二张人脸的 embedding。

    Returns:
        相似度分数（0~1），越高越相似。
    """
    sim = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    return float(sim)


def _extract_faces(raw_faces, confidence: float) -> list[FaceBox]:
    """从 InsightFace 结果中提取人脸。"""
    faces = []
    for face in raw_faces:
        det_score = float(face.det_score)
        if det_score < confidence:
            continue
        bbox = face.bbox.tolist()
        faces.append(FaceBox(
            x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
            confidence=det_score,
            age=int(face.age) if hasattr(face, "age") else None,
            gender="M" if getattr(face, "gender", None) == 1 else "F" if getattr(face, "gender", None) == 0 else None,
            embedding=face.normed_embedding if hasattr(face, "normed_embedding") else None,
        ))
    return faces


def _save_annotated(
    img: np.ndarray,
    faces: list[FaceBox],
    image_path: Path,
    output_path: str | Path | None,
) -> Path:
    """保存标注后的图片。"""
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_annotated{image_path.suffix}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotated = img.copy()
    for face in faces:
        x1, y1, x2, y2 = int(face.x1), int(face.y1), int(face.x2), int(face.y2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{face.confidence:.2f}"
        if face.age is not None:
            label += f" age:{face.age}"
        if face.gender is not None:
            label += f" {face.gender}"
        cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(str(output_path), annotated)
    logger.info(f"标注图片已保存: {output_path}")
    return output_path
