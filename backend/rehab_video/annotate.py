from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from backend.rehab_video.features import KEYPOINT_COUNT


# COCO-17 limb pairs: head, arms, torso and legs.
SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
DEFAULT_MAX_FRAMES = 900


class PixelPoseEstimator(Protocol):
    def detect_pixel_keypoints(
        self, frames: Sequence[np.ndarray], batch_size: int = 16
    ) -> list[np.ndarray]: ...


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    confidence_threshold: float = 0.25,
    caption: str | None = None,
) -> np.ndarray:
    """Draw COCO-17 limbs and joints on a copy of a BGR frame."""
    import cv2

    points = np.asarray(keypoints, dtype=np.float32)
    if points.shape != (KEYPOINT_COUNT, 3):
        raise ValueError(f"Expected keypoints shape (17, 3), received {points.shape}.")

    canvas = frame.copy()
    thickness = max(2, round(min(canvas.shape[:2]) / 250))
    visible = points[:, 2] >= confidence_threshold
    for start, end in SKELETON_EDGES:
        if visible[start] and visible[end]:
            cv2.line(
                canvas,
                (int(points[start, 0]), int(points[start, 1])),
                (int(points[end, 0]), int(points[end, 1])),
                (0, 200, 255),
                thickness,
                cv2.LINE_AA,
            )
    for index in np.flatnonzero(visible):
        cv2.circle(
            canvas,
            (int(points[index, 0]), int(points[index, 1])),
            thickness + 1,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )
    if caption:
        cv2.putText(
            canvas,
            caption,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def annotate_video(
    video_path: Path | str,
    output_path: Path | str,
    estimator: PixelPoseEstimator,
    start_frame: int = 0,
    end_frame: int | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    confidence_threshold: float = 0.25,
    label: str | None = None,
) -> dict:
    """Write a copy of the selected segment with the estimated skeleton drawn."""
    import cv2

    if start_frame < 0 or (end_frame is not None and end_frame < start_frame):
        raise ValueError("Frame range must satisfy 0 <= start_frame <= end_frame.")
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video file: {video_path}")
    writer = None
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        last = frame_count - 1 if end_frame is None else end_frame
        if start_frame >= frame_count or last >= frame_count:
            raise ValueError(
                f"Frame range {start_frame}-{last} is outside the video "
                f"({frame_count} frames)."
            )
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stride = max(1, math.ceil((last - start_frame + 1) / max_frames))

        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create the annotated video file.")

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        written = 0
        detected = 0
        batch: list[np.ndarray] = []
        batch_indices: list[int] = []

        def flush() -> None:
            nonlocal written, detected
            if not batch:
                return
            for frame, index, pose in zip(
                batch, batch_indices, estimator.detect_pixel_keypoints(batch)
            ):
                detected += int(bool(np.any(pose[:, 2] >= confidence_threshold)))
                caption = f"{label} | frame {index}" if label else f"frame {index}"
                writer.write(draw_skeleton(frame, pose, confidence_threshold, caption))
                written += 1
            batch.clear()
            batch_indices.clear()

        for index in range(start_frame, last + 1):
            success, frame = capture.read()
            if not success or frame is None:
                break
            if (index - start_frame) % stride:
                continue
            batch.append(frame)
            batch_indices.append(index)
            if len(batch) >= 16:
                flush()
        flush()
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if written == 0:
        raise ValueError("No frames could be read from the selected video segment.")
    if detected == 0:
        raise ValueError("No person pose was detected in the selected video segment.")
    return {
        "frames_written": written,
        "frames_with_pose": detected,
        "frame_stride": stride,
        "fps": fps / stride,
    }


def transcode_to_h264(path: Path | str) -> bool:
    """Re-encode in place to H.264 so browsers can play it; keep the original on failure."""
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return False

    source = Path(path)
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "h264.mp4"
        completed = subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(target),
            ],
            capture_output=True,
            timeout=300,
        )
        if completed.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
            return False
        shutil.move(str(target), str(source))
    return True
