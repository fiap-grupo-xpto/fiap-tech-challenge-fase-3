from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from backend.rehab_video.features import KEYPOINT_COUNT, normalize_pose


class PoseEstimator(Protocol):
    def estimate_frames(
        self, frames: Sequence[np.ndarray], batch_size: int = 16
    ) -> np.ndarray: ...


class YoloPoseEstimator:
    def __init__(
        self,
        weights: str = "yolov8n-pose.pt",
        image_size: int = 384,
        confidence_threshold: float = 0.25,
        device: str = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for pose extraction. Install requirements.txt."
            ) from exc
        if image_size < 128:
            raise ValueError("YOLO image_size must be at least 128 pixels.")
        if not 0.0 < confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        self.model = YOLO(weights)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.device = device

    def estimate_frames(
        self, frames: Sequence[np.ndarray], batch_size: int = 16
    ) -> np.ndarray:
        if len(frames) == 0:
            return np.empty((0, KEYPOINT_COUNT, 3), dtype=np.float32)
        poses = [
            normalize_pose(
                pixels,
                width=frame.shape[1],
                height=frame.shape[0],
                confidence_threshold=self.confidence_threshold,
            )
            for frame, pixels in zip(frames, self.detect_pixel_keypoints(frames, batch_size))
        ]
        return np.stack(poses).astype(np.float32, copy=False)

    def detect_pixel_keypoints(
        self, frames: Sequence[np.ndarray], batch_size: int = 16
    ) -> list[np.ndarray]:
        """Return the main person's COCO-17 keypoints (x, y, conf) in pixels."""
        if len(frames) == 0:
            return []
        results = self.model.predict(
            source=list(frames),
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            device=self.device,
            batch=batch_size,
            verbose=False,
            save=False,
        )
        if len(results) != len(frames):
            raise RuntimeError(
                f"Pose model returned {len(results)} results for {len(frames)} frames."
            )

        poses: list[np.ndarray] = []
        for result in results:
            keypoints = getattr(result, "keypoints", None)
            if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
                poses.append(np.zeros((KEYPOINT_COUNT, 3), dtype=np.float32))
                continue

            candidates = keypoints.data.cpu().numpy()
            if candidates.ndim != 3 or candidates.shape[1:] != (KEYPOINT_COUNT, 3):
                raise RuntimeError(
                    f"Expected COCO-17 poses, received shape {candidates.shape}."
                )
            boxes = getattr(result, "boxes", None)
            if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy):
                box_array = boxes.xyxy.cpu().numpy()
                confidence = boxes.conf.cpu().numpy()
                area = np.maximum(0, box_array[:, 2] - box_array[:, 0]) * np.maximum(
                    0, box_array[:, 3] - box_array[:, 1]
                )
                selected = int(np.argmax(area * confidence))
            else:
                selected = 0

            poses.append(candidates[selected].astype(np.float32, copy=False))
        return poses


def sample_video_frames(
    capture,
    start_frame: int,
    end_frame: int,
    sample_count: int,
) -> list[np.ndarray]:
    import cv2

    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("Frame range must satisfy 0 <= start_frame <= end_frame.")
    if sample_count < 2:
        raise ValueError("At least two frames must be sampled per repetition.")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame >= frame_count:
        raise ValueError(
            f"End frame {end_frame} is outside the video ({frame_count} frames)."
        )

    indices = np.linspace(start_frame, end_frame, sample_count, dtype=int)
    frames: list[np.ndarray] = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if not success or frame is None:
            raise RuntimeError(f"Could not decode video frame {frame_index}.")
        frames.append(frame)
    return frames


def extract_pose_sequence(
    video_path: Path | str,
    start_frame: int,
    end_frame: int | None,
    estimator: PoseEstimator,
    sample_count: int = 12,
) -> np.ndarray:
    import cv2

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video file: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_end = frame_count - 1 if end_frame is None else end_frame
        frames = sample_video_frames(capture, start_frame, actual_end, sample_count)
        poses = estimator.estimate_frames(frames)
    finally:
        capture.release()

    if poses.shape != (sample_count, KEYPOINT_COUNT, 3):
        raise RuntimeError(
            f"Pose estimator returned invalid sequence shape {poses.shape}."
        )
    if not np.any(poses[:, :, 2] >= 0.2):
        raise ValueError("No person pose was detected in the selected video segment.")
    return poses
