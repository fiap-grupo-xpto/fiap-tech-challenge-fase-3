from __future__ import annotations

import numpy as np


KEYPOINT_COUNT = 17
EXERCISE_COUNT = 6
FEATURES_PER_KEYPOINT = 9
FEATURE_VERSION = "coco17-temporal-v1"


def normalize_pose(
    keypoints: np.ndarray,
    width: int,
    height: int,
    confidence_threshold: float = 0.2,
) -> np.ndarray:
    points = np.asarray(keypoints, dtype=np.float32)
    if points.shape != (KEYPOINT_COUNT, 3):
        raise ValueError(f"Expected pose shape (17, 3), received {points.shape}.")
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    if not np.isfinite(points).all():
        raise ValueError("Pose keypoints must contain finite values.")

    normalized = np.zeros_like(points)
    confidence = np.clip(points[:, 2], 0.0, 1.0)
    visible = confidence >= confidence_threshold
    normalized[:, 2] = confidence
    if not visible.any():
        return normalized

    coordinates = points[visible, :2]
    shoulder_hip_indices = np.asarray([5, 6, 11, 12])
    torso_visible = visible[shoulder_hip_indices]
    if torso_visible.any():
        center = points[shoulder_hip_indices[torso_visible], :2].mean(axis=0)
    else:
        center = coordinates.mean(axis=0)

    extent = np.ptp(coordinates, axis=0)
    scale = max(float(extent.max()), 1.0)
    normalized[visible, 0] = (points[visible, 0] - center[0]) / scale
    normalized[visible, 1] = (points[visible, 1] - center[1]) / scale
    return normalized


def temporal_features(sequence: np.ndarray, exercise_id: int) -> np.ndarray:
    poses = np.asarray(sequence, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (KEYPOINT_COUNT, 3):
        raise ValueError(
            f"Expected pose sequence with shape (frames, 17, 3), received {poses.shape}."
        )
    if len(poses) < 2:
        raise ValueError("At least two pose frames are required.")
    if not np.isfinite(poses).all():
        raise ValueError("Pose sequence must contain finite values.")
    if exercise_id not in range(1, EXERCISE_COUNT + 1):
        raise ValueError(f"Unsupported exercise_id: {exercise_id}")

    per_joint: list[float] = []
    for joint_index in range(KEYPOINT_COUNT):
        joint = poses[:, joint_index, :]
        visible = joint[:, 2] >= 0.2
        if visible.any():
            coordinates = joint[visible, :2]
            mean = coordinates.mean(axis=0)
            standard_deviation = coordinates.std(axis=0)
            movement = np.abs(np.diff(joint[:, :2], axis=0))
            moving_pairs = visible[1:] & visible[:-1]
            mean_movement = (
                movement[moving_pairs].mean(axis=0)
                if moving_pairs.any()
                else np.zeros(2, dtype=np.float32)
            )
            extent = np.ptp(coordinates, axis=0)
            confidence_mean = joint[visible, 2].mean()
        else:
            mean = standard_deviation = mean_movement = extent = np.zeros(
                2, dtype=np.float32
            )
            confidence_mean = 0.0
        per_joint.extend(
            [
                float(mean[0]),
                float(mean[1]),
                float(standard_deviation[0]),
                float(standard_deviation[1]),
                float(extent[0]),
                float(extent[1]),
                float(mean_movement[0]),
                float(mean_movement[1]),
                float(confidence_mean),
            ]
        )

    exercise_one_hot = np.zeros(EXERCISE_COUNT, dtype=np.float32)
    exercise_one_hot[exercise_id - 1] = 1.0
    return np.concatenate(
        [np.asarray(per_joint, dtype=np.float32), exercise_one_hot]
    )
