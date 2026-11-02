from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.rehab_video.dataset import (
    RepetitionRecord,
    load_segmentation,
    split_by_person,
    validate_dataset,
    video_frame_range,
    video_path_for,
)
from backend.rehab_video.features import FEATURE_VERSION, temporal_features
from backend.rehab_video.pose import PoseEstimator, sample_video_frames


def _extract_features(
    data_root: Path,
    records: Sequence[RepetitionRecord],
    estimator: PoseEstimator,
    sample_frames: int,
    clip_batch_size: int,
    inference_batch_size: int,
) -> np.ndarray:
    import cv2

    features: list[np.ndarray | None] = [None] * len(records)
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record.exercise_id, record.video_id)].append(index)

    for (exercise_id, video_id), record_indices in sorted(grouped.items()):
        record = records[record_indices[0]]
        video_path = video_path_for(data_root, record)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open dataset video: {video_path}")
        try:
            for offset in range(0, len(record_indices), clip_batch_size):
                chunk_indices = record_indices[offset : offset + clip_batch_size]
                frames: list[np.ndarray] = []
                frame_ranges: list[tuple[int, int, int, RepetitionRecord]] = []
                for index in chunk_indices:
                    item = records[index]
                    first = len(frames)
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    start_frame, end_frame = video_frame_range(item, frame_count)
                    frames.extend(
                        sample_video_frames(
                            capture,
                            start_frame,
                            end_frame,
                            sample_frames,
                        )
                    )
                    frame_ranges.append((index, first, len(frames), item))

                poses = estimator.estimate_frames(
                    frames,
                    batch_size=inference_batch_size,
                )
                if poses.shape != (len(frames), 17, 3):
                    raise RuntimeError(
                        f"Pose estimator returned invalid shape {poses.shape} "
                        f"for {video_id}."
                    )
                for index, first, last, item in frame_ranges:
                    sequence = poses[first:last]
                    if not np.any(sequence[:, :, 2] >= 0.2):
                        raise ValueError(
                            "No person pose was detected for "
                            f"{item.video_id} repetition {item.repetition_number}."
                        )
                    features[index] = temporal_features(sequence, exercise_id)
        finally:
            capture.release()

    if any(row is None for row in features):
        raise RuntimeError("Pose feature extraction did not produce every repetition.")
    return np.vstack(features)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["incorrect", "correct"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "per_class": report,
    }


def train_and_evaluate(
    data_root: Path | str,
    output_model_path: Path | str,
    report_path: Path | str,
    estimator: PoseEstimator,
    sample_frames: int = 12,
    clip_batch_size: int = 4,
    inference_batch_size: int = 16,
    random_state: int = 42,
) -> dict:
    root = Path(data_root)
    records = load_segmentation(root / "Segmentation.csv")
    dataset_summary = validate_dataset(root, records)
    split = split_by_person(records, random_state=random_state)
    features = _extract_features(
        root,
        records,
        estimator,
        sample_frames=sample_frames,
        clip_batch_size=clip_batch_size,
        inference_batch_size=inference_batch_size,
    )
    labels = np.asarray([record.correctness for record in records], dtype=np.int64)
    exercises = np.asarray([record.exercise_id for record in records], dtype=np.int64)

    train_indices = np.asarray(split.train)
    validation_indices = np.asarray(split.validation)
    test_indices = np.asarray(split.test)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        ),
    )
    model.fit(features[train_indices], labels[train_indices])

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(features[train_indices], labels[train_indices])
    validation_metrics = _classification_metrics(
        labels[validation_indices], model.predict(features[validation_indices])
    )
    test_predictions = model.predict(features[test_indices])
    test_metrics = _classification_metrics(labels[test_indices], test_predictions)
    baseline_metrics = _classification_metrics(
        labels[test_indices], baseline.predict(features[test_indices])
    )

    per_exercise: dict[str, dict] = {}
    for exercise_id in range(1, 7):
        exercise_mask = exercises[test_indices] == exercise_id
        if exercise_mask.any():
            per_exercise[str(exercise_id)] = _classification_metrics(
                labels[test_indices][exercise_mask],
                test_predictions[exercise_mask],
            )

    split_people = {
        "train": sorted({records[index].person_id for index in split.train}),
        "validation": sorted({records[index].person_id for index in split.validation}),
        "test": sorted({records[index].person_id for index in split.test}),
    }
    report = {
        "dataset": "REHAB24-6",
        "feature_version": FEATURE_VERSION,
        "split_strategy": "GroupShuffleSplit by person_id (60/20/20)",
        "participants": split_people,
        "dataset_validation": dataset_summary,
        "sample_frames_per_repetition": sample_frames,
        "baseline_test": baseline_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_by_exercise": per_exercise,
        "limitations": [
            "The binary correctness label does not identify a specific mistake type.",
            "The test set contains unseen participants from a small controlled dataset.",
            "The majority-class baseline is not a clinical or pose-ground-truth baseline.",
        ],
    }

    model_path = Path(output_model_path)
    metrics_path = Path(report_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_version": FEATURE_VERSION,
            "supported_exercises": list(range(1, 7)),
            "sample_frames": sample_frames,
        },
        model_path,
    )
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
