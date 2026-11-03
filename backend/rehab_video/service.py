from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from backend.rehab_video.annotate import (
    DEFAULT_MAX_FRAMES,
    annotate_video,
    transcode_to_h264,
)
from backend.rehab_video.features import FEATURE_VERSION, temporal_features
from backend.rehab_video.pose import YoloPoseEstimator, extract_pose_sequence
from backend.rehab_video.report import build_deviation_report


EXERCISE_LABELS = {
    1: "Abducao de braco",
    2: "Bracos V/W",
    3: "Flexao apoiada",
    4: "Abducao de perna",
    5: "Afundo",
    6: "Agachamento",
}


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "rehab_pose_classifier.joblib"
)


@lru_cache(maxsize=1)
def _get_pose_estimator() -> YoloPoseEstimator:
    return YoloPoseEstimator(
        weights=os.getenv("REHAB_POSE_WEIGHTS", "yolov8n-pose.pt"),
        image_size=int(os.getenv("REHAB_POSE_IMAGE_SIZE", "384")),
        confidence_threshold=float(os.getenv("REHAB_POSE_CONFIDENCE", "0.25")),
        device=os.getenv("REHAB_POSE_DEVICE", "cpu"),
    )


def annotate_repetition_video(
    video_path: Path | str,
    output_path: Path | str,
    exercise_id: int,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict:
    if exercise_id not in range(1, 7):
        raise ValueError("exercise_id must be between 1 and 6.")
    estimator = _get_pose_estimator()
    report = None
    try:
        report = build_deviation_report(
            extract_pose_sequence(
                video_path,
                start_frame=start_frame,
                end_frame=end_frame,
                estimator=estimator,
                sample_count=12,
            ),
            exercise_id,
        )
    except ValueError:
        pass
    label = EXERCISE_LABELS[exercise_id]
    if report is not None:
        label = f"{label} | desvios: {report['deviation_count']}"
    summary = annotate_video(
        video_path,
        output_path,
        estimator,
        start_frame=start_frame,
        end_frame=end_frame,
        max_frames=int(os.getenv("REHAB_ANNOTATE_MAX_FRAMES", str(DEFAULT_MAX_FRAMES))),
        confidence_threshold=estimator.confidence_threshold,
        label=label,
    )
    summary["deviation_report"] = report
    summary["browser_compatible"] = transcode_to_h264(output_path)
    return summary


def analyze_repetition(
    video_path: Path | str,
    exercise_id: int,
    start_frame: int = 0,
    end_frame: int | None = None,
    model_path: Path | str = MODEL_PATH,
) -> dict:
    if exercise_id not in range(1, 7):
        raise ValueError("exercise_id must be between 1 and 6.")
    artifact_path = Path(model_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            "Rehabilitation classifier is not trained. Run "
            "`py -3.11 entregas\\fase04\\src\\train_classifier.py` first."
        )

    artifact = joblib.load(artifact_path)
    if artifact.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError(
            "The rehabilitation model uses an incompatible feature definition. "
            "Retrain it with the current project version."
        )
    sample_count = int(artifact.get("sample_frames", 12))
    poses = extract_pose_sequence(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        estimator=_get_pose_estimator(),
        sample_count=sample_count,
    )
    features = temporal_features(poses, exercise_id).reshape(1, -1)
    deviation_report = build_deviation_report(poses, exercise_id)
    model = artifact.get("model")
    if model is None:
        raise RuntimeError("The rehabilitation model artifact is missing its model.")

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    class_ids = list(model.classes_)
    probability_correct = float(probabilities[class_ids.index(1)])
    return {
        "status": "success",
        "exercise_id": exercise_id,
        "classification": "correct" if prediction == 1 else "incorrect",
        "correctness": bool(prediction),
        "probability_correct": probability_correct,
        "confidence": float(np.max(probabilities)),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "sampled_frames": sample_count,
        "deviation_report": deviation_report,
        "requires_physiotherapist_review": True,
        "warning": (
            "Experimental result based on a small, controlled rehabilitation "
            "dataset; it is not a clinical assessment."
        ),
    }
