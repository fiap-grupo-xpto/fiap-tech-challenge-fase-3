from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.rehab_video.pose import YoloPoseEstimator
from backend.rehab_video.training import train_and_evaluate


DATA_ROOT = PROJECT_ROOT / "entregas" / "fase04" / "data" / "raw" / "rehab24-6"
MODEL_PATH = PROJECT_ROOT / "backend" / "model" / "rehab_pose_classifier.joblib"
REPORT_PATH = (
    PROJECT_ROOT / "entregas" / "fase04" / "results" / "training_report.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the REHAB24-6 repetition classifier."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--weights", default="yolov8n-pose.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--sample-frames", type=int, default=12)
    parser.add_argument("--clip-batch-size", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    estimator = YoloPoseEstimator(
        weights=args.weights,
        image_size=args.image_size,
        device=args.device,
    )
    report = train_and_evaluate(
        data_root=args.data_root,
        output_model_path=args.model_path,
        report_path=args.report_path,
        estimator=estimator,
        sample_frames=args.sample_frames,
        clip_batch_size=args.clip_batch_size,
        inference_batch_size=args.inference_batch_size,
        random_state=args.seed,
    )
    print(f"Test metrics: {report['test']}")
    print(f"Model: {args.model_path}")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
