from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


REQUIRED_COLUMNS = {
    "video_id",
    "repetition_number",
    "exercise_id",
    "person_id",
    "first_frame",
    "last_frame",
    "correctness",
}


@dataclass(frozen=True)
class RepetitionRecord:
    video_id: str
    repetition_number: int
    exercise_id: int
    person_id: int
    first_frame: int
    last_frame: int
    correctness: int


@dataclass(frozen=True)
class PersonSplit:
    train: list[int]
    validation: list[int]
    test: list[int]


def load_segmentation(path: Path | str) -> list[RepetitionRecord]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Segmentation file not found: {path}")

    with path.open(encoding="utf-8-sig", newline="") as source:
        first_line = source.readline()
        if not first_line:
            raise ValueError(f"Segmentation file is empty: {path}")
        delimiter = ";" if first_line.count(";") >= first_line.count(",") else ","
        source.seek(0)
        reader = csv.DictReader(source, delimiter=delimiter)
        headers = {header.strip() for header in (reader.fieldnames or [])}
        missing = REQUIRED_COLUMNS - headers
        if missing:
            raise ValueError(f"Segmentation file is missing columns: {sorted(missing)}")

        records: list[RepetitionRecord] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                record = RepetitionRecord(
                    video_id=row["video_id"].strip(),
                    repetition_number=int(row["repetition_number"]),
                    exercise_id=int(row["exercise_id"]),
                    person_id=int(row["person_id"]),
                    first_frame=int(row["first_frame"]),
                    last_frame=int(row["last_frame"]),
                    correctness=int(row["correctness"]),
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(
                    f"Invalid segmentation values on line {line_number}: {exc}"
                ) from exc

            if not record.video_id:
                raise ValueError(f"Missing video_id on line {line_number}.")
            if record.exercise_id not in range(1, 7):
                raise ValueError(f"Unsupported exercise_id on line {line_number}.")
            if record.person_id < 1:
                raise ValueError(f"Invalid person_id on line {line_number}.")
            if record.first_frame < 0 or record.last_frame < record.first_frame:
                raise ValueError(f"Invalid frame range on line {line_number}.")
            if record.correctness not in (0, 1):
                raise ValueError(f"Invalid correctness label on line {line_number}.")
            records.append(record)

    if not records:
        raise ValueError(f"Segmentation file contains no repetitions: {path}")
    return records


def video_path_for(data_root: Path | str, record: RepetitionRecord) -> Path:
    return (
        Path(data_root)
        / f"Ex{record.exercise_id}"
        / f"{record.video_id}-Camera17-30fps.mp4"
    )


def video_frame_range(
    record: RepetitionRecord, frame_count: int | None = None
) -> tuple[int, int]:
    """Convert the dataset's 1-based inclusive frame numbers to 0-based indexes."""
    if record.first_frame < 1:
        raise ValueError("REHAB24-6 annotation frame numbers must start at 1.")
    start_frame = record.first_frame - 1
    end_frame = record.last_frame - 1
    if frame_count is not None:
        if start_frame >= frame_count or end_frame > frame_count:
            raise ValueError(
                f"Annotated frame range {record.first_frame}-{record.last_frame} "
                f"exceeds a video with {frame_count} frames."
            )
        if end_frame == frame_count:
            end_frame = frame_count - 1
    return start_frame, end_frame


def validate_dataset(
    data_root: Path | str, records: Sequence[RepetitionRecord]
) -> dict[str, int | float]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to validate the video files.") from exc

    grouped: dict[tuple[int, str], list[RepetitionRecord]] = {}
    for record in records:
        grouped.setdefault((record.exercise_id, record.video_id), []).append(record)

    errors: list[str] = []
    checked_frames = 0
    checked_videos = 0
    adjusted_repetitions = 0
    for video_records in grouped.values():
        record = video_records[0]
        path = video_path_for(data_root, record)
        if not path.is_file():
            errors.append(f"Missing video: {path}")
            continue

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            errors.append(f"Video cannot be decoded: {path}")
            continue

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        checked_videos += 1
        checked_frames += frame_count
        if abs(fps - 30.0) > 0.5:
            errors.append(f"Expected 30 FPS, found {fps:.3f}: {path}")
        for repetition in video_records:
            try:
                start_frame, end_frame = video_frame_range(repetition, frame_count)
                if end_frame < repetition.last_frame - 1:
                    adjusted_repetitions += 1
            except ValueError as exc:
                errors.append(
                    f"{repetition.video_id} repetition "
                    f"{repetition.repetition_number}: {exc}"
                )

    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors[:25]))
    return {
        "repetitions": len(records),
        "videos": checked_videos,
        "frames": checked_frames,
        "fps": 30,
        "boundary_adjusted_repetitions": adjusted_repetitions,
    }


def split_by_person(
    records: Sequence[RepetitionRecord], random_state: int = 42
) -> PersonSplit:
    if len(records) < 3:
        raise ValueError("At least three repetitions are required for grouped splits.")
    if len({record.person_id for record in records}) < 5:
        raise ValueError("At least five participants are required for grouped splits.")

    indices = np.arange(len(records))
    groups = np.asarray([record.person_id for record in records])
    first_split = GroupShuffleSplit(
        n_splits=1, test_size=0.4, random_state=random_state
    )
    train_indices, remainder_indices = next(
        first_split.split(indices, groups=groups)
    )
    remainder_groups = groups[remainder_indices]
    second_split = GroupShuffleSplit(
        n_splits=1, test_size=0.5, random_state=random_state + 1
    )
    validation_positions, test_positions = next(
        second_split.split(remainder_indices, groups=remainder_groups)
    )
    validation_indices = remainder_indices[validation_positions]
    test_indices = remainder_indices[test_positions]

    for name, split_indices in (
        ("train", train_indices),
        ("validation", validation_indices),
        ("test", test_indices),
    ):
        labels = {records[index].correctness for index in split_indices}
        if labels != {0, 1}:
            raise ValueError(
                f"The {name} participant split must contain both correctness labels."
            )

    return PersonSplit(
        train=train_indices.tolist(),
        validation=validation_indices.tolist(),
        test=test_indices.tolist(),
    )
