from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.rehab_video.annotate import annotate_video, draw_skeleton


class FakeEstimator:
    def __init__(self, visible: bool = True) -> None:
        self.visible = visible

    def detect_pixel_keypoints(self, frames, batch_size=16):
        pose = np.zeros((17, 3), dtype=np.float32)
        if self.visible:
            pose[:, 0] = np.linspace(10, 50, 17)
            pose[:, 1] = np.linspace(10, 50, 17)
            pose[:, 2] = 0.9
        return [pose.copy() for _ in frames]


def make_video(path: Path, frames: int = 10, size=(64, 48)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, size)
    for _ in range(frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


def test_draw_skeleton_marks_visible_joints_without_mutating_input():
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    pose = np.zeros((17, 3), dtype=np.float32)
    pose[:, :2] = np.linspace(10, 50, 17)[:, None]
    pose[:, 2] = 0.9

    annotated = draw_skeleton(frame, pose, caption="teste")

    assert annotated.shape == frame.shape
    assert annotated.any()
    assert not frame.any()


def test_draw_skeleton_rejects_invalid_shape():
    with pytest.raises(ValueError, match="shape"):
        draw_skeleton(np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((5, 3)))


def test_annotate_video_writes_selected_frames(tmp_path):
    source = tmp_path / "in.mp4"
    target = tmp_path / "out.mp4"
    make_video(source, frames=10)

    summary = annotate_video(source, target, FakeEstimator(), start_frame=2, end_frame=7)

    assert summary["frames_written"] == 6
    assert summary["frames_with_pose"] == 6
    assert summary["frame_stride"] == 1
    assert target.stat().st_size > 0


def test_annotate_video_uses_stride_to_cap_frames(tmp_path):
    source = tmp_path / "in.mp4"
    make_video(source, frames=10)

    summary = annotate_video(
        source, tmp_path / "out.mp4", FakeEstimator(), max_frames=4
    )

    assert summary["frame_stride"] == 3
    assert summary["frames_written"] == 4


def test_annotate_video_rejects_range_outside_video(tmp_path):
    source = tmp_path / "in.mp4"
    make_video(source, frames=5)

    with pytest.raises(ValueError, match="outside the video"):
        annotate_video(source, tmp_path / "out.mp4", FakeEstimator(), end_frame=20)


def test_annotate_video_requires_a_detected_person(tmp_path):
    source = tmp_path / "in.mp4"
    make_video(source, frames=5)

    with pytest.raises(ValueError, match="No person pose"):
        annotate_video(source, tmp_path / "out.mp4", FakeEstimator(visible=False))


def test_annotate_endpoint_returns_video_and_cleans_up(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_main, "UPLOAD_DIR", tmp_path)

    def fake_annotate(input_path, output_path, exercise_id, start_frame, end_frame):
        Path(output_path).write_bytes(b"annotated-video")
        return {"frames_written": 1}

    monkeypatch.setattr(backend_main, "annotate_repetition_video", fake_annotate)

    with TestClient(backend_main.app) as client:
        response = client.post(
            "/annotate-rehabilitation-video",
            files={"video": ("sample.mp4", b"video-data", "video/mp4")},
            data={"exercise_id": "6"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == b"annotated-video"
    assert list(tmp_path.iterdir()) == []


def test_annotate_endpoint_reports_errors_and_cleans_up(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_main, "UPLOAD_DIR", tmp_path)

    def failing_annotate(*args):
        raise ValueError("No person pose was detected in the selected video segment.")

    monkeypatch.setattr(backend_main, "annotate_repetition_video", failing_annotate)

    with TestClient(backend_main.app) as client:
        response = client.post(
            "/annotate-rehabilitation-video",
            files={"video": ("sample.mp4", b"video-data", "video/mp4")},
            data={"exercise_id": "1"},
        )

    assert response.status_code == 422
    assert list(tmp_path.iterdir()) == []


def test_annotate_endpoint_rejects_non_mp4(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_main, "UPLOAD_DIR", tmp_path)

    with TestClient(backend_main.app) as client:
        response = client.post(
            "/annotate-rehabilitation-video",
            files={"video": ("sample.avi", b"video-data", "video/x-msvideo")},
            data={"exercise_id": "1"},
        )

    assert response.status_code == 415
    assert list(tmp_path.iterdir()) == []
