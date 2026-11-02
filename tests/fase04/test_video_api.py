from pathlib import Path

from fastapi.testclient import TestClient

import backend.main as backend_main


def test_rehabilitation_video_endpoint_returns_analysis(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_main, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(
        backend_main,
        "analyze_repetition",
        lambda path, exercise_id, start_frame, end_frame: {
            "status": "success",
            "exercise_id": exercise_id,
            "classification": "incorrect",
            "correctness": False,
            "start_frame": start_frame,
            "end_frame": end_frame,
        },
    )

    with TestClient(backend_main.app) as client:
        response = client.post(
            "/analyze-rehabilitation-video",
            files={"video": ("sample.mp4", b"video-data", "video/mp4")},
            data={"exercise_id": "6", "start_frame": "20", "end_frame": "90"},
        )

    assert response.status_code == 200
    assert response.json()["classification"] == "incorrect"
    assert response.json()["exercise_id"] == 6
    assert list(tmp_path.iterdir()) == []


def test_rehabilitation_video_endpoint_rejects_non_mp4(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_main, "UPLOAD_DIR", tmp_path)

    with TestClient(backend_main.app) as client:
        response = client.post(
            "/analyze-rehabilitation-video",
            files={"video": ("sample.avi", b"video-data", "video/x-msvideo")},
            data={"exercise_id": "1"},
        )

    assert response.status_code == 415
    assert "mp4" in response.json()["detail"].lower()


def test_rehabilitation_video_endpoint_rejects_oversized_upload(monkeypatch, tmp_path):
    monkeypatch.setattr(backend_main, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(backend_main, "MAX_REHAB_VIDEO_BYTES", 4)

    with TestClient(backend_main.app) as client:
        response = client.post(
            "/analyze-rehabilitation-video",
            files={"video": ("sample.mp4", b"12345", "video/mp4")},
            data={"exercise_id": "1"},
        )

    assert response.status_code == 413
    assert list(tmp_path.iterdir()) == []
