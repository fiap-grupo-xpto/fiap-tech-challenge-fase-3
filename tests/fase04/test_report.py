import numpy as np
import pytest

from backend.rehab_video.report import build_deviation_report


def squat_sequence(left_bend: float, right_bend: float, frames: int = 8) -> np.ndarray:
    """Agachamento sintetico: o joelho dobra ate `bend` graus em cada lado."""
    poses = np.zeros((frames, 17, 3), dtype=np.float32)
    poses[:, :, 2] = 0.9
    poses[:, 5] = poses[:, 6] = (0.0, -0.4, 0.9)
    for hip, knee, ankle, bend in ((11, 13, 15, left_bend), (12, 14, 16, right_bend)):
        x = -0.1 if hip == 11 else 0.1
        poses[:, hip, :2] = (x, 0.0)
        poses[:, knee, :2] = (x, 0.3)
        for index, fraction in enumerate(np.linspace(0, 1, frames)):
            angle = np.radians(bend * fraction)
            poses[index, ankle, :2] = (x + 0.3 * np.sin(angle), 0.3 + 0.3 * np.cos(angle))
    return poses


def test_good_squat_has_no_deviation():
    report = build_deviation_report(squat_sequence(80, 80), exercise_id=6)

    assert report["deviation_count"] == 0
    assert "Nenhum desvio" in report["summary"]


def test_reduced_amplitude_is_reported():
    report = build_deviation_report(squat_sequence(10, 10), exercise_id=6)

    codes = {item["code"] for item in report["findings"]}
    assert "amplitude_reduzida" in codes
    assert "abaixo do esperado" in report["summary"]


def test_asymmetry_is_reported_with_weaker_side():
    report = build_deviation_report(squat_sequence(90, 40), exercise_id=6)

    item = next(f for f in report["findings"] if f["code"] == "assimetria")
    assert "lado direito" in item["message"]


def test_low_confidence_returns_insufficient_data():
    poses = squat_sequence(80, 80)
    poses[:, :, 2] = 0.05

    report = build_deviation_report(poses, exercise_id=6)

    assert report["findings"][0]["code"] == "dados_insuficientes"
    assert report["deviation_count"] == 0


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="shape"):
        build_deviation_report(np.zeros((4, 5, 3)), exercise_id=6)
    with pytest.raises(ValueError, match="Unsupported"):
        build_deviation_report(squat_sequence(80, 80), exercise_id=9)
