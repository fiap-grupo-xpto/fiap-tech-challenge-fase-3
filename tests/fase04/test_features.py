import numpy as np

from backend.rehab_video.features import normalize_pose, temporal_features


def test_normalize_pose_centers_and_scales_visible_skeleton():
    points = np.zeros((17, 3), dtype=np.float32)
    points[:, 0] = np.linspace(100, 300, 17)
    points[:, 1] = np.linspace(80, 280, 17)
    points[:, 2] = 0.9

    normalized = normalize_pose(points, width=400, height=400)

    assert normalized.shape == (17, 3)
    assert np.isfinite(normalized).all()
    assert normalized[:, 0].min() >= -0.6
    assert normalized[:, 0].max() <= 0.6
    assert normalized[:, 1].min() >= -0.6
    assert normalized[:, 1].max() <= 0.6
    assert np.allclose(normalized[:, 2], 0.9)


def test_normalize_pose_returns_zeroes_when_no_keypoints_are_visible():
    normalized = normalize_pose(np.zeros((17, 3), dtype=np.float32), 320, 240)

    assert np.array_equal(normalized, np.zeros((17, 3), dtype=np.float32))


def test_temporal_features_have_fixed_shape_and_finite_values():
    sequence = np.zeros((12, 17, 3), dtype=np.float32)
    sequence[:, :, 0] = np.linspace(-0.2, 0.2, 12)[:, None]
    sequence[:, :, 1] = 0.1
    sequence[:, :, 2] = 0.9

    features = temporal_features(sequence, exercise_id=3)

    assert features.shape == (17 * 9 + 6,)
    assert np.isfinite(features).all()
    assert features[-6:].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
