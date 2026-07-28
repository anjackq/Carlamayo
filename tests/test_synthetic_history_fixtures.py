import numpy as np
import pytest

from scripts.build_synthetic_history_fixtures import synthetic_history_xyz


def test_stationary_history_is_zero():
    history = synthetic_history_xyz("stationary", 0.0)

    assert history.shape == (16, 3)
    assert history == pytest.approx(np.zeros((16, 3)))


def test_steady_history_ends_at_origin_with_constant_spacing():
    history = synthetic_history_xyz("steady", 2.0)

    assert history[-1] == pytest.approx([0.0, 0.0, 0.0])
    assert history[0, 0] == pytest.approx(-3.0)
    assert np.diff(history[:, 0]) == pytest.approx(np.full(15, 0.2))


def test_launch_history_accelerates_toward_capture_time():
    history = synthetic_history_xyz("launch", 2.0)
    steps = np.diff(history[:, 0])

    assert history[-1, 0] == pytest.approx(0.0)
    assert np.all(np.diff(steps) > 0.0)
    assert steps[-1] > steps[0]


@pytest.mark.parametrize("kind", ["bad", "", "moving"])
def test_synthetic_history_rejects_unknown_kind(kind):
    with pytest.raises(ValueError):
        synthetic_history_xyz(kind, 1.0)
