from __future__ import annotations

import numpy as np
import pytest

from tremor.features import StrokePoint, compute_features, validate_points
from tremor.synthetic import generate_synthetic_spiral


@pytest.mark.parametrize("injected_freq_hz", [3.0, 5.0, 6.5, 8.0, 10.0, 12.0])
def test_recovered_frequency_within_half_hz_of_injected(injected_freq_hz: float) -> None:
    points = generate_synthetic_spiral(
        duration_s=6.0,
        sample_rate_hz=120.0,
        tremor_freq_hz=injected_freq_hz,
        tremor_amplitude=6.0,
        seed=1,
    )
    result = compute_features(points)
    recovered = result.features["peak_freq_hz"]
    assert not np.isnan(recovered)
    assert abs(recovered - injected_freq_hz) <= 0.5


def test_tremor_band_ratio_increases_with_amplitude() -> None:
    ratios = []
    for amplitude in (0.0, 2.0, 5.0, 10.0):
        points = generate_synthetic_spiral(
            duration_s=6.0,
            sample_rate_hz=120.0,
            tremor_freq_hz=6.0,
            tremor_amplitude=amplitude,
            seed=2,
        )
        result = compute_features(points)
        ratios.append(result.features["tremor_band_ratio"])
    assert ratios == sorted(ratios)
    assert ratios[-1] > ratios[0]


def test_no_tremor_gives_low_band_ratio() -> None:
    points = generate_synthetic_spiral(duration_s=6.0, sample_rate_hz=120.0, seed=3)
    result = compute_features(points)
    assert result.features["tremor_band_ratio"] < 0.3


def test_short_input_returns_warning_not_exception() -> None:
    points = [StrokePoint(x=0, y=0, t_ms=0), StrokePoint(x=1, y=1, t_ms=100)]
    result = compute_features(points)
    assert result.warnings
    assert any("duration too short" in w or "too few points" in w for w in result.warnings)


def test_empty_input_returns_warning_not_exception() -> None:
    result = compute_features([])
    assert result.features == {}
    assert result.warnings


def test_sparse_input_warns_about_sampling_rate() -> None:
    points = [StrokePoint(x=float(i), y=float(i), t_ms=float(i * 500)) for i in range(10)]
    warnings = validate_points(points)
    assert any("sampling rate too low" in w for w in warnings)


def test_mouse_input_warns() -> None:
    points = generate_synthetic_spiral(
        duration_s=3.0, sample_rate_hz=60.0, mouse_noise=True, seed=4
    )
    warnings = validate_points(points)
    assert any("mouse input" in w for w in warnings)


def test_pen_input_does_not_warn_about_mouse() -> None:
    points = generate_synthetic_spiral(
        duration_s=3.0, sample_rate_hz=60.0, mouse_noise=False, seed=4
    )
    warnings = validate_points(points)
    assert not any("mouse input" in w for w in warnings)


def test_features_are_deterministic() -> None:
    points = generate_synthetic_spiral(
        duration_s=4.0, tremor_freq_hz=7.0, tremor_amplitude=4.0, seed=5
    )
    first = compute_features(points)
    second = compute_features(points)
    assert first.features == second.features


def test_pressure_features_present_for_pen_input() -> None:
    points = generate_synthetic_spiral(duration_s=3.0, mouse_noise=False, seed=6)
    result = compute_features(points)
    assert "pressure_mean" in result.features
    assert "pressure_sd" in result.features
    assert 0.0 <= result.features["pressure_mean"] <= 1.0


def test_pressure_features_absent_for_mouse_input() -> None:
    points = generate_synthetic_spiral(duration_s=3.0, mouse_noise=True, seed=6)
    result = compute_features(points)
    assert "pressure_mean" not in result.features


def test_spiral_deviation_low_for_ideal_spiral() -> None:
    # A perfectly generated Archimedean spiral still has a small, stable
    # residual here from resampling and discrete-sample fitting error, not
    # from any imperfection in the drawing itself.
    points = generate_synthetic_spiral(duration_s=5.0, sample_rate_hz=100.0, seed=7)
    result = compute_features(points)
    assert result.features["spiral_deviation"] < 0.08


def test_spiral_deviation_higher_with_heavy_position_noise() -> None:
    clean = generate_synthetic_spiral(duration_s=5.0, sample_rate_hz=100.0, seed=8)
    noisy = generate_synthetic_spiral(
        duration_s=5.0, sample_rate_hz=100.0, position_noise_std=8.0, seed=8
    )
    clean_result = compute_features(clean)
    noisy_result = compute_features(noisy)
    assert noisy_result.features["spiral_deviation"] > clean_result.features["spiral_deviation"]


def test_path_length_positive_for_a_real_spiral() -> None:
    points = generate_synthetic_spiral(duration_s=4.0, seed=9)
    result = compute_features(points)
    assert result.features["path_length"] > 0


def test_duration_matches_point_span() -> None:
    points = generate_synthetic_spiral(duration_s=4.0, sample_rate_hz=100.0, seed=10)
    result = compute_features(points)
    assert abs(result.features["duration_s"] - 4.0) < 0.1
