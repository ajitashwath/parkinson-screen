from __future__ import annotations

from pathlib import Path

from tremor.validate import (
    amplitude_response_data,
    frequency_recovery_data,
    run_validation,
    stability_data,
)


def test_frequency_recovery_data_matches_input_length() -> None:
    frequencies = [4.0, 8.0, 12.0]
    freqs, recovered = frequency_recovery_data(frequencies, amplitude=6.0, seed=0)
    assert freqs == frequencies
    assert len(recovered) == len(frequencies)


def test_frequency_recovery_data_is_accurate() -> None:
    frequencies = [4.0, 8.0, 12.0]
    freqs, recovered = frequency_recovery_data(frequencies, amplitude=6.0, seed=0)
    for injected, measured in zip(freqs, recovered, strict=True):
        assert abs(injected - measured) <= 0.5


def test_amplitude_response_data_increases() -> None:
    amplitudes = [0.0, 3.0, 8.0]
    amps, ratios = amplitude_response_data(amplitudes, freq_hz=6.0, seed=0)
    assert amps == amplitudes
    assert ratios == sorted(ratios)
    assert ratios[-1] > ratios[0]


def test_stability_data_has_three_conditions_with_expected_seed_count() -> None:
    results = stability_data(seeds=3)
    assert set(results.keys()) == {"clean", "sampling_jitter", "mouse_noise"}
    for values in results.values():
        assert len(values) == 3


def test_run_validation_writes_three_html_files(tmp_path: Path) -> None:
    written = run_validation(tmp_path, seed=0)
    assert set(written.keys()) == {"frequency_recovery", "amplitude_response", "stability"}
    for path in written.values():
        assert path.exists()
        assert path.stat().st_size > 1000
