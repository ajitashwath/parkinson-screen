"""Validate tremor/features.py against tremor/synthetic.py's known ground
truth. Produces the recovery and stability plots that justify trusting the
tremor feature pipeline, since it cannot be validated against a real
labeled dataset (see tremor/synthetic.py's module docstring).

Usage:
    python -m tremor.validate --config configs/tremor.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import yaml

from tremor.features import compute_features
from tremor.synthetic import generate_synthetic_spiral

DEFAULT_FREQUENCIES_HZ = [3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
DEFAULT_AMPLITUDES = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]


def frequency_recovery_data(
    frequencies_hz: list[float], amplitude: float = 6.0, seed: int = 0
) -> tuple[list[float], list[float]]:
    recovered = []
    for freq in frequencies_hz:
        points = generate_synthetic_spiral(
            duration_s=6.0,
            sample_rate_hz=120.0,
            tremor_freq_hz=freq,
            tremor_amplitude=amplitude,
            seed=seed,
        )
        result = compute_features(points)
        recovered.append(result.features["peak_freq_hz"])
    return frequencies_hz, recovered


def amplitude_response_data(
    amplitudes: list[float], freq_hz: float = 6.0, seed: int = 0
) -> tuple[list[float], list[float]]:
    ratios = []
    for amplitude in amplitudes:
        points = generate_synthetic_spiral(
            duration_s=6.0,
            sample_rate_hz=120.0,
            tremor_freq_hz=freq_hz,
            tremor_amplitude=amplitude,
            seed=seed,
        )
        result = compute_features(points)
        ratios.append(result.features["tremor_band_ratio"])
    return amplitudes, ratios


def stability_data(
    freq_hz: float = 6.0, amplitude: float = 6.0, seeds: int = 10
) -> dict[str, list[float]]:
    """Feature stability under sampling jitter and mouse-level noise:
    recovered peak_freq_hz across repeated draws with different random
    seeds, for a clean, a jittered, and a mouse-noise condition."""
    conditions: list[tuple[str, float, bool]] = [
        ("clean", 0.0, False),
        ("sampling_jitter", 8.0, False),
        ("mouse_noise", 0.0, True),
    ]
    results: dict[str, list[float]] = {}
    for name, jitter_std_ms, mouse_noise in conditions:
        recovered = []
        for seed in range(seeds):
            points = generate_synthetic_spiral(
                duration_s=6.0,
                sample_rate_hz=120.0 if not mouse_noise else 60.0,
                tremor_freq_hz=freq_hz,
                tremor_amplitude=amplitude,
                sampling_jitter_std_ms=jitter_std_ms,
                mouse_noise=mouse_noise,
                seed=seed,
            )
            result = compute_features(points)
            recovered.append(result.features["peak_freq_hz"])
        results[name] = recovered
    return results


def plot_frequency_recovery(frequencies_hz: list[float], recovered: list[float]) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frequencies_hz,
            y=frequencies_hz,
            mode="lines",
            name="perfect recovery",
            line={"dash": "dash", "color": "gray"},
        )
    )
    figure.add_trace(
        go.Scatter(x=frequencies_hz, y=recovered, mode="markers+lines", name="recovered")
    )
    figure.update_layout(
        title="Injected vs recovered tremor frequency",
        xaxis_title="Injected frequency (Hz)",
        yaxis_title="Recovered peak_freq_hz (Hz)",
    )
    return figure


def plot_amplitude_response(amplitudes: list[float], ratios: list[float]) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=amplitudes, y=ratios, mode="markers+lines"))
    figure.update_layout(
        title="Tremor amplitude vs tremor_band_ratio",
        xaxis_title="Injected tremor amplitude (px)",
        yaxis_title="tremor_band_ratio",
    )
    return figure


def plot_stability(results: dict[str, list[float]], injected_freq_hz: float) -> go.Figure:
    figure = go.Figure()
    for name, values in results.items():
        figure.add_trace(go.Box(y=values, name=name, boxpoints="all"))
    figure.add_hline(y=injected_freq_hz, line_dash="dash", annotation_text="injected frequency")
    figure.update_layout(
        title="Recovered frequency stability under jitter and mouse noise",
        yaxis_title="Recovered peak_freq_hz (Hz)",
    )
    return figure


def run_validation(reports_dir: Path, seed: int = 0) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    frequencies_hz, recovered = frequency_recovery_data(DEFAULT_FREQUENCIES_HZ, seed=seed)
    figure = plot_frequency_recovery(frequencies_hz, recovered)
    out_path = reports_dir / "tremor_frequency_recovery.html"
    figure.write_html(out_path, include_plotlyjs="cdn")
    written["frequency_recovery"] = out_path

    amplitudes, ratios = amplitude_response_data(DEFAULT_AMPLITUDES, seed=seed)
    figure = plot_amplitude_response(amplitudes, ratios)
    out_path = reports_dir / "tremor_amplitude_response.html"
    figure.write_html(out_path, include_plotlyjs="cdn")
    written["amplitude_response"] = out_path

    stability = stability_data(seeds=10)
    figure = plot_stability(stability, injected_freq_hz=6.0)
    out_path = reports_dir / "tremor_stability.html"
    figure.write_html(out_path, include_plotlyjs="cdn")
    written["stability"] = out_path

    max_error = max(abs(r - f) for f, r in zip(frequencies_hz, recovered, strict=True))
    print(f"Frequency recovery: max error {max_error:.3f} Hz across {frequencies_hz} Hz")
    print(f"Amplitude response: tremor_band_ratio {ratios[0]:.3f} to {ratios[-1]:.3f}")
    for name, values in stability.items():
        arr = np.array(values)
        print(f"Stability [{name}]: mean {arr.mean():.2f} Hz, std {arr.std():.2f} Hz")
    for key, path in written.items():
        print(f"Wrote {key} to {path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tremor.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fast", action="store_true", help="unused, kept for CLI consistency")
    args = parser.parse_args()

    config: dict[str, Any] = {}
    if Path(args.config).exists():
        with open(args.config, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    reports_dir = Path(str(config.get("reports_dir", "reports")))

    run_validation(reports_dir, seed=seed)


if __name__ == "__main__":
    main()
