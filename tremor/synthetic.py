"""Synthetic spiral generator with a known, injected tremor frequency.

There is no way to get ground-truth tremor frequency from a real
recording without specialized lab equipment (an accelerometer study, not
a drawing task), so validating tremor/features.py against a real dataset
is not possible. Instead this module generates an ideal Archimedean
spiral trajectory, optionally injects a band-limited radial oscillation
at a chosen frequency and amplitude standing in for tremor, and
tremor/validate.py checks that compute_features recovers the frequency
and amplitude it was given. This is the standard way to validate a signal
processing pipeline when no labeled ground truth exists.
"""

from __future__ import annotations

import numpy as np

from tremor.features import PointerType, StrokePoint


def generate_synthetic_spiral(
    duration_s: float = 6.0,
    sample_rate_hz: float = 120.0,
    turns: float = 4.0,
    max_radius: float = 100.0,
    center: tuple[float, float] = (0.0, 0.0),
    tremor_freq_hz: float | None = None,
    tremor_amplitude: float = 0.0,
    position_noise_std: float = 0.0,
    sampling_jitter_std_ms: float = 0.0,
    mouse_noise: bool = False,
    seed: int = 0,
) -> list[StrokePoint]:
    """Generate a synthetic spiral trace.

    tremor_freq_hz / tremor_amplitude: inject a sinusoidal radial
        oscillation at this frequency (Hz) and amplitude (same units as
        max_radius), standing in for physiological or pathological tremor.
        No oscillation is added when tremor_freq_hz is None.
    position_noise_std: Gaussian noise added to x and y independently, a
        generic drawing-imprecision proxy.
    sampling_jitter_std_ms: Gaussian jitter added to each nominal sample
        time, since real pointer events are not perfectly periodic.
    mouse_noise: when True, simulates coarser mouse input by dropping
        roughly 30% of samples and rounding to integer pixels, and omits
        pressure (mice do not report pressure).
    """
    rng = np.random.default_rng(seed)
    nominal_t_s = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)

    if sampling_jitter_std_ms > 0:
        jitter_s = rng.normal(0, sampling_jitter_std_ms / 1000.0, size=nominal_t_s.shape)
        t_s = np.sort(nominal_t_s + jitter_s)
        t_s = np.clip(t_s, 0.0, None)
    else:
        t_s = nominal_t_s

    theta = 2 * np.pi * turns * (t_s / duration_s)
    radius = max_radius * (t_s / duration_s)

    cx, cy = center
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)

    if tremor_freq_hz is not None and tremor_amplitude > 0:
        # A small circular wobble added directly in Cartesian space, at a
        # fixed frequency independent of the spiral's own angular speed.
        # Modulating the radius in the spiral's rotating frame instead
        # would mix with the spiral's angular velocity and show up as a
        # harmonic rather than the injected frequency itself.
        phase = rng.uniform(0, 2 * np.pi)
        x = x + tremor_amplitude * np.cos(2 * np.pi * tremor_freq_hz * t_s + phase)
        y = y + tremor_amplitude * np.sin(2 * np.pi * tremor_freq_hz * t_s + phase)

    if position_noise_std > 0:
        x = x + rng.normal(0, position_noise_std, size=x.shape)
        y = y + rng.normal(0, position_noise_std, size=y.shape)

    pointer_type: PointerType = "mouse" if mouse_noise else "pen"
    pressures: np.ndarray | None = None
    if not mouse_noise:
        pressures = np.clip(rng.normal(0.5, 0.05, size=x.shape), 0.0, 1.0)

    if mouse_noise:
        keep = rng.random(size=x.shape) > 0.3
        keep[0], keep[-1] = True, True
        t_s, x, y = t_s[keep], np.round(x[keep]), np.round(y[keep])

    t_ms = (t_s - t_s[0]) * 1000.0
    points = [
        StrokePoint(
            x=float(x[i]),
            y=float(y[i]),
            t_ms=float(t_ms[i]),
            pressure=float(pressures[i]) if pressures is not None else None,
            pointer_type=pointer_type,
        )
        for i in range(len(t_s))
    ]
    return points
