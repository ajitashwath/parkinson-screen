"""Turn a spiral or wave drawing trace into tremor features.

Input is a sequence of StrokePoint (x, y, t_ms, pressure, pointer_type),
plus optional pen-lift markers. This module never raises on messy input
(too short, too sparse, mouse noise): validate_points returns warnings and
compute_features keeps going, since a live canvas trace from a real user
is exactly the kind of input that will sometimes be short or noisy.

Feature glossary (plain names):
    duration_s          total time span of the stroke, in seconds.
    path_length         total on-screen distance traveled.
    mean_speed          average speed, in pixels per second.
    speed_cv            coefficient of variation of speed (std / mean):
                         how uneven the drawing speed was.
    jerk_rms            root-mean-square jerk (rate of change of
                         acceleration), normalized by mean speed so it is
                         comparable across strokes of different size and
                         duration.
    pen_lifts           number of times the pen left the surface, from
                         explicit pen_lift markers or a large time gap
                         between consecutive points.
    pause_ratio         fraction of the stroke's duration spent nearly
                         motionless.
    tremor_band_ratio   fraction of the speed signal's power that falls in
                         the 4 to 12 Hz band, from a Welch power spectrum.
                         Physiological tremor concentrates in this band.
    peak_freq_hz        the single dominant frequency within 3 to 12 Hz.
    spectral_entropy    Shannon entropy of the normalized power spectrum,
                         in [0, 1]; higher means power is spread across
                         more frequencies rather than concentrated.
    spiral_deviation    fits an Archimedean spiral (r = a + b*theta) to the
                         trace and reports the residual RMS divided by the
                         mean radius: how far the drawing strays from a
                         perfect spiral, scale-independent.
    pressure_mean/sd    mean and standard deviation of stylus pressure,
                         only computed when the input is pen input with
                         pressure values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, welch

PointerType = Literal["mouse", "pen", "touch"]

MIN_DURATION_S = 2.0
MIN_POINTS = 50
MIN_SAMPLING_RATE_HZ = 20.0
RESAMPLE_RATE_HZ = 100.0
TREMOR_BAND_HZ = (4.0, 12.0)
PEAK_SEARCH_BAND_HZ = (3.0, 12.0)
PEN_LIFT_GAP_MS = 150.0
PAUSE_SPEED_THRESHOLD_FRAC = 0.02
VOLUNTARY_MOVEMENT_CUTOFF_HZ = 2.0


@dataclass(frozen=True)
class StrokePoint:
    x: float
    y: float
    t_ms: float
    pressure: float | None = None
    pointer_type: PointerType = "mouse"
    pen_lift: bool = False


@dataclass(frozen=True)
class FeatureResult:
    features: dict[str, float]
    warnings: list[str]


def validate_points(
    points: list[StrokePoint],
    min_duration_s: float = MIN_DURATION_S,
    min_points: int = MIN_POINTS,
    min_sampling_rate_hz: float = MIN_SAMPLING_RATE_HZ,
) -> list[str]:
    """Check input quality and return warnings; never raises. A short,
    sparse or mouse-driven stroke is still valid input, just less
    reliable, and the caller (the backend, the UI) decides what to do
    with the warnings."""
    warnings: list[str] = []
    if len(points) < min_points:
        warnings.append(f"too few points: {len(points)} (minimum {min_points})")
    if len(points) >= 2:
        duration_s = (points[-1].t_ms - points[0].t_ms) / 1000.0
        if duration_s < min_duration_s:
            warnings.append(f"duration too short: {duration_s:.2f}s (minimum {min_duration_s}s)")
        if duration_s > 0:
            effective_rate = len(points) / duration_s
            if effective_rate < min_sampling_rate_hz:
                warnings.append(
                    f"sampling rate too low: {effective_rate:.1f} Hz "
                    f"(minimum {min_sampling_rate_hz} Hz); tremor frequency "
                    "estimates above half this rate are not reliable"
                )
    else:
        warnings.append("fewer than 2 points: cannot compute duration or any feature")
    if points and all(p.pointer_type == "mouse" for p in points):
        warnings.append(
            "mouse input: lacks the sub-pixel precision of touch or stylus input, "
            "which adds noise to tremor features"
        )
    return warnings


def resample_uniform(
    points: list[StrokePoint], rate_hz: float = RESAMPLE_RATE_HZ
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Interpolate x, y (and pressure, if present) onto a uniform time
    grid at rate_hz. Returns (t_s, x, y, pressure_or_none)."""
    t_ms = np.array([p.t_ms for p in points], dtype=np.float64)
    x = np.array([p.x for p in points], dtype=np.float64)
    y = np.array([p.y for p in points], dtype=np.float64)
    order = np.argsort(t_ms)
    t_ms, x, y = t_ms[order], x[order], y[order]
    t_ms, unique_index = np.unique(t_ms, return_index=True)
    x, y = x[unique_index], y[unique_index]

    t_s = t_ms / 1000.0
    step = 1.0 / rate_hz
    grid = np.arange(t_s[0], t_s[-1], step)
    if len(grid) < 2:
        grid = np.array([t_s[0], t_s[-1]])
    x_grid = np.interp(grid, t_s, x)
    y_grid = np.interp(grid, t_s, y)

    pressures = [p.pressure for p in points]
    pressure_grid = None
    if all(pr is not None for pr in pressures):
        pressure_arr = np.array(pressures, dtype=np.float64)[order][unique_index]
        pressure_grid = np.interp(grid, t_s, pressure_arr)
    return grid, x_grid, y_grid, pressure_grid


def smooth(signal: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing with a window adapted to the signal
    length, since a short stroke cannot support a long filter window."""
    if len(signal) < 5:
        return signal
    window = min(11, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
    window = max(5, window)
    polyorder = min(3, window - 2)
    return savgol_filter(signal, window_length=window, polyorder=polyorder)


def compute_kinematics(
    t_s: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Speed, acceleration, jerk, and the underlying vx, vy velocity
    components, from smoothed x, y at a uniform sample rate. speed,
    acceleration and jerk are shorter than the input by 0, 1 and 2 samples
    respectively (from successive np.diff calls); vx, vy match speed's
    length."""
    if len(t_s) < 2:
        empty = np.array([])
        return empty, empty, empty, empty, empty
    dt = t_s[1] - t_s[0]
    x_smooth, y_smooth = smooth(x), smooth(y)
    vx, vy = np.diff(x_smooth) / dt, np.diff(y_smooth) / dt
    speed = np.sqrt(vx**2 + vy**2)
    acceleration = np.diff(speed) / dt if len(speed) >= 2 else np.array([])
    jerk = np.diff(acceleration) / dt if len(acceleration) >= 2 else np.array([])
    return speed, acceleration, jerk, vx, vy


def remove_voluntary_movement(signal: np.ndarray, rate_hz: float) -> np.ndarray:
    """High-pass filter a velocity component to remove the slow,
    large-amplitude "voluntary movement" component of drawing a spiral
    (the stroke's own changing velocity as it grows outward). Standard
    practice in tremor analysis; zero-phase (filtfilt) so it does not
    shift the signal in time."""
    nyquist = rate_hz / 2
    normalized_cutoff = VOLUNTARY_MOVEMENT_CUTOFF_HZ / nyquist
    b, a = butter(4, normalized_cutoff, btype="highpass")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(signal) <= padlen:
        return signal
    return np.asarray(filtfilt(b, a, signal))


def spectral_features(vx: np.ndarray, vy: np.ndarray, rate_hz: float) -> tuple[float, float, float]:
    """tremor_band_ratio, peak_freq_hz and spectral_entropy from the
    combined power spectrum of the vx and vy velocity components (the
    sum of their individual Welch spectra), after removing the voluntary
    movement component from each (see remove_voluntary_movement).

    This is deliberately not a Welch spectrum of the speed (magnitude)
    signal directly: speed is a nonlinear function (sqrt(vx^2 + vy^2)) of
    a rotating velocity vector, and on a spiral, the baseline drawing
    motion itself is a second, slower-rotating vector. Combining two
    rotating vectors through a magnitude produces intermodulation
    sidebands at |tremor_freq +/- baseline_rotation_freq| rather than a
    clean peak at tremor_freq, which this was verified empirically to
    reproduce. Summing the linear vx and vy power spectra avoids that
    nonlinearity and recovers the injected frequency cleanly; see the
    synthetic recovery tests and docs/ASSUMPTIONS.md."""
    if len(vx) < 8:
        return float("nan"), float("nan"), float("nan")
    vx_filtered = remove_voluntary_movement(vx, rate_hz)
    vy_filtered = remove_voluntary_movement(vy, rate_hz)
    nperseg = min(256, len(vx_filtered))
    freqs, power_x = welch(vx_filtered, fs=rate_hz, nperseg=nperseg)
    _, power_y = welch(vy_filtered, fs=rate_hz, nperseg=nperseg)
    power = power_x + power_y
    total_power = power.sum()
    if total_power <= 0:
        return float("nan"), float("nan"), float("nan")

    band_mask = (freqs >= TREMOR_BAND_HZ[0]) & (freqs <= TREMOR_BAND_HZ[1])
    tremor_band_ratio = float(power[band_mask].sum() / total_power)

    peak_mask = (freqs >= PEAK_SEARCH_BAND_HZ[0]) & (freqs <= PEAK_SEARCH_BAND_HZ[1])
    if peak_mask.any():
        peak_freq_hz = float(freqs[peak_mask][np.argmax(power[peak_mask])])
    else:
        peak_freq_hz = float("nan")

    nonzero = power[power > 0]
    probabilities = nonzero / nonzero.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    max_entropy = np.log2(len(power)) if len(power) > 1 else 1.0
    spectral_entropy = entropy / max_entropy if max_entropy > 0 else float("nan")

    return tremor_band_ratio, peak_freq_hz, spectral_entropy


def spiral_deviation_feature(x: np.ndarray, y: np.ndarray) -> float:
    """Fit an Archimedean spiral r = a + b*theta to the trace and return
    residual RMS divided by mean radius: how far the drawing strays from
    a perfect spiral, independent of the drawing's scale. Centered on the
    trace's own centroid, since the true template center is not always
    known by the caller. Points near the center are excluded from the fit:
    the angle atan2(dy, dx) is numerically undefined at radius 0, so
    including them inflates the residual even for a perfect spiral."""
    if len(x) < 8:
        return float("nan")
    cx, cy = x.mean(), y.mean()
    dx, dy = x - cx, y - cy
    radius = np.sqrt(dx**2 + dy**2)
    away_from_center = radius > 0.05 * radius.max()
    if away_from_center.sum() < 8:
        return float("nan")
    dx, dy, radius = dx[away_from_center], dy[away_from_center], radius[away_from_center]
    theta = np.unwrap(np.arctan2(dy, dx))
    if np.allclose(theta, theta[0]):
        return float("nan")
    design = np.column_stack([np.ones_like(theta), theta])
    coefficients, *_ = np.linalg.lstsq(design, radius, rcond=None)
    predicted = design @ coefficients
    residual = radius - predicted
    mean_radius = radius.mean()
    if mean_radius <= 0:
        return float("nan")
    return float(np.sqrt(np.mean(residual**2)) / mean_radius)


def count_pen_lifts(points: list[StrokePoint]) -> int:
    lifts = sum(1 for p in points if p.pen_lift)
    for previous, current in zip(points, points[1:], strict=False):
        if current.t_ms - previous.t_ms > PEN_LIFT_GAP_MS:
            lifts += 1
    return lifts


def compute_features(points: list[StrokePoint]) -> FeatureResult:
    """Compute the full tremor feature set for one stroke. Deterministic:
    the same points always produce the same features, since every step is
    a fixed function with no randomness."""
    warnings = validate_points(points)
    if len(points) < 2:
        return FeatureResult(features={}, warnings=warnings)

    t_s, x, y, pressure = resample_uniform(points)
    duration_s = float(t_s[-1] - t_s[0]) if len(t_s) >= 2 else 0.0
    path_length = float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)))

    speed, _acceleration, jerk, vx, vy = compute_kinematics(t_s, x, y)
    mean_speed = float(speed.mean()) if len(speed) else float("nan")
    speed_cv = float(speed.std() / mean_speed) if len(speed) and mean_speed > 0 else float("nan")
    if len(jerk) and mean_speed and mean_speed > 0:
        jerk_rms = float(np.sqrt(np.mean(jerk**2)) / mean_speed)
    else:
        jerk_rms = float("nan")

    pause_ratio = float("nan")
    if len(speed) and mean_speed and mean_speed > 0:
        pause_threshold = mean_speed * PAUSE_SPEED_THRESHOLD_FRAC
        pause_ratio = float(np.mean(speed < pause_threshold))

    tremor_band_ratio, peak_freq_hz, spectral_entropy = spectral_features(vx, vy, RESAMPLE_RATE_HZ)
    deviation = spiral_deviation_feature(x, y)

    features: dict[str, float] = {
        "duration_s": duration_s,
        "path_length": path_length,
        "mean_speed": mean_speed,
        "speed_cv": speed_cv,
        "jerk_rms": jerk_rms,
        "pen_lifts": float(count_pen_lifts(points)),
        "pause_ratio": pause_ratio,
        "tremor_band_ratio": tremor_band_ratio,
        "peak_freq_hz": peak_freq_hz,
        "spectral_entropy": spectral_entropy,
        "spiral_deviation": deviation,
    }

    pen_points = [p for p in points if p.pointer_type == "pen" and p.pressure is not None]
    if pen_points:
        pressures = np.array([p.pressure for p in pen_points], dtype=np.float64)
        features["pressure_mean"] = float(pressures.mean())
        features["pressure_sd"] = float(pressures.std())

    return FeatureResult(features=features, warnings=warnings)
