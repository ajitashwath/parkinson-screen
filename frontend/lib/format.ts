export const DISCLAIMER =
  "Screening aid only. This tool provides a screening signal and cannot determine a medical condition. It does not replace a qualified clinician.";

export const DEMO_BADGE = "Development preview";

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

export function formatPercent(value: number): string {
  return `${Math.round(clamp(value, 0, 1) * 100)}%`;
}

export function formatDecimal(value: number, digits = 2): string {
  return value.toFixed(digits);
}

export function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function labelText(label: string): string {
  if (label === "lower_concern") {
    return "Lower concern";
  }
  if (label === "higher_concern") {
    return "Higher concern";
  }
  return "Uncertain";
}

export function featureText(feature: string): string {
  const labels: Record<string, string> = {
    tremor_band_ratio: "Tremor-band energy",
    peak_freq_hz: "Peak movement frequency",
    speed_cv: "Speed variation",
    jerk_rms: "Jerk intensity",
    spiral_deviation: "Spiral deviation",
    pause_ratio: "Pause ratio",
    path_length: "Path length",
    mean_speed: "Mean speed",
    duration_s: "Duration",
    pen_lifts: "Pen lifts",
    spectral_entropy: "Spectral entropy",
    pressure_mean: "Mean pressure",
    pressure_sd: "Pressure variation",
  };
  return labels[feature] ?? feature.replaceAll("_", " ");
}

export function statusClass(label: string): string {
  if (label === "higher_concern") {
    return "border-violet/30 bg-linen text-violet";
  }
  if (label === "uncertain") {
    return "border-hairline bg-linen text-graphite";
  }
    return "border-hairline bg-lime-100 text-abyss";
}
