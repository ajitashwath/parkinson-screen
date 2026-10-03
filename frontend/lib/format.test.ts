import { describe, expect, it } from "vitest";
import { featureText, formatPercent, labelText, statusClass } from "./format";

describe("format helpers", () => {
  it("formats probabilities as bounded percentages", () => {
    expect(formatPercent(0.256)).toBe("26%");
    expect(formatPercent(1.4)).toBe("100%");
    expect(formatPercent(-0.2)).toBe("0%");
  });

  it("maps screening labels to readable copy", () => {
    expect(labelText("lower_concern")).toBe("Lower concern");
    expect(labelText("higher_concern")).toBe("Higher concern");
    expect(labelText("uncertain")).toBe("Uncertain");
  });

  it("returns distinct status styles", () => {
    expect(statusClass("lower_concern")).toContain("lime");
    expect(statusClass("higher_concern")).toContain("violet");
    expect(statusClass("uncertain")).toContain("graphite");
  });

  it("gives every backend feature a capitalized label", () => {
    for (const key of ["duration_s", "pen_lifts", "spectral_entropy", "pressure_mean", "pressure_sd"]) {
      const text = featureText(key);
      expect(text).not.toContain("_");
      expect(text[0]).toBe(text[0].toUpperCase());
    }
  });
});
