import { beforeEach, describe, expect, it } from "vitest";
import { clearStoredData, getSessions, getStoredResult, saveResult } from "./storage";
import type { PredictionResult } from "./types";

const result: PredictionResult = {
  id: "test-result",
  createdAt: "2026-01-01T10:00:00.000Z",
  kind: "spiral",
  pointerType: "pen",
  probability: 0.62,
  label: "higher_concern",
  threshold: 0.5,
  uncertainMargin: 0.05,
  modelVersion: "test",
  provenance: "test",
  occlusionGrid: [[0.1]],
  occlusionGridSize: 1,
  tremorFeatures: { tremor_band_ratio: 0.2 },
  tremorScore: 0.2,
  spiralDeviation: 0.1,
  peakFrequencyHz: 6,
  warnings: [],
  contributions: [],
  disclaimer: "test disclaimer",
};

describe("local result storage", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("stores result metadata and exposes it in history", () => {
    saveResult(result);
    expect(getStoredResult(result.id)).toEqual(result);
    expect(getSessions()).toEqual([
      {
        id: result.id,
        createdAt: result.createdAt,
        kind: result.kind,
        label: result.label,
        probability: result.probability,
        tremorScore: result.tremorScore,
        spiralDeviation: result.spiralDeviation,
        pointerType: result.pointerType,
      },
    ]);
  });

  it("removes all browser history records", () => {
    saveResult(result);
    clearStoredData();
    expect(getStoredResult(result.id)).toBeNull();
    expect(getSessions()).toEqual([]);
  });
});
