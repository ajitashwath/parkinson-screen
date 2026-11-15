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

  it("returns sessions sorted newest-first and caps history at 30", () => {
    for (let index = 0; index < 35; index += 1) {
      saveResult({
        ...result,
        id: `result-${index}`,
        createdAt: `2026-01-${String(index + 1).padStart(2, "0")}T10:00:00.000Z`,
      });
    }
    const sessions = getSessions();
    expect(sessions).toHaveLength(30);
    expect(sessions[0]?.id).toBe("result-34");
    expect(sessions.at(-1)?.id).toBe("result-5");
  });

  it("returns null for a missing result and tolerates corrupt stored data", () => {
    expect(getStoredResult("missing")).toBeNull();
    window.localStorage.setItem("screen-results-v1", "{not-json");
    window.localStorage.setItem("screen-sessions-v1", "[broken");
    expect(getStoredResult("anything")).toBeNull();
    expect(getSessions()).toEqual([]);
  });
});
