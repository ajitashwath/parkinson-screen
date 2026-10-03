import { describe, expect, it } from "vitest";
import { makeMockResult } from "./api";
import type { DrawingCapture, StrokePoint } from "./types";

const points: StrokePoint[] = Array.from({ length: 60 }, (_, index) => ({
  x: index * 2,
  y: 300 - Math.sin(index / 5) * 40,
  t_ms: index * 30,
  pressure: 0.5,
  pointer_type: "pen",
  pen_lift: false,
}));

const capture: DrawingCapture = {
  imageDataUrl: "data:image/png;base64,ZmFrZQ==",
  kind: "spiral",
  points,
};

describe("mock API adapter", () => {
  it("returns a deterministic response shape for the same capture", () => {
    const first = makeMockResult(capture, "same-id");
    const second = makeMockResult(capture, "same-id");
    expect(first.probability).toBe(second.probability);
    expect(first.label).toBe(second.label);
    expect(first.occlusionGrid).toEqual(second.occlusionGrid);
    expect(first.tremorFeatures).toEqual(second.tremorFeatures);
    expect(first.pointerType).toBe("pen");
  });

  it("labels the response as a development preview", () => {
    const result = makeMockResult(capture);
    expect(result.modelVersion).toBe("demo-preview");
    expect(result.provenance).toContain("Development preview");
    expect(result.disclaimer).toContain("Screening aid only");
  });
});
