export const DRAWING_KINDS = ["spiral", "wave"] as const;

export type DrawingKind = (typeof DRAWING_KINDS)[number];
export type ScreeningLabel = "lower_concern" | "higher_concern" | "uncertain";
export type PointerType = "mouse" | "pen" | "touch" | string;

export interface StrokePoint {
  x: number;
  y: number;
  t_ms: number;
  pressure: number;
  pointer_type: PointerType;
  pen_lift: boolean;
}

export interface DrawingCapture {
  imageDataUrl: string;
  kind: DrawingKind;
  points: StrokePoint[];
}

export interface DrawingQuality {
  durationS: number;
  isReady: boolean;
  pointCount: number;
  warnings: string[];
}

export interface FeatureContribution {
  feature: string;
  value: number;
  explanation: string;
}

export interface PredictionResult {
  id: string;
  createdAt: string;
  kind: DrawingKind;
  pointerType: PointerType;
  probability: number;
  label: ScreeningLabel;
  threshold: number;
  uncertainMargin: number;
  modelVersion: string;
  provenance: string;
  occlusionGrid: number[][];
  occlusionGridSize: number;
  tremorFeatures: Record<string, number>;
  tremorScore: number | null;
  spiralDeviation: number | null;
  peakFrequencyHz: number | null;
  warnings: string[];
  contributions: FeatureContribution[];
  disclaimer: string;
}

export interface SessionRecord {
  id: string;
  createdAt: string;
  kind: DrawingKind;
  label: ScreeningLabel;
  probability: number;
  tremorScore: number | null;
  spiralDeviation: number | null;
  pointerType: PointerType;
}

export interface HealthStatus {
  status: "ok" | "demo" | "unavailable";
  imageModel: string;
  tremorModel: string;
  message: string;
}

export interface TremorReport {
  features: Record<string, number>;
  warnings: string[];
  score: number | null;
  topContributions: FeatureContribution[];
  provenance: string;
}

export interface ApiPredictionResponse {
  id?: string;
  probability: number;
  label?: ScreeningLabel;
  threshold?: number;
  uncertain_margin?: number;
  model_version?: string;
  provenance?: string;
  occlusion_map?: {
    grid_size?: number;
    values?: number[][];
  };
  warnings?: string[];
}

export interface ApiTremorResponse {
  features?: Record<string, number>;
  warnings?: string[];
  score?: number | null;
  top_contributions?: FeatureContribution[];
  provenance?: string;
}
