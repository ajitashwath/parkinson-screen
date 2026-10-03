import { DISCLAIMER, clamp } from "./format";
import type {
  ApiPredictionResponse,
  ApiTremorResponse,
  DrawingCapture,
  DrawingKind,
  FeatureContribution,
  HealthStatus,
  PredictionResult,
  SessionRecord,
  StrokePoint,
  TremorReport,
} from "./types";
import { saveResult, getStoredResult } from "./storage";

const apiUrl = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");
const useMockApi = process.env.NEXT_PUBLIC_USE_MOCK_API === "true" || !apiUrl;

export const isMockMode = useMockApi;

export class ApiClientError extends Error {
  status: number;

  constructor(message: string, status = 500) {
    super(message);
    this.name = "ApiClientError";
    this.status = status;
  }
}

function textHash(value: string): number {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash << 5) - hash + value.charCodeAt(index);
    hash |= 0;
  }
  return Math.abs(hash);
}

function pointsSignature(points: StrokePoint[]): string {
  return points
    .filter((_, index) => index % 3 === 0)
    .map((point) => `${point.x.toFixed(1)}:${point.y.toFixed(1)}`)
    .join("|");
}

function featureSet(kind: DrawingKind, seed: number): Record<string, number> {
  const kindOffset = kind === "spiral" ? 0.02 : -0.01;
  return {
    duration_s: 5.2 + (seed % 17) / 10,
    path_length: 820 + (seed % 240),
    mean_speed: 126 + (seed % 54),
    speed_cv: 0.31 + ((seed % 12) / 100) + kindOffset,
    jerk_rms: 0.22 + ((seed % 9) / 100) + kindOffset,
    pen_lifts: seed % 4,
    pause_ratio: 0.04 + ((seed % 8) / 100),
    tremor_band_ratio: clamp(0.2 + ((seed % 48) / 100) + kindOffset, 0.02, 0.98),
    peak_freq_hz: 4.2 + ((seed % 55) / 10),
    spectral_entropy: 0.55 + ((seed % 20) / 100),
    spiral_deviation: kind === "spiral" ? 0.07 + ((seed % 15) / 100) : 0,
    pressure_mean: 0.46 + ((seed % 12) / 100),
    pressure_sd: 0.08 + ((seed % 7) / 100),
  };
}

function contributions(seed: number): FeatureContribution[] {
  const values = [
    ["tremor_band_ratio", 0.18 + (seed % 11) / 100, "Energy in the movement frequency band"],
    ["peak_freq_hz", 0.1 + (seed % 7) / 100, "Frequency of the strongest movement rhythm"],
    ["speed_cv", 0.07 + (seed % 6) / 100, "Variation in movement speed"],
    ["jerk_rms", 0.04 + (seed % 5) / 100, "Rate of change in movement direction"],
  ];
  return values.map(([feature, value, explanation]) => ({
    feature: String(feature),
    value: Number(value),
    explanation: String(explanation),
  }));
}

function occlusionGrid(seed: number, size: number): number[][] {
  const center = (size - 1) / 2;
  return Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, column) => {
      const distance = Math.sqrt((row - center) ** 2 + (column - center) ** 2) / center;
      const ripple = Math.sin((row + 1) * 1.7 + seed) * 0.035;
      return clamp(0.82 - distance * 0.72 + ripple, 0.02, 0.98);
    }),
  );
}

function pointerType(points: StrokePoint[]): string {
  const types = new Set(points.map((point) => point.pointer_type));
  return types.size === 1 ? String(types.values().next().value ?? "mouse") : "mixed";
}

export function makeMockResult(capture?: DrawingCapture, forcedId?: string): PredictionResult {
  const points = capture?.points ?? [];
  const kind = capture?.kind ?? "spiral";
  const seed = textHash(`${kind}:${pointsSignature(points)}:${capture?.imageDataUrl.length ?? 0}`);
  const probability = clamp(0.22 + (seed % 57) / 100, 0.05, 0.94);
  const threshold = 0.5;
  const uncertainMargin = 0.05;
  const label =
    Math.abs(probability - threshold) <= uncertainMargin
      ? "uncertain"
      : probability >= threshold
        ? "higher_concern"
        : "lower_concern";
  const features = featureSet(kind, seed);
  const warnings = [
    "Demo response: connect the model service for research-grade interpretation.",
    ...(pointerType(points) === "mouse" ? ["Mouse input can add movement noise. A stylus or touchscreen is preferred."] : []),
    ...(points.length < 50 ? ["This trace has fewer than 50 samples, so kinematic features may be unstable."] : []),
  ];
  return {
    id: forcedId ?? `demo-${Date.now().toString(36)}`,
    createdAt: new Date().toISOString(),
    kind,
    pointerType: pointerType(points),
    probability,
    label,
    threshold,
    uncertainMargin,
    modelVersion: "demo-preview",
    provenance: "Development preview. No clinical model result is being presented.",
    occlusionGrid: occlusionGrid(seed, 8),
    occlusionGridSize: 8,
    tremorFeatures: features,
    tremorScore: features.tremor_band_ratio ?? null,
    spiralDeviation: kind === "spiral" ? features.spiral_deviation : null,
    peakFrequencyHz: features.peak_freq_hz ?? null,
    warnings,
    contributions: contributions(seed),
    disclaimer: DISCLAIMER,
  };
}

function mockHealth(): HealthStatus {
  return {
    status: "demo",
    imageModel: "demo-preview",
    tremorModel: "demo-preview",
    message: "The local demo adapter is active. Connect FastAPI to load research models.",
  };
}

function mapPrediction(raw: ApiPredictionResponse, kind: DrawingKind): PredictionResult {
  const threshold = raw.threshold ?? 0.5;
  const uncertainMargin = raw.uncertain_margin ?? 0.05;
  const probability = clamp(raw.probability, 0, 1);
  const label =
    raw.label ??
    (Math.abs(probability - threshold) <= uncertainMargin
      ? "uncertain"
      : probability >= threshold
        ? "higher_concern"
        : "lower_concern");
  const grid = raw.occlusion_map?.values ?? [];
  const gridSize = raw.occlusion_map?.grid_size ?? grid.length;
  return {
    id: raw.id ?? `result-${Date.now().toString(36)}`,
    createdAt: new Date().toISOString(),
    kind,
    pointerType: "mixed",
    probability,
    label,
    threshold,
    uncertainMargin,
    modelVersion: raw.model_version ?? "api-model",
    provenance: raw.provenance ?? "Model service response",
    occlusionGrid: grid,
    occlusionGridSize: gridSize,
    tremorFeatures: {},
    tremorScore: null,
    spiralDeviation: null,
    peakFrequencyHz: null,
    warnings: raw.warnings ?? [],
    contributions: [],
    disclaimer: DISCLAIMER,
  };
}

function mapTremor(raw: ApiTremorResponse): TremorReport {
  const features = raw.features ?? {};
  return {
    features,
    warnings: raw.warnings ?? [],
    score: raw.score ?? null,
    topContributions: raw.top_contributions ?? [],
    provenance: raw.provenance ?? "Model service response",
  };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (!apiUrl) {
    throw new ApiClientError("The API URL is not configured.");
  }
  const response = await fetch(`${apiUrl}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let message = `Request failed with status ${response.status}.`;
    try {
      const body = (await response.json()) as { detail?: string };
      message = body.detail ?? message;
    } catch {
      message = response.statusText || message;
    }
    throw new ApiClientError(message, response.status);
  }
  return (await response.json()) as T;
}

export async function getHealth(): Promise<HealthStatus> {
  if (isMockMode) {
    return mockHealth();
  }
  return request<HealthStatus>("/api/v1/health");
}

export async function createPrediction(capture: DrawingCapture): Promise<PredictionResult> {
  let result: PredictionResult;
  if (isMockMode) {
    result = makeMockResult(capture);
  } else {
    const imageBase64 = capture.imageDataUrl.includes(",")
      ? capture.imageDataUrl.split(",")[1]
      : capture.imageDataUrl;
    const raw = await request<ApiPredictionResponse>("/api/v1/predict", {
      method: "POST",
      body: JSON.stringify({ image_base64: imageBase64, kind: capture.kind }),
    });
    result = { ...mapPrediction(raw, capture.kind), pointerType: pointerType(capture.points) };
    if (capture.points.length > 0) {
      try {
        const tremorRaw = await request<ApiTremorResponse>("/api/v1/tremor", {
          method: "POST",
          body: JSON.stringify({ points: capture.points, pointer_type: pointerType(capture.points) }),
        });
        const tremor = mapTremor(tremorRaw);
        result = {
          ...result,
          tremorFeatures: tremor.features,
          tremorScore: tremor.score,
          warnings: [...result.warnings, ...tremor.warnings],
          contributions: tremor.topContributions,
          provenance: `${result.provenance} Tremor: ${tremor.provenance}`,
          spiralDeviation: tremor.features.spiral_deviation ?? result.spiralDeviation,
          peakFrequencyHz: tremor.features.peak_freq_hz ?? result.peakFrequencyHz,
        };
      } catch (error) {
        result = {
          ...result,
          warnings: [
            ...result.warnings,
            error instanceof Error ? error.message : "Tremor analysis was unavailable.",
          ],
        };
      }
    }
  }
  saveResult(result);
  if (!isMockMode) {
    await saveSessionToApi(result);
  }
  return result;
}

async function saveSessionToApi(result: PredictionResult): Promise<void> {
  const { getDeviceId } = await import("./device");
  try {
    await request<unknown>("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify({
        device_id: getDeviceId(),
        kind: result.kind,
        label: result.label,
        probability: result.probability,
        tremor_score: result.tremorScore,
        spiral_deviation: result.spiralDeviation,
        pointer_type: result.pointerType,
        model_version: result.modelVersion,
        features: result.tremorFeatures,
      }),
    });
  } catch {
    // History is a convenience. A failed save must not hide the result.
  }
}

export async function getResult(id: string): Promise<PredictionResult | null> {
  const stored = getStoredResult(id);
  if (stored) {
    return stored;
  }
  if (isMockMode) {
    const result = makeMockResult(undefined, id);
    saveResult(result);
    return result;
  }
  return null;
}

export async function getSessionsFromApi(): Promise<SessionRecord[] | null> {
  if (isMockMode) {
    return null;
  }
  const { getDeviceId } = await import("./device");
  const data = await request<{ sessions: SessionRecord[] }>(
    `/api/v1/sessions?device_id=${encodeURIComponent(getDeviceId())}`,
  );
  return data.sessions;
}

export async function deleteSessionsFromApi(): Promise<boolean> {
  if (isMockMode) {
    return true;
  }
  const { getDeviceId } = await import("./device");
  await request<unknown>(`/api/v1/sessions?device_id=${encodeURIComponent(getDeviceId())}`, {
    method: "DELETE",
  });
  return true;
}
