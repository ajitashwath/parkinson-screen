import type { PredictionResult, SessionRecord } from "./types";

const RESULTS_KEY = "screen-results-v1";
const HISTORY_KEY = "screen-sessions-v1";

function isBrowser(): boolean {
  return typeof window !== "undefined";
}

function readJson<T>(key: string, fallback: T): T {
  if (!isBrowser()) {
    return fallback;
  }
  try {
    const value = window.localStorage.getItem(key);
    return value ? (JSON.parse(value) as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeJson<T>(key: string, value: T): void {
  if (!isBrowser()) {
    return;
  }
  window.localStorage.setItem(key, JSON.stringify(value));
}

export function saveResult(result: PredictionResult): void {
  const results = readJson<Record<string, PredictionResult>>(RESULTS_KEY, {});
  results[result.id] = result;
  writeJson(RESULTS_KEY, results);

  const sessions = readJson<SessionRecord[]>(HISTORY_KEY, []);
  const record: SessionRecord = {
    id: result.id,
    createdAt: result.createdAt,
    kind: result.kind,
    label: result.label,
    probability: result.probability,
    tremorScore: result.tremorScore,
    spiralDeviation: result.spiralDeviation,
    pointerType: result.pointerType,
  };
  const withoutCurrent = sessions.filter((session) => session.id !== result.id);
  writeJson(HISTORY_KEY, [record, ...withoutCurrent].slice(0, 30));
}

export function getStoredResult(id: string): PredictionResult | null {
  const results = readJson<Record<string, PredictionResult>>(RESULTS_KEY, {});
  return results[id] ?? null;
}

export function getSessions(): SessionRecord[] {
  return readJson<SessionRecord[]>(HISTORY_KEY, []).sort((left, right) =>
    right.createdAt.localeCompare(left.createdAt),
  );
}

export function clearStoredData(): void {
  if (!isBrowser()) {
    return;
  }
  window.localStorage.removeItem(RESULTS_KEY);
  window.localStorage.removeItem(HISTORY_KEY);
}
