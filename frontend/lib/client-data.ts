import { getSessions, clearStoredData } from "./storage";

export function loadLocalSessions(): ReturnType<typeof getSessions> {
  return getSessions();
}

export function removeLocalData(): void {
  clearStoredData();
}
