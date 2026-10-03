const DEVICE_KEY = "screen-device-id";

function createDeviceId(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `device-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function getDeviceId(): string {
  if (typeof window === "undefined") {
    return "server-device";
  }
  const existing = window.localStorage.getItem(DEVICE_KEY);
  if (existing) {
    return existing;
  }
  const deviceId = createDeviceId();
  window.localStorage.setItem(DEVICE_KEY, deviceId);
  return deviceId;
}

export function getDeviceLabel(): string {
  const id = getDeviceId();
  return id.slice(0, 8).toUpperCase();
}
