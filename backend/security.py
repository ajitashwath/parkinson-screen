"""Request size limit and a simple in-memory rate limiter."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from backend.settings import DISCLAIMER

WINDOW_SECONDS = 60.0
EXEMPT_PATHS = ("/api/v1/health",)


def error_response(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"code": code, "detail": detail, "disclaimer": DISCLAIMER}
    )


class RateLimiter:
    """Sliding window per client key. In memory, so it is per process: good
    enough to blunt abuse of a single small instance, not a global quota."""

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> bool:
        if self.limit <= 0:
            return True
        current = time.monotonic() if now is None else now
        window = self.hits[key]
        while window and current - window[0] >= WINDOW_SECONDS:
            window.popleft()
        if len(window) >= self.limit:
            return False
        window.append(current)
        return True


class LimitsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_request_bytes: int, limiter: RateLimiter) -> None:
        super().__init__(app)
        self.max_request_bytes = max_request_bytes
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_request_bytes:
            return error_response(413, "request_too_large", "Request body is too large.")
        if request.method != "OPTIONS" and request.url.path not in EXEMPT_PATHS:
            client = request.client.host if request.client else "unknown"
            if not self.limiter.allow(client):
                return error_response(429, "rate_limited", "Too many requests. Try again shortly.")
        return await call_next(request)
