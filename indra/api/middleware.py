"""Request correlation, API-key authentication, and lightweight local rate limiting."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from indra.core.config import Settings
from indra.core.exceptions import AuthenticationError, RateLimitExceededError
from indra.core.ids import correlation_context, new_id
from indra.core.logging import get_logger

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a correlation id to every request and response."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID") or new_id("job")
        with correlation_context(correlation_id, agent="api"):
            response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response


class SecurityMiddleware(BaseHTTPMiddleware):
    """Enforce configured API keys and an in-process per-IP sliding-window rate limit."""

    def __init__(self, app: object, settings: Settings) -> None:
        super().__init__(app)
        self._settings = settings
        self._requests: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.url.path in {"/health", "/docs", "/openapi.json"}:
            return await call_next(request)
        try:
            self._authorize(request)
            self._rate_limit(request)
        except AuthenticationError as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.to_dict())
        except RateLimitExceededError as exc:
            return JSONResponse(status_code=exc.status_code, content=exc.to_dict())
        return await call_next(request)

    def _authorize(self, request: Request) -> None:
        if not self._settings.auth_enabled:
            return
        supplied = request.headers.get("X-API-Key")
        expected = {item.get_secret_value() for item in self._settings.api_keys}
        if not supplied or supplied not in expected:
            raise AuthenticationError("Supply a valid X-API-Key header to access INDRA.")

    def _rate_limit(self, request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self._requests[client]
        while window and now - window[0] >= 60.0:
            window.popleft()
        if len(window) >= self._settings.rate_limit_per_minute:
            raise RateLimitExceededError("Request limit reached. Wait one minute before retrying.")
        window.append(now)


__all__ = ["RequestContextMiddleware", "SecurityMiddleware"]
