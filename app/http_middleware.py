from time import perf_counter
from uuid import uuid4

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from app.observability import log_event
from app.rate_limit import RateLimiter


def should_rate_limit(path: str) -> bool:
    return path == "/search" or path == "/ask" or path.startswith("/articles/")


async def request_context_middleware(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    request_id = request.headers.get("X-Request-ID") or str(uuid4())

    request.state.request_id = request_id

    started = perf_counter()
    status_code = 500

    try:
        if should_rate_limit(request.url.path):
            rate_limiter: RateLimiter = request.app.state.rate_limiter

            client_id = request.client.host if request.client is not None else "unknown"

            if not rate_limiter.allow(client_id):
                response = JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Too many requests",
                    },
                )
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)

        status_code = response.status_code

        response.headers["X-Request-ID"] = request_id

        return response

    finally:
        duration_ms = (perf_counter() - started) * 1000

        log_event(
            "http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_ms=round(duration_ms, 3),
        )
