from unittest.mock import patch

import pytest

from app.rate_limit import RateLimiter


def test_rate_limiter_allows_requests_below_limit() -> None:
    limiter = RateLimiter(
        max_requests=3,
        window_seconds=60,
    )

    with patch(
        "app.rate_limit.perf_counter",
        side_effect=[
            10.0,
            20.0,
            30.0,
        ],
    ):
        assert limiter.allow("client-a") is True
        assert limiter.allow("client-a") is True
        assert limiter.allow("client-a") is True


def test_rate_limiter_rejects_request_over_limit() -> None:
    limiter = RateLimiter(
        max_requests=2,
        window_seconds=60,
    )

    with patch(
        "app.rate_limit.perf_counter",
        side_effect=[
            10.0,
            20.0,
            30.0,
        ],
    ):
        assert limiter.allow("client-a") is True
        assert limiter.allow("client-a") is True
        assert limiter.allow("client-a") is False


def test_rate_limiter_allows_request_after_window_expires() -> None:
    limiter = RateLimiter(
        max_requests=2,
        window_seconds=60,
    )

    with patch(
        "app.rate_limit.perf_counter",
        side_effect=[
            10.0,
            20.0,
            71.0,
        ],
    ):
        assert limiter.allow("client-a") is True
        assert limiter.allow("client-a") is True
        assert limiter.allow("client-a") is True


def test_rate_limiter_tracks_clients_separately() -> None:
    limiter = RateLimiter(
        max_requests=1,
        window_seconds=60,
    )

    with patch(
        "app.rate_limit.perf_counter",
        side_effect=[
            10.0,
            11.0,
            12.0,
        ],
    ):
        assert limiter.allow("client-a") is True
        assert limiter.allow("client-a") is False

        assert limiter.allow("client-b") is True


@pytest.mark.parametrize(
    ("max_requests", "window_seconds"),
    [
        (0, 60),
        (-1, 60),
        (10, 0),
        (10, -1),
    ],
)
def test_rate_limiter_rejects_invalid_configuration(
    max_requests: int,
    window_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        RateLimiter(
            max_requests=max_requests,
            window_seconds=window_seconds,
        )
