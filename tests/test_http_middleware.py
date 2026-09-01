from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.deps import get_retriever
from app.main import app
from app.rate_limit import RateLimiter
from app.search.retriever import RetrievalResult, Retriever


def test_request_id_from_header_is_preserved() -> None:
    with (
        patch("app.main.Embedder"),
        patch("app.main.ChromaStore"),
        patch("app.http_middleware.log_event") as log_event_mock,
    ):
        with TestClient(app) as client:
            response = client.get(
                "/healthz",
                headers={
                    "X-Request-ID": "request-123",
                },
            )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "request-123"

    log_event_mock.assert_called_once()

    args, kwargs = log_event_mock.call_args

    assert args == ("http_request",)

    assert kwargs["request_id"] == "request-123"
    assert kwargs["method"] == "GET"
    assert kwargs["path"] == "/healthz"
    assert kwargs["status_code"] == 200
    assert kwargs["duration_ms"] >= 0


def test_request_id_is_generated_when_header_missing() -> None:
    with (
        patch("app.main.Embedder"),
        patch("app.main.ChromaStore"),
        patch(
            "app.http_middleware.uuid4",
            return_value="generated-request-id",
        ),
        patch("app.http_middleware.log_event") as log_event_mock,
    ):
        with TestClient(app) as client:
            response = client.get("/healthz")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "generated-request-id"

    _, kwargs = log_event_mock.call_args

    assert kwargs["request_id"] == "generated-request-id"


def test_rate_limit_returns_429_when_limit_exceeded() -> None:
    fake_retriever = MagicMock(spec=Retriever)
    fake_retriever.collection_name = "test-collection"
    fake_retriever.index_revision = "b" * 64

    fake_retriever.retrieve_with_metrics.return_value = RetrievalResult(
        hits=[],
        embed_ms=12.0,
        search_ms=8.0,
    )

    app.dependency_overrides[get_retriever] = lambda: fake_retriever

    try:
        with (
            patch("app.main.Embedder"),
            patch("app.main.ChromaStore"),
        ):
            with TestClient(app) as client:
                app.state.rate_limiter = RateLimiter(
                    max_requests=1,
                    window_seconds=60,
                )

                first_response = client.get(
                    "/search",
                    params={
                        "q": "Кто является источником власти?",
                    },
                )

                second_response = client.get(
                    "/search",
                    params={
                        "q": "Кто является источником власти?",
                    },
                )
    finally:
        app.dependency_overrides.clear()

    assert first_response.status_code == 200

    assert second_response.status_code == 429
    assert second_response.json() == {
        "detail": "Too many requests",
    }

    assert "X-Request-ID" in second_response.headers

    fake_retriever.retrieve_with_metrics.assert_called_once_with(
        "Кто является источником власти?",
        5,
        False,
    )
