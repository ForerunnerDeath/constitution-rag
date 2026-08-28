from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.search.embedder import Embedder
from app.search.store import ChromaStore, Hit


def test_lifespan_initializes_dependencies() -> None:
    fake_embedder = MagicMock()
    fake_store = MagicMock()

    with (
        patch(
            "app.main.Embedder",
            return_value=fake_embedder,
        ) as embedder_class,
        patch(
            "app.main.ChromaStore",
            return_value=fake_store,
        ) as store_class,
    ):
        with TestClient(app):
            assert app.state.embedder is fake_embedder
            assert app.state.store is fake_store

    embedder_class.assert_called_once_with("intfloat/multilingual-e5-small")

    store_class.assert_called_once()


def test_healthz() -> None:
    with (
        patch("app.main.Embedder"),
        patch("app.main.ChromaStore"),
    ):
        with TestClient(app) as client:
            response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_returns_200_when_collection_has_data() -> None:
    fake_store = MagicMock(spec=ChromaStore)
    fake_store.count.return_value = 383

    with (
        patch("app.main.Embedder"),
        patch(
            "app.main.ChromaStore",
            return_value=fake_store,
        ),
    ):
        with TestClient(app) as client:
            response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "stored": 383,
    }


def test_readyz_returns_503_when_collection_is_empty() -> None:
    fake_store = MagicMock(spec=ChromaStore)
    fake_store.count.return_value = 0

    with (
        patch("app.main.Embedder"),
        patch(
            "app.main.ChromaStore",
            return_value=fake_store,
        ),
    ):
        with TestClient(app) as client:
            response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json() == {"detail": "Chroma collection is empty"}


def test_search_returns_retrieval_results() -> None:
    fake_embedder = MagicMock(spec=Embedder)
    fake_embedder.embed_query.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    fake_store = MagicMock(spec=ChromaStore)
    fake_store.collection_name = "test-collection"
    fake_store.search.return_value = [
        Hit(
            id="art-3-p-1",
            quote="Носителем суверенитета является народ.",
            ref="Статья 3, часть 1",
            article="3",
            part=1,
            part_label="1",
            score=0.91,
        )
    ]

    with (
        patch(
            "app.main.Embedder",
            return_value=fake_embedder,
        ),
        patch(
            "app.main.ChromaStore",
            return_value=fake_store,
        ),
    ):
        with TestClient(app) as client:
            response = client.get(
                "/search",
                params={
                    "q": "Кто является источником власти?",
                    "k": 3,
                },
            )

    assert response.status_code == 200

    body = response.json()

    assert body["hits"] == [
        {
            "id": "art-3-p-1",
            "quote": "Носителем суверенитета является народ.",
            "ref": "Статья 3, часть 1",
            "article": "3",
            "part": 1,
            "part_label": "1",
            "score": 0.91,
        }
    ]

    assert body["collection_version"] == "test-collection"
    assert body["took_ms"] >= 0

    fake_embedder.embed_query.assert_called_once_with("Кто является источником власти?")

    fake_store.search.assert_called_once_with(
        [0.1, 0.2, 0.3],
        3,
    )


@pytest.mark.parametrize(
    "q",
    [
        "",
        "a",
        "ab",
        "x" * 501,
    ],
)
def test_search_rejects_invalid_query_length(q: str) -> None:
    fake_embedder = MagicMock(spec=Embedder)
    fake_store = MagicMock(spec=ChromaStore)

    with (
        patch(
            "app.main.Embedder",
            return_value=fake_embedder,
        ),
        patch(
            "app.main.ChromaStore",
            return_value=fake_store,
        ),
    ):
        with TestClient(app) as client:
            response = client.get(
                "/search",
                params={"q": q},
            )

    assert response.status_code == 422
    fake_embedder.embed_query.assert_not_called()
    fake_store.search.assert_not_called()


@pytest.mark.parametrize(
    "k",
    [
        0,
        -1,
        21,
        100,
    ],
)
def test_search_rejects_invalid_k(k: int) -> None:
    fake_embedder = MagicMock(spec=Embedder)
    fake_store = MagicMock(spec=ChromaStore)

    with (
        patch(
            "app.main.Embedder",
            return_value=fake_embedder,
        ),
        patch(
            "app.main.ChromaStore",
            return_value=fake_store,
        ),
    ):
        with TestClient(app) as client:
            response = client.get(
                "/search",
                params={
                    "q": "Корректный вопрос",
                    "k": k,
                },
            )

    assert response.status_code == 422
    fake_embedder.embed_query.assert_not_called()
    fake_store.search.assert_not_called()


def test_search_runs_blocking_operations_in_threadpool() -> None:
    fake_embedder = MagicMock(spec=Embedder)

    fake_store = MagicMock(spec=ChromaStore)
    fake_store.collection_name = "test-collection"

    fake_vector = [0.1, 0.2, 0.3]
    fake_hits = []

    with (
        patch(
            "app.main.Embedder",
            return_value=fake_embedder,
        ),
        patch(
            "app.main.ChromaStore",
            return_value=fake_store,
        ),
        patch(
            "app.main.run_in_threadpool",
            new=AsyncMock(side_effect=[fake_vector, fake_hits]),
        ) as run_in_threadpool_mock,
    ):
        with TestClient(app) as client:
            response = client.get(
                "/search",
                params={
                    "q": "Кто является источником власти?",
                },
            )

    assert response.status_code == 200

    assert run_in_threadpool_mock.await_count == 2

    assert run_in_threadpool_mock.await_args_list[0].args == (
        fake_embedder.embed_query,
        "Кто является источником власти?",
    )

    assert run_in_threadpool_mock.await_args_list[1].args == (
        fake_store.search,
        fake_vector,
        5,
    )
