from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.search.embedder import Embedder
from app.search.retriever import Retriever
from app.search.store import ChromaStore, Hit


def test_lifespan_initializes_dependencies() -> None:
    fake_embedder = MagicMock()
    fake_store = MagicMock()
    fake_retriever = MagicMock()
    fake_lexical_index = MagicMock()
    fake_corpus = [MagicMock()]
    fake_store.get_all.return_value = fake_corpus

    with (
        patch(
            "app.main.Embedder",
            return_value=fake_embedder,
        ) as embedder_class,
        patch(
            "app.main.ChromaStore",
            return_value=fake_store,
        ) as store_class,
        patch(
            "app.main.Retriever",
            return_value=fake_retriever,
        ) as retriever_class,
        patch(
            "app.main.LexicalIndex",
            return_value=fake_lexical_index,
        ) as lexical_index_class,
    ):
        with TestClient(app):
            assert app.state.embedder is fake_embedder
            assert app.state.store is fake_store
            assert app.state.lexical_index is fake_lexical_index
            assert app.state.retriever is fake_retriever

    embedder_class.assert_called_once_with("intfloat/multilingual-e5-small")

    store_class.assert_called_once()

    fake_store.get_all.assert_called_once_with()

    lexical_index_class.assert_called_once_with(fake_corpus)

    retriever_class.assert_called_once_with(
        embedder=fake_embedder,
        store=fake_store,
        lexical_index=fake_lexical_index,
        min_score=0.80,
    )


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

    fake_store = MagicMock(spec=ChromaStore)
    fake_store.collection_name = "test-collection"

    fake_retriever = MagicMock(spec=Retriever)
    fake_retriever.collection_name = "test-collection"
    fake_retriever.retrieve.return_value = [
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
        patch(
            "app.main.Retriever",
            return_value=fake_retriever,
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

    fake_retriever.retrieve.assert_called_once_with(
        "Кто является источником власти?",
        3,
        False,
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


def test_search_runs_retriever_in_threadpool() -> None:
    fake_embedder = MagicMock(spec=Embedder)

    fake_store = MagicMock(spec=ChromaStore)
    fake_store.collection_name = "test-collection"

    fake_retriever = MagicMock(spec=Retriever)
    fake_retriever.collection_name = "test-collection"

    fake_hits: list[Hit] = []

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
            "app.main.Retriever",
            return_value=fake_retriever,
        ),
        patch(
            "app.main.run_in_threadpool",
            new=AsyncMock(return_value=fake_hits),
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

    run_in_threadpool_mock.assert_awaited_once_with(
        fake_retriever.retrieve,
        "Кто является источником власти?",
        5,
        False,
    )


def test_get_article_returns_all_chunks_in_order() -> None:
    fake_embedder = MagicMock(spec=Embedder)

    fake_store = MagicMock(spec=ChromaStore)
    fake_store.collection_name = "test-collection"

    fake_store.get_by_article.return_value = [
        Hit(
            id="art-81-p-1",
            quote="Часть 1.",
            ref="Статья 81, часть 1",
            article="81",
            part=1,
            part_label="1",
            score=1.0,
        ),
        Hit(
            id="art-81-p-2-c-1",
            quote="Часть 2, фрагмент 1.",
            ref="Статья 81, часть 2",
            article="81",
            part=2,
            part_label="2",
            score=1.0,
        ),
        Hit(
            id="art-81-p-2-c-2",
            quote="Часть 2, фрагмент 2.",
            ref="Статья 81, часть 2",
            article="81",
            part=2,
            part_label="2",
            score=1.0,
        ),
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
            response = client.get("/articles/81")

    assert response.status_code == 200

    data = response.json()

    assert data["article"] == "81"
    assert [chunk["id"] for chunk in data["chunks"]] == [
        "art-81-p-1",
        "art-81-p-2-c-1",
        "art-81-p-2-c-2",
    ]

    assert "score" not in data["chunks"][0]


def test_get_article_accepts_decimal_article_number() -> None:
    fake_embedder = MagicMock(spec=Embedder)

    fake_store = MagicMock(spec=ChromaStore)
    fake_store.collection_name = "test-collection"

    fake_store.get_by_article.return_value = [
        Hit(
            id="art-67.1-p-1",
            quote="Текст статьи 67.1.",
            ref="Статья 67.1, часть 1",
            article="67.1",
            part=1,
            part_label="1",
            score=1.0,
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
            response = client.get("/articles/67.1")

    assert response.status_code == 200
    assert response.json()["article"] == "67.1"

    fake_store.get_by_article.assert_called_once_with("67.1")


def test_get_article_returns_404_when_article_not_found() -> None:
    fake_embedder = MagicMock(spec=Embedder)

    fake_store = MagicMock(spec=ChromaStore)
    fake_store.collection_name = "test-collection"
    fake_store.get_by_article.return_value = []

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
            response = client.get("/articles/999")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Article 999 not found",
    }


def test_get_article_does_not_use_embedder() -> None:
    fake_embedder = MagicMock(spec=Embedder)

    fake_store = MagicMock(spec=ChromaStore)
    fake_store.collection_name = "test-collection"
    fake_store.get_by_article.return_value = [
        Hit(
            id="art-67.1",
            quote="Текст статьи 67.1.",
            ref="Статья 67.1",
            article="67.1",
            part=None,
            part_label=None,
            score=1.0,
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
            response = client.get("/articles/67.1")

    assert response.status_code == 200

    fake_embedder.embed_query.assert_not_called()
    fake_store.search.assert_not_called()

    fake_store.get_by_article.assert_called_once_with("67.1")


def test_search_enables_hybrid_retrieval() -> None:
    fake_embedder = MagicMock(spec=Embedder)

    fake_store = MagicMock(spec=ChromaStore)
    fake_store.collection_name = "test-collection"

    fake_retriever = MagicMock(spec=Retriever)
    fake_retriever.collection_name = "test-collection"
    fake_retriever.retrieve.return_value = []

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
            "app.main.Retriever",
            return_value=fake_retriever,
        ),
    ):
        with TestClient(app) as client:
            response = client.get(
                "/search",
                params={
                    "q": "статья 15",
                    "k": 5,
                    "use_hybrid": "true",
                },
            )

    assert response.status_code == 200

    fake_retriever.retrieve.assert_called_once_with(
        "статья 15",
        5,
        True,
    )
