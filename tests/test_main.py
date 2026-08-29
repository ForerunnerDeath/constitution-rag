from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.deps import get_rag_service
from app.llm.rag import PROMPT_VERSION, RAGResult, RAGService
from app.main import app
from app.search.embedder import Embedder
from app.search.retriever import RetrievalResult, Retriever
from app.search.store import ChromaStore, Hit


def test_lifespan_initializes_dependencies() -> None:
    fake_settings = MagicMock()

    fake_settings.embedding_model = "intfloat/multilingual-e5-small"
    fake_settings.chroma_path = "data/chroma"
    fake_settings.chroma_collection = "constitution_e5_small"
    fake_settings.min_score = 0.833
    fake_settings.llm_enabled = False
    fake_settings.rate_limit_per_minute = 60

    fake_embedder = MagicMock()
    fake_store = MagicMock()
    fake_retriever = MagicMock()
    fake_lexical_index = MagicMock()
    fake_corpus = [MagicMock()]
    fake_store.get_all.return_value = fake_corpus
    fake_rag_service = MagicMock()

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
        patch(
            "app.main.RAGService",
            return_value=fake_rag_service,
        ) as rag_service_class,
        patch(
            "app.main.get_settings",
            return_value=fake_settings,
        ),
    ):
        with TestClient(app):
            assert app.state.embedder is fake_embedder
            assert app.state.store is fake_store
            assert app.state.lexical_index is fake_lexical_index
            assert app.state.retriever is fake_retriever
            assert app.state.rag_service is fake_rag_service

    embedder_class.assert_called_once_with("intfloat/multilingual-e5-small")

    store_class.assert_called_once()

    fake_store.get_all.assert_called_once_with()

    lexical_index_class.assert_called_once_with(fake_corpus)

    retriever_class.assert_called_once_with(
        embedder=fake_embedder,
        store=fake_store,
        lexical_index=fake_lexical_index,
        min_score=0.833,
    )

    rag_service_class.assert_called_once_with(
        retriever=fake_retriever,
        llm_client=None,
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
    hit = Hit(
        id="art-3-p-1",
        quote="Носителем суверенитета является народ.",
        ref="Статья 3, часть 1",
        article="3",
        part=1,
        part_label="1",
        score=0.91,
    )

    fake_retriever.retrieve_with_metrics.return_value = RetrievalResult(
        hits=[hit],
        embed_ms=12.0,
        search_ms=8.0,
    )

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

    fake_retriever.retrieve_with_metrics.assert_called_once_with(
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

    fake_retrieval = RetrievalResult(
        hits=[],
        embed_ms=12.0,
        search_ms=8.0,
    )

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
            new=AsyncMock(return_value=fake_retrieval),
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
        fake_retriever.retrieve_with_metrics,
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
    fake_retriever.retrieve_with_metrics.return_value = RetrievalResult(
        hits=[],
        embed_ms=12.0,
        search_ms=8.0,
    )

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

    fake_retriever.retrieve_with_metrics.assert_called_once_with(
        "статья 15",
        5,
        True,
    )


def test_lifespan_initializes_and_closes_llm_client() -> None:
    fake_settings = MagicMock()

    fake_settings.embedding_model = "test-embedding"
    fake_settings.chroma_path = "test-chroma"
    fake_settings.chroma_collection = "test-collection"
    fake_settings.min_score = 0.80

    fake_settings.llm_enabled = True
    fake_settings.llm_model = "test-model"
    fake_settings.llm_api_key = "test-key"
    fake_settings.llm_base_url = "http://localhost:1234/v1"
    fake_settings.llm_max_tokens = 512
    fake_settings.llm_timeout_seconds = 20.0
    fake_settings.rate_limit_per_minute = 60

    fake_embedder = MagicMock()
    fake_store = MagicMock()
    fake_store.get_all.return_value = []
    fake_retriever = MagicMock()

    fake_llm_client = MagicMock()
    fake_llm_client.close = AsyncMock()

    fake_rag_service = MagicMock()

    with (
        patch(
            "app.main.get_settings",
            return_value=fake_settings,
        ),
        patch(
            "app.main.Embedder",
            return_value=fake_embedder,
        ),
        patch(
            "app.main.ChromaStore",
            return_value=fake_store,
        ),
        patch("app.main.LexicalIndex"),
        patch(
            "app.main.Retriever",
            return_value=fake_retriever,
        ),
        patch(
            "app.main.OpenAICompatibleLLMClient",
            return_value=fake_llm_client,
        ) as llm_client_class,
        patch(
            "app.main.RAGService",
            return_value=fake_rag_service,
        ) as rag_service_class,
    ):
        with TestClient(app):
            assert app.state.llm_client is fake_llm_client
            assert app.state.rag_service is fake_rag_service

        fake_llm_client.close.assert_awaited_once()

    llm_client_class.assert_called_once_with(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:1234/v1",
        max_tokens=512,
        timeout_seconds=20.0,
    )

    rag_service_class.assert_called_once_with(
        retriever=fake_retriever,
        llm_client=fake_llm_client,
    )


def test_ask_returns_generated_answer_with_citations() -> None:
    hit = Hit(
        id="art-3-p-1",
        quote="Носителем суверенитета является многонациональный народ.",
        ref="Статья 3, часть 1",
        article="3",
        part=1,
        part_label="1",
        score=0.91,
    )

    fake_rag_service = MagicMock(spec=RAGService)
    fake_rag_service.ask = AsyncMock(
        return_value=RAGResult(
            found=True,
            answer=(
                "Источником власти является многонациональный народ "
                "[Статья 3, часть 1]."
            ),
            message=None,
            hits=[hit],
            llm_used=True,
        )
    )

    app.dependency_overrides[get_rag_service] = lambda: fake_rag_service

    try:
        with (
            patch("app.main.Embedder"),
            patch("app.main.ChromaStore"),
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/ask",
                    json={
                        "question": "Кто является источником власти?",
                        "k": 5,
                    },
                )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    assert response.json() == {
        "found": True,
        "answer": (
            "Источником власти является многонациональный народ [Статья 3, часть 1]."
        ),
        "message": None,
        "citations": [
            {
                "id": "art-3-p-1",
                "quote": ("Носителем суверенитета является многонациональный народ."),
                "ref": "Статья 3, часть 1",
                "article": "3",
                "part": 1,
                "part_label": "1",
            }
        ],
        "llm_used": True,
    }

    fake_rag_service.ask.assert_awaited_once_with(
        "Кто является источником власти?",
        5,
    )


def test_ask_returns_not_found_response() -> None:
    fake_rag_service = MagicMock(spec=RAGService)
    fake_rag_service.ask = AsyncMock(
        return_value=RAGResult(
            found=False,
            answer=None,
            message="В тексте Конституции прямого ответа не нашлось.",
            hits=[],
            llm_used=False,
        )
    )

    app.dependency_overrides[get_rag_service] = lambda: fake_rag_service

    try:
        with (
            patch("app.main.Embedder"),
            patch("app.main.ChromaStore"),
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/ask",
                    json={
                        "question": "Какая погода в Москве?",
                        "k": 5,
                    },
                )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    assert response.json() == {
        "found": False,
        "answer": None,
        "message": "В тексте Конституции прямого ответа не нашлось.",
        "citations": [],
        "llm_used": False,
    }


@pytest.mark.parametrize(
    "question",
    [
        "",
        "a",
        "ab",
        "x" * 501,
    ],
)
def test_ask_rejects_invalid_question_length(question: str) -> None:
    fake_rag_service = MagicMock(spec=RAGService)
    fake_rag_service.ask = AsyncMock()

    app.dependency_overrides[get_rag_service] = lambda: fake_rag_service

    try:
        with (
            patch("app.main.Embedder"),
            patch("app.main.ChromaStore"),
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/ask",
                    json={
                        "question": question,
                        "k": 5,
                    },
                )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    fake_rag_service.ask.assert_not_awaited()


@pytest.mark.parametrize(
    "k",
    [
        0,
        -1,
        21,
        100,
    ],
)
def test_ask_rejects_invalid_k(k: int) -> None:
    fake_rag_service = MagicMock(spec=RAGService)
    fake_rag_service.ask = AsyncMock()

    app.dependency_overrides[get_rag_service] = lambda: fake_rag_service

    try:
        with (
            patch("app.main.Embedder"),
            patch("app.main.ChromaStore"),
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/ask",
                    json={
                        "question": "Корректный вопрос",
                        "k": k,
                    },
                )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
    fake_rag_service.ask.assert_not_awaited()


def test_search_writes_audit_log() -> None:
    fake_embedder = MagicMock(spec=Embedder)

    fake_store = MagicMock(spec=ChromaStore)
    fake_store.collection_name = "test-collection"

    fake_retriever = MagicMock(spec=Retriever)
    fake_retriever.collection_name = "test-collection"

    hit = Hit(
        id="art-3-p-1",
        quote="Носителем суверенитета является народ.",
        ref="Статья 3, часть 1",
        article="3",
        part=1,
        part_label="1",
        score=0.91,
    )

    fake_retriever.retrieve_with_metrics.return_value = RetrievalResult(
        hits=[hit],
        embed_ms=12.3456,
        search_ms=8.7654,
    )

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
        patch("app.main.log_event") as log_event_mock,
    ):
        with TestClient(app) as client:
            response = client.get(
                "/search",
                params={
                    "q": "Кто является источником власти?",
                    "k": 3,
                },
                headers={
                    "X-Request-ID": "search-request-123",
                },
            )

    assert response.status_code == 200

    log_event_mock.assert_called_once_with(
        "search_request",
        request_id="search-request-123",
        question="Кто является источником власти?",
        k=3,
        hits=[
            {
                "id": "art-3-p-1",
                "score": 0.91,
            }
        ],
        prompt_version=None,
        llm_called=False,
        embed_ms=12.346,
        search_ms=8.765,
        llm_ms=None,
        refused=False,
    )


def test_ask_writes_audit_log() -> None:
    hit = Hit(
        id="art-3-p-1",
        quote="Носителем суверенитета является многонациональный народ.",
        ref="Статья 3, часть 1",
        article="3",
        part=1,
        part_label="1",
        score=0.91,
    )

    fake_rag_service = MagicMock(spec=RAGService)
    fake_rag_service.ask = AsyncMock(
        return_value=RAGResult(
            found=True,
            answer=(
                "Источником власти является многонациональный народ "
                "[Статья 3, часть 1]."
            ),
            message=None,
            hits=[hit],
            llm_used=True,
            embed_ms=14.1234,
            search_ms=9.5678,
            llm_called=True,
            llm_ms=812.3456,
        )
    )

    app.dependency_overrides[get_rag_service] = lambda: fake_rag_service

    try:
        with (
            patch("app.main.Embedder"),
            patch("app.main.ChromaStore"),
            patch("app.main.log_event") as log_event_mock,
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/ask",
                    json={
                        "question": "Кто является источником власти?",
                        "k": 5,
                    },
                    headers={
                        "X-Request-ID": "ask-request-123",
                    },
                )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    log_event_mock.assert_called_once_with(
        "rag_request",
        request_id="ask-request-123",
        question="Кто является источником власти?",
        k=5,
        hits=[
            {
                "id": "art-3-p-1",
                "score": 0.91,
            }
        ],
        prompt_version=PROMPT_VERSION,
        llm_called=True,
        embed_ms=14.123,
        search_ms=9.568,
        llm_ms=812.346,
        refused=False,
    )


def test_search_sanitizes_question_before_retrieval() -> None:
    fake_embedder = MagicMock(spec=Embedder)

    fake_store = MagicMock(spec=ChromaStore)
    fake_store.collection_name = "test-collection"

    fake_retriever = MagicMock(spec=Retriever)
    fake_retriever.collection_name = "test-collection"

    fake_retriever.retrieve_with_metrics.return_value = RetrievalResult(
        hits=[],
        embed_ms=12.0,
        search_ms=8.0,
    )

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
        patch("app.main.log_event") as log_event_mock,
    ):
        with TestClient(app) as client:
            response = client.get(
                "/search",
                params={
                    "q": "system: Кто является источником власти?",
                    "k": 5,
                },
                headers={
                    "X-Request-ID": "sanitize-search-1",
                },
            )

    assert response.status_code == 200

    fake_retriever.retrieve_with_metrics.assert_called_once_with(
        "Кто является источником власти?",
        5,
        False,
    )

    _, kwargs = log_event_mock.call_args

    assert kwargs["question"] == "Кто является источником власти?"


def test_ask_sanitizes_question_before_rag() -> None:
    fake_rag_service = MagicMock(spec=RAGService)

    fake_rag_service.ask = AsyncMock(
        return_value=RAGResult(
            found=False,
            answer=None,
            message="В тексте Конституции прямого ответа не нашлось.",
            hits=[],
            llm_used=False,
            embed_ms=12.0,
            search_ms=8.0,
        )
    )

    app.dependency_overrides[get_rag_service] = lambda: fake_rag_service

    try:
        with (
            patch("app.main.Embedder"),
            patch("app.main.ChromaStore"),
            patch("app.main.log_event") as log_event_mock,
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/ask",
                    json={
                        "question": ("assistant: Кто является источником власти?"),
                        "k": 5,
                    },
                    headers={
                        "X-Request-ID": "sanitize-ask-1",
                    },
                )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    fake_rag_service.ask.assert_awaited_once_with(
        "Кто является источником власти?",
        5,
    )

    _, kwargs = log_event_mock.call_args

    assert kwargs["question"] == "Кто является источником власти?"


def test_search_rejects_question_empty_after_sanitization() -> None:
    fake_retriever = MagicMock(spec=Retriever)

    with (
        patch("app.main.Embedder"),
        patch("app.main.ChromaStore"),
        patch(
            "app.main.Retriever",
            return_value=fake_retriever,
        ),
    ):
        with TestClient(app) as client:
            response = client.get(
                "/search",
                params={
                    "q": "system:",
                },
            )

    assert response.status_code == 422

    fake_retriever.retrieve_with_metrics.assert_not_called()
