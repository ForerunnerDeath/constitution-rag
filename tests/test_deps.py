from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from app.deps import get_embedder, get_rag_service, get_store
from app.llm.rag import RAGService
from app.search.embedder import Embedder
from app.search.store import ChromaStore


def _make_request(app: FastAPI) -> Request:
    scope = {
        "type": "http",
        "app": app,
    }

    return Request(scope)


def test_get_embedder_returns_initialized_embedder() -> None:
    app = FastAPI()
    embedder = MagicMock(spec=Embedder)
    app.state.embedder = embedder

    request = _make_request(app)

    result = get_embedder(request)

    assert result is embedder


def test_get_embedder_rejects_missing_embedder() -> None:
    app = FastAPI()
    request = _make_request(app)

    with pytest.raises(
        RuntimeError,
        match="Embedder is not initialized",
    ):
        get_embedder(request)


def test_get_store_returns_initialized_store() -> None:
    app = FastAPI()
    store = MagicMock(spec=ChromaStore)
    app.state.store = store

    request = _make_request(app)

    result = get_store(request)

    assert result is store


def test_get_store_rejects_missing_store() -> None:
    app = FastAPI()
    request = _make_request(app)

    with pytest.raises(
        RuntimeError,
        match="ChromaStore is not initialized",
    ):
        get_store(request)


def test_get_rag_service_returns_initialized_service() -> None:
    app = FastAPI()
    rag_service = MagicMock(spec=RAGService)
    app.state.rag_service = rag_service

    request = _make_request(app)

    result = get_rag_service(request)

    assert result is rag_service


def test_get_rag_service_rejects_missing_service() -> None:
    app = FastAPI()
    request = _make_request(app)

    with pytest.raises(
        RuntimeError,
        match="RAGService is not initialized",
    ):
        get_rag_service(request)
