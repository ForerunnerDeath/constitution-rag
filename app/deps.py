from fastapi import Request

from app.search.embedder import Embedder
from app.search.retriever import Retriever
from app.search.store import ChromaStore


def get_embedder(request: Request) -> Embedder:
    embedder = getattr(request.app.state, "embedder", None)

    if not isinstance(embedder, Embedder):
        raise RuntimeError("Embedder is not initialized")

    return embedder


def get_store(request: Request) -> ChromaStore:
    store = getattr(request.app.state, "store", None)

    if not isinstance(store, ChromaStore):
        raise RuntimeError("ChromaStore is not initialized")

    return store


def get_retriever(request: Request) -> Retriever:
    retriever = getattr(request.app.state, "retriever", None)

    if not isinstance(retriever, Retriever):
        raise RuntimeError("Retriever is not initialized")

    return retriever
