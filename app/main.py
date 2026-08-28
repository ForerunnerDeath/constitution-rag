from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.deps import get_retriever, get_store
from app.schemas import (
    ArticleChunkResponse,
    ArticleResponse,
    SearchHitResponse,
    SearchResponse,
)
from app.search.embedder import Embedder
from app.search.lexical import LexicalIndex
from app.search.retriever import Retriever
from app.search.store import ChromaStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()

    app.state.embedder = Embedder(settings.embedding_model)

    app.state.store = ChromaStore(
        path=settings.chroma_path,
        collection_name=settings.chroma_collection,
    )

    app.state.lexical_index = LexicalIndex(app.state.store.get_all())

    app.state.retriever = Retriever(
        embedder=app.state.embedder,
        store=app.state.store,
        lexical_index=app.state.lexical_index,
        min_score=settings.min_score,
    )

    yield


app = FastAPI(
    title="Constitution RAG",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz(store: ChromaStore = Depends(get_store)) -> dict[str, str | int]:
    stored = store.count()

    if stored == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chroma collection is empty",
        )

    return {
        "status": "ok",
        "stored": stored,
    }


@app.get("/search", response_model=SearchResponse)
async def search(
    q: Annotated[str, Query(min_length=3, max_length=500)],
    k: Annotated[int, Query(ge=1, le=20)] = 5,
    use_hybrid: bool = False,
    retriever: Retriever = Depends(get_retriever),
) -> SearchResponse:
    started = perf_counter()

    hits = await run_in_threadpool(retriever.retrieve, q, k, use_hybrid)

    took_ms = (perf_counter() - started) * 1000

    return SearchResponse(
        hits=[SearchHitResponse.model_validate(hit) for hit in hits],
        took_ms=took_ms,
        collection_version=retriever.collection_name,
    )


@app.get("/articles/{number}", response_model=ArticleResponse)
async def get_article(
    number: str, store: ChromaStore = Depends(get_store)
) -> ArticleResponse:
    hits = await run_in_threadpool(store.get_by_article, number)

    if not hits:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article {number} not found",
        )

    return ArticleResponse(
        article=number,
        chunks=[ArticleChunkResponse.model_validate(hit) for hit in hits],
    )
