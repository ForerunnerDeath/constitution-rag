from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.deps import get_embedder, get_store
from app.schemas import SearchHitResponse, SearchResponse
from app.search.embedder import Embedder
from app.search.store import ChromaStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()

    app.state.embedder = Embedder(settings.embedding_model)

    app.state.store = ChromaStore(
        path=settings.chroma_path,
        collection_name=settings.chroma_collection,
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
    embedder: Embedder = Depends(get_embedder),
    store: ChromaStore = Depends(get_store),
) -> SearchResponse:
    started = perf_counter()

    vector = await run_in_threadpool(embedder.embed_query, q)

    hits = await run_in_threadpool(store.search, vector, k)

    took_ms = (perf_counter() - started) * 1000

    return SearchResponse(
        hits=[SearchHitResponse.model_validate(hit) for hit in hits],
        took_ms=took_ms,
        collection_version=store.collection_name,
    )
