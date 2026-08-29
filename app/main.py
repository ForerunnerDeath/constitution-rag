from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.concurrency import run_in_threadpool

from app.config import get_settings
from app.deps import get_rag_service, get_retriever, get_store
from app.http_middleware import request_context_middleware
from app.input_sanitizer import sanitize_question
from app.llm.client import OpenAICompatibleLLMClient
from app.llm.rag import PROMPT_VERSION, RAGService
from app.observability import log_event
from app.rate_limit import RateLimiter
from app.schemas import (
    ArticleChunkResponse,
    ArticleResponse,
    AskRequest,
    AskResponse,
    CitationResponse,
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

    app.state.rate_limiter = RateLimiter(max_requests=settings.rate_limit_per_minute)

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

    llm_client = None

    if settings.llm_enabled:
        llm_client = OpenAICompatibleLLMClient(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    app.state.llm_client = llm_client

    app.state.rag_service = RAGService(
        retriever=app.state.retriever,
        llm_client=llm_client,
    )

    try:
        yield
    finally:
        if llm_client is not None:
            await llm_client.close()


app = FastAPI(
    title="Constitution RAG",
    version="0.1.0",
    lifespan=lifespan,
)
app.middleware("http")(request_context_middleware)


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
    request: Request,
    q: Annotated[str, Query(min_length=3, max_length=500)],
    k: Annotated[int, Query(ge=1, le=20)] = 5,
    use_hybrid: bool = False,
    retriever: Retriever = Depends(get_retriever),
) -> SearchResponse:
    started = perf_counter()
    sanitized_q = sanitize_question(q)

    if len(sanitized_q) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Question is too short after sanitization",
        )

    retrieval = await run_in_threadpool(
        retriever.retrieve_with_metrics,
        sanitized_q,
        k,
        use_hybrid,
    )

    took_ms = (perf_counter() - started) * 1000

    log_event(
        "search_request",
        request_id=request.state.request_id,
        question=sanitized_q,
        k=k,
        hits=[
            {
                "id": hit.id,
                "score": hit.score,
            }
            for hit in retrieval.hits
        ],
        prompt_version=None,
        llm_called=False,
        embed_ms=round(retrieval.embed_ms, 3),
        search_ms=round(retrieval.search_ms, 3),
        llm_ms=None,
        refused=not retrieval.hits,
    )

    return SearchResponse(
        hits=[SearchHitResponse.model_validate(hit) for hit in retrieval.hits],
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


@app.post("/ask", response_model=AskResponse)
async def ask(
    request: Request,
    payload: AskRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    sanitized_question = sanitize_question(payload.question)

    if len(sanitized_question) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Question is too short after sanitization",
        )

    result = await rag_service.ask(
        sanitized_question,
        payload.k,
    )

    log_event(
        "rag_request",
        request_id=request.state.request_id,
        question=sanitized_question,
        k=payload.k,
        hits=[
            {
                "id": hit.id,
                "score": hit.score,
            }
            for hit in result.hits
        ],
        prompt_version=PROMPT_VERSION,
        llm_called=result.llm_called,
        embed_ms=round(result.embed_ms, 3),
        search_ms=round(result.search_ms, 3),
        llm_ms=(round(result.llm_ms, 3) if result.llm_ms is not None else None),
        refused=not result.found,
    )

    return AskResponse(
        found=result.found,
        answer=result.answer,
        message=result.message,
        citations=[CitationResponse.model_validate(hit) for hit in result.hits],
        llm_used=result.llm_used,
    )
