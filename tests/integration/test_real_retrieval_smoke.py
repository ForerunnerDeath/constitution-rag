import os
from pathlib import Path

import pytest

from app.ingest.chunker import Chunk
from app.search.embedder import Embedder
from app.search.store import ChromaStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION") != "1",
        reason="Set RUN_INTEGRATION=1 to run real dependency integration tests",
    ),
]


def _chunk(
    *,
    chunk_id: str,
    article: str,
    quote: str,
) -> Chunk:
    ref = f"Статья {article}"

    return Chunk(
        id=chunk_id,
        embed_text=f"{ref}. {quote}",
        quote=quote,
        ref=ref,
        chapter=1,
        chapter_title="Основы конституционного строя",
        article=article,
        part=None,
        part_label=None,
        kind="article",
    )


def test_real_embedder_and_chroma_retrieval(tmp_path: Path) -> None:
    chunks = [
        _chunk(
            chunk_id="art-3",
            article="3",
            quote=(
                "Носителем суверенитета и единственным источником власти "
                "в Российской Федерации является ее многонациональный народ."
            ),
        ),
        _chunk(
            chunk_id="art-80",
            article="80",
            quote="Президент Российской Федерации является главой государства.",
        ),
        _chunk(
            chunk_id="art-110",
            article="110",
            quote=(
                "Исполнительную власть Российской Федерации осуществляет "
                "Правительство Российской Федерации."
            ),
        ),
        _chunk(
            chunk_id="art-118",
            article="118",
            quote=("Правосудие в Российской Федерации осуществляется только судом."),
        ),
    ]

    embedder = Embedder("intfloat/multilingual-e5-small")

    store = ChromaStore(
        path=tmp_path / "chroma",
        collection_name="integration-smoke",
    )

    store.ensure_embedding_compatibility(
        model_name=embedder.model_name,
        dim=embedder.dim,
    )

    vectors = embedder.embed_passages([chunk.embed_text for chunk in chunks])

    store.upsert(chunks, vectors)

    assert store.count() == len(chunks)

    query_vector = embedder.embed_query(
        "Кто является источником власти в Российской Федерации?"
    )

    hits = store.search(
        query_vector,
        k=len(chunks),
    )

    assert len(hits) == len(chunks)

    assert hits[0].id == "art-3"
    assert hits[0].article == "3"
    assert hits[0].score is not None
    assert hits[0].score > 0.0

    scores = [hit.score for hit in hits]

    assert all(score is not None for score in scores)

    numeric_scores = [score for score in scores if score is not None]

    assert numeric_scores == sorted(
        numeric_scores,
        reverse=True,
    )
