from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.ingest.chunker import Chunk
from app.search.store import (
    ChromaStore,
    _build_hit,
    _chunk_to_metadata,
)


def _make_chunk(
    *,
    chunk_id: str,
    article: str = "1",
    part: int | None = 1,
    part_label: str | None = "1",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        embed_text="Глава 1. Статья 1. Тест.",
        quote=f"Текст {chunk_id}.",
        ref=f"Статья {article}",
        chapter=1,
        chapter_title="ТЕСТОВАЯ ГЛАВА",
        article=article,
        part=part,
        part_label=part_label,
        kind="article",
    )


def test_chunk_to_metadata_for_regular_article_part() -> None:
    chunk = Chunk(
        id="art-3-p-1",
        embed_text="Глава 1. Статья 3, часть 1. Текст.",
        quote="Текст.",
        ref="Статья 3, часть 1",
        chapter=1,
        chapter_title="ОСНОВЫ КОНСТИТУЦИОННОГО СТРОЯ",
        article="3",
        part=1,
        part_label="1",
        kind="article",
    )

    metadata = _chunk_to_metadata(chunk)

    assert metadata == {
        "ref": "Статья 3, часть 1",
        "kind": "article",
        "chapter": 1,
        "chapter_title": "ОСНОВЫ КОНСТИТУЦИОННОГО СТРОЯ",
        "article": "3",
        "part": 1,
        "part_label": "1",
    }


def test_chunk_to_metadata_for_decimal_part_omits_numeric_part() -> None:
    chunk = Chunk(
        id="art-67-p-2.1",
        embed_text="Глава 3. Статья 67, часть 2.1. Текст.",
        quote="Текст.",
        ref="Статья 67, часть 2.1",
        chapter=3,
        chapter_title="ФЕДЕРАТИВНОЕ УСТРОЙСТВО",
        article="67",
        part=None,
        part_label="2.1",
        kind="article",
    )

    metadata = _chunk_to_metadata(chunk)

    assert metadata == {
        "ref": "Статья 67, часть 2.1",
        "kind": "article",
        "chapter": 3,
        "chapter_title": "ФЕДЕРАТИВНОЕ УСТРОЙСТВО",
        "article": "67",
        "part_label": "2.1",
    }

    assert "part" not in metadata


def test_chunk_to_metadata_for_preamble_omits_missing_fields() -> None:
    chunk = Chunk(
        id="preamble",
        embed_text="Преамбула Конституции Российской Федерации. Текст.",
        quote="Текст.",
        ref="Преамбула",
        chapter=None,
        chapter_title=None,
        article=None,
        part=None,
        part_label=None,
        kind="preamble",
    )

    metadata = _chunk_to_metadata(chunk)

    assert metadata == {
        "ref": "Преамбула",
        "kind": "preamble",
    }

    assert "chapter" not in metadata
    assert "chapter_title" not in metadata
    assert "article" not in metadata
    assert "part" not in metadata
    assert "part_label" not in metadata


def test_upsert_passes_chunk_data_to_chroma() -> None:
    chunk = _make_chunk(chunk_id="art-1-p-1")
    vector = [0.1, 0.2, 0.3]

    with patch("app.search.store.chromadb.PersistentClient") as client_class:
        client = MagicMock()
        collection = MagicMock()
        client.get_or_create_collection.return_value = collection
        client_class.return_value = client

        store = ChromaStore(
            path=Path("test-chroma"),
            collection_name="test-collection",
        )

        store.upsert([chunk], [vector])

    collection.upsert.assert_called_once()

    kwargs = collection.upsert.call_args.kwargs

    assert kwargs["ids"] == ["art-1-p-1"]
    assert kwargs["documents"] == ["Текст art-1-p-1."]
    assert kwargs["metadatas"] == [
        {
            "ref": "Статья 1",
            "kind": "article",
            "chapter": 1,
            "chapter_title": "ТЕСТОВАЯ ГЛАВА",
            "article": "1",
            "part": 1,
            "part_label": "1",
        }
    ]

    np.testing.assert_array_equal(
        kwargs["embeddings"],
        np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
    )


def test_upsert_rejects_mismatched_chunks_and_vectors() -> None:
    with patch("app.search.store.chromadb.PersistentClient") as client_class:
        client = MagicMock()
        collection = MagicMock()
        client.get_or_create_collection.return_value = collection
        client_class.return_value = client

        store = ChromaStore(
            path=Path("test-chroma"),
            collection_name="test-collection",
        )

        with pytest.raises(ValueError):
            store.upsert(
                [_make_chunk(chunk_id="art-1")],
                [],
            )

    collection.upsert.assert_not_called()


def test_upsert_splits_data_into_batches_of_512() -> None:
    chunks = [_make_chunk(chunk_id=f"art-{index}") for index in range(513)]
    vectors = [[float(index), 0.0, 1.0] for index in range(513)]

    with patch("app.search.store.chromadb.PersistentClient") as client_class:
        client = MagicMock()
        collection = MagicMock()
        client.get_or_create_collection.return_value = collection
        client_class.return_value = client

        store = ChromaStore(
            path=Path("test-chroma"),
            collection_name="test-collection",
        )

        store.upsert(chunks, vectors)

    assert collection.upsert.call_count == 2

    first_call = collection.upsert.call_args_list[0].kwargs
    second_call = collection.upsert.call_args_list[1].kwargs

    assert len(first_call["ids"]) == 512
    assert first_call["ids"][0] == "art-0"
    assert first_call["ids"][-1] == "art-511"

    assert len(second_call["ids"]) == 1
    assert second_call["ids"] == ["art-512"]

    assert first_call["embeddings"].shape == (512, 3)
    assert second_call["embeddings"].shape == (1, 3)


def test_build_hit_converts_cosine_distance_to_score() -> None:
    hit = _build_hit(
        chunk_id="art-28-p-1",
        quote="Текст статьи.",
        metadata={
            "ref": "Статья 28, часть 1",
            "kind": "article",
            "article": "28",
            "part": 1,
            "part_label": "1",
        },
        distance=0.16,
    )

    assert hit.id == "art-28-p-1"
    assert hit.quote == "Текст статьи."
    assert hit.ref == "Статья 28, часть 1"
    assert hit.article == "28"
    assert hit.part == 1
    assert hit.part_label == "1"
    assert hit.score == pytest.approx(0.84)


def test_build_hit_handles_missing_optional_metadata() -> None:
    hit = _build_hit(
        chunk_id="preamble",
        quote="Текст преамбулы.",
        metadata={
            "ref": "Преамбула",
            "kind": "preamble",
        },
        distance=0.25,
    )

    assert hit.article is None
    assert hit.part is None
    assert hit.part_label is None
    assert hit.score == pytest.approx(0.75)


def test_search_queries_chroma_and_returns_hits() -> None:
    with patch("app.search.store.chromadb.PersistentClient") as client_class:
        client = MagicMock()
        collection = MagicMock()
        client.get_or_create_collection.return_value = collection
        client_class.return_value = client

        collection.query.return_value = {
            "ids": [["art-28"]],
            "documents": [["Текст статьи 28."]],
            "metadatas": [
                [
                    {
                        "ref": "Статья 28",
                        "kind": "article",
                        "article": "28",
                    }
                ]
            ],
            "distances": [[0.16]],
        }

        store = ChromaStore(
            path=Path("test-chroma"),
            collection_name="test-collection",
        )

        hits = store.search(
            vector=[0.1, 0.2, 0.3],
            k=3,
        )

    assert len(hits) == 1

    hit = hits[0]

    assert hit.id == "art-28"
    assert hit.quote == "Текст статьи 28."
    assert hit.ref == "Статья 28"
    assert hit.article == "28"
    assert hit.part is None
    assert hit.part_label is None
    assert hit.score == pytest.approx(0.84)

    kwargs = collection.query.call_args.kwargs

    assert kwargs["n_results"] == 3
    assert kwargs["where"] is None
    assert kwargs["include"] == [
        "documents",
        "metadatas",
        "distances",
    ]

    np.testing.assert_array_equal(
        kwargs["query_embeddings"],
        np.array(
            [[0.1, 0.2, 0.3]],
            dtype=np.float32,
        ),
    )


def test_search_passes_where_filter_to_chroma() -> None:
    with patch("app.search.store.chromadb.PersistentClient") as client_class:
        client = MagicMock()
        collection = MagicMock()
        client.get_or_create_collection.return_value = collection
        client_class.return_value = client

        collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        store = ChromaStore(
            path=Path("test-chroma"),
            collection_name="test-collection",
        )

        hits = store.search(
            vector=[0.1, 0.2, 0.3],
            k=5,
            where={"article": "29"},
        )

    assert hits == []

    kwargs = collection.query.call_args.kwargs

    assert kwargs["where"] == {"article": "29"}
    assert kwargs["n_results"] == 5


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_non_positive_k(k: int) -> None:
    with patch("app.search.store.chromadb.PersistentClient") as client_class:
        client = MagicMock()
        collection = MagicMock()
        client.get_or_create_collection.return_value = collection
        client_class.return_value = client

        store = ChromaStore(
            path=Path("test-chroma"),
            collection_name="test-collection",
        )

        with pytest.raises(ValueError):
            store.search(
                vector=[0.1, 0.2, 0.3],
                k=k,
            )

    collection.query.assert_not_called()


def test_search_rejects_missing_required_chroma_fields() -> None:
    with patch("app.search.store.chromadb.PersistentClient") as client_class:
        client = MagicMock()
        collection = MagicMock()
        client.get_or_create_collection.return_value = collection
        client_class.return_value = client

        collection.query.return_value = {
            "ids": [["art-28"]],
            "documents": None,
            "metadatas": None,
            "distances": None,
        }

        store = ChromaStore(
            path=Path("test-chroma"),
            collection_name="test-collection",
        )

        with pytest.raises(RuntimeError):
            store.search(
                vector=[0.1, 0.2, 0.3],
                k=1,
            )


def test_get_by_article_returns_matching_chunks() -> None:
    with patch("app.search.store.chromadb.PersistentClient") as client_class:
        client = MagicMock()
        collection = MagicMock()
        client.get_or_create_collection.return_value = collection
        client_class.return_value = client

        collection.get.return_value = {
            "ids": [
                "art-29-p-1",
                "art-29-p-2",
            ],
            "documents": [
                "Текст первой части.",
                "Текст второй части.",
            ],
            "metadatas": [
                {
                    "ref": "Статья 29, часть 1",
                    "kind": "article",
                    "article": "29",
                    "part": 1,
                    "part_label": "1",
                },
                {
                    "ref": "Статья 29, часть 2",
                    "kind": "article",
                    "article": "29",
                    "part": 2,
                    "part_label": "2",
                },
            ],
        }

        store = ChromaStore(
            path=Path("test-chroma"),
            collection_name="test-collection",
        )

        hits = store.get_by_article("29")

    assert len(hits) == 2

    assert hits[0].id == "art-29-p-1"
    assert hits[0].quote == "Текст первой части."
    assert hits[0].article == "29"
    assert hits[0].part == 1
    assert hits[0].part_label == "1"
    assert hits[0].score is None

    assert hits[1].id == "art-29-p-2"
    assert hits[1].part == 2
    assert hits[1].score is None

    collection.get.assert_called_once_with(
        where={"article": "29"},
        include=[
            "documents",
            "metadatas",
        ],
    )


def test_get_by_article_returns_empty_list_when_article_not_found() -> None:
    with patch("app.search.store.chromadb.PersistentClient") as client_class:
        client = MagicMock()
        collection = MagicMock()
        client.get_or_create_collection.return_value = collection
        client_class.return_value = client

        collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }

        store = ChromaStore(
            path=Path("test-chroma"),
            collection_name="test-collection",
        )

        hits = store.get_by_article("999")

    assert hits == []


def test_get_by_article_rejects_missing_required_chroma_fields() -> None:
    with patch("app.search.store.chromadb.PersistentClient") as client_class:
        client = MagicMock()
        collection = MagicMock()
        client.get_or_create_collection.return_value = collection
        client_class.return_value = client

        collection.get.return_value = {
            "ids": ["art-29"],
            "documents": None,
            "metadatas": None,
        }

        store = ChromaStore(
            path=Path("test-chroma"),
            collection_name="test-collection",
        )

        with pytest.raises(RuntimeError):
            store.get_by_article("29")


def test_count_returns_collection_count() -> None:
    with patch("app.search.store.chromadb.PersistentClient") as client_class:
        client = MagicMock()
        collection = MagicMock()
        collection.count.return_value = 383

        client.get_or_create_collection.return_value = collection
        client_class.return_value = client

        store = ChromaStore(
            path=Path("test-chroma"),
            collection_name="test-collection",
        )

        result = store.count()

    assert result == 383
    collection.count.assert_called_once_with()


def test_recreate_deletes_and_recreates_collection() -> None:
    with patch("app.search.store.chromadb.PersistentClient") as client_class:
        client = MagicMock()

        old_collection = MagicMock()
        new_collection = MagicMock()

        client.get_or_create_collection.side_effect = [
            old_collection,
            new_collection,
        ]
        client_class.return_value = client

        store = ChromaStore(
            path=Path("test-chroma"),
            collection_name="test-collection",
        )

        assert store.collection is old_collection

        store.recreate()

    client.delete_collection.assert_called_once_with(
        name="test-collection",
    )

    assert client.get_or_create_collection.call_count == 2

    recreate_call = client.get_or_create_collection.call_args_list[1]

    assert recreate_call.kwargs == {
        "name": "test-collection",
        "embedding_function": None,
        "configuration": {
            "hnsw": {
                "space": "cosine",
            }
        },
    }

    assert store.collection is new_collection


def test_get_by_article_returns_chunks_in_document_order() -> None:
    with patch("app.search.store.chromadb.PersistentClient") as client_class:
        client = MagicMock()
        collection = MagicMock()
        client.get_or_create_collection.return_value = collection
        client_class.return_value = client

        collection.get.return_value = {
            "ids": [
                "art-81-p-2-c-2",
                "art-81-p-4",
                "art-81-p-1",
                "art-81-p-3-3.1",
                "art-81-p-2-c-1",
            ],
            "documents": [
                "Часть 2, фрагмент 2.",
                "Часть 4.",
                "Часть 1.",
                "Части 3-3.1.",
                "Часть 2, фрагмент 1.",
            ],
            "metadatas": [
                {
                    "ref": "Статья 81, часть 2",
                    "kind": "article",
                    "article": "81",
                    "part": 2,
                    "part_label": "2",
                },
                {
                    "ref": "Статья 81, часть 4",
                    "kind": "article",
                    "article": "81",
                    "part": 4,
                    "part_label": "4",
                },
                {
                    "ref": "Статья 81, часть 1",
                    "kind": "article",
                    "article": "81",
                    "part": 1,
                    "part_label": "1",
                },
                {
                    "ref": "Статья 81, части 3-3.1",
                    "kind": "article",
                    "article": "81",
                    "part_label": "3-3.1",
                },
                {
                    "ref": "Статья 81, часть 2",
                    "kind": "article",
                    "article": "81",
                    "part": 2,
                    "part_label": "2",
                },
            ],
        }

        store = ChromaStore(
            path=Path("test-chroma"),
            collection_name="test-collection",
        )

        hits = store.get_by_article("81")

    assert [hit.id for hit in hits] == [
        "art-81-p-1",
        "art-81-p-2-c-1",
        "art-81-p-2-c-2",
        "art-81-p-3-3.1",
        "art-81-p-4",
    ]


def test_get_all_returns_all_chunks() -> None:
    with patch("app.search.store.chromadb.PersistentClient") as client_class:
        client = MagicMock()
        collection = MagicMock()
        client.get_or_create_collection.return_value = collection
        client_class.return_value = client

        collection.get.return_value = {
            "ids": [
                "art-28",
                "art-3-p-1",
            ],
            "documents": [
                "Свобода совести.",
                "Источником власти является народ.",
            ],
            "metadatas": [
                {
                    "ref": "Статья 28",
                    "kind": "article",
                    "article": "28",
                },
                {
                    "ref": "Статья 3, часть 1",
                    "kind": "article",
                    "article": "3",
                    "part": 1,
                    "part_label": "1",
                },
            ],
        }

        store = ChromaStore(
            path=Path("test-chroma"),
            collection_name="test-collection",
        )

        hits = store.get_all()

    assert [hit.id for hit in hits] == [
        "art-28",
        "art-3-p-1",
    ]

    assert hits[0].quote == "Свобода совести."
    assert hits[1].article == "3"

    collection.get.assert_called_once_with(
        include=[
            "documents",
            "metadatas",
        ]
    )
