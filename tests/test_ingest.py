from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from app.ingest.chunker import Chunk
from app.ingest.parser import Unit
from scripts.ingest import main, run_ingest


def test_run_ingest_embeds_chunk_embed_texts() -> None:
    unit = Unit(
        section="первый",
        chapter=1,
        chapter_title="ТЕСТОВАЯ ГЛАВА",
        article="1",
        part=1,
        part_label="1",
        text="Исходный текст.",
        kind="article",
    )

    chunks = [
        Chunk(
            id="art-1-p-1",
            embed_text="Структурный контекст. Первый текст.",
            quote="Первый текст.",
            ref="Статья 1, часть 1",
            chapter=1,
            chapter_title="ТЕСТОВАЯ ГЛАВА",
            article="1",
            part=1,
            part_label="1",
            kind="article",
        ),
        Chunk(
            id="art-1-p-2",
            embed_text="Структурный контекст. Второй текст.",
            quote="Второй текст.",
            ref="Статья 1, часть 2",
            chapter=1,
            chapter_title="ТЕСТОВАЯ ГЛАВА",
            article="1",
            part=2,
            part_label="2",
            kind="article",
        ),
    ]

    fake_vectors = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    with (
        patch(
            "scripts.ingest.load_text",
            return_value="Нормализованный текст.",
        ) as load_text,
        patch(
            "scripts.ingest.parse_text",
            return_value=[unit],
        ) as parse_text,
        patch(
            "scripts.ingest.chunk_units",
            return_value=chunks,
        ) as chunk_units,
        patch("scripts.ingest.Embedder") as embedder_class,
        patch("scripts.ingest.ChromaStore") as store_class,
    ):
        embedder = MagicMock()
        embedder.model_name = "test-model"
        embedder.dim = 3
        embedder.embed_passages.return_value = fake_vectors
        embedder_class.return_value = embedder

        store = MagicMock()
        store.count.return_value = 2
        store_class.return_value = store

        result = run_ingest(
            source_path=Path("constitution.txt"),
            chroma_path=Path("chroma"),
            collection_name="test-collection",
            embedding_model="test-model",
        )

    load_text.assert_called_once_with(Path("constitution.txt"))
    parse_text.assert_called_once_with("Нормализованный текст.")
    chunk_units.assert_called_once_with([unit])

    embedder_class.assert_called_once_with("test-model")

    embedder.embed_passages.assert_called_once_with(
        [
            "Структурный контекст. Первый текст.",
            "Структурный контекст. Второй текст.",
        ]
    )

    assert result == {
        "units": 1,
        "chunks": 2,
        "vectors": 2,
        "stored": 2,
    }

    store_class.assert_called_once_with(
        path=Path("chroma"),
        collection_name="test-collection",
    )

    store.recreate.assert_not_called()

    store.ensure_embedding_compatibility.assert_called_once_with(
        model_name="test-model",
        dim=3,
    )

    store.assert_has_calls(
        [
            call.ensure_embedding_compatibility(
                model_name="test-model",
                dim=3,
            ),
            call.upsert(
                chunks,
                fake_vectors,
            ),
        ]
    )

    store.upsert.assert_called_once_with(
        chunks,
        fake_vectors,
    )

    store.count.assert_called_once_with()


def test_run_ingest_recreates_collection_when_requested() -> None:
    with (
        patch(
            "scripts.ingest.load_text",
            return_value="Текст.",
        ),
        patch(
            "scripts.ingest.parse_text",
            return_value=[],
        ),
        patch(
            "scripts.ingest.chunk_units",
            return_value=[],
        ),
        patch("scripts.ingest.Embedder") as embedder_class,
        patch("scripts.ingest.ChromaStore") as store_class,
    ):
        embedder = MagicMock()
        embedder.model_name = "test-model"
        embedder.dim = 3
        embedder.embed_passages.return_value = []
        embedder_class.return_value = embedder

        store = MagicMock()
        store.count.return_value = 0
        store_class.return_value = store

        run_ingest(
            source_path=Path("constitution.txt"),
            chroma_path=Path("chroma"),
            collection_name="test-collection",
            embedding_model="test-model",
            recreate=True,
        )

    store.recreate.assert_called_once_with()

    store.ensure_embedding_compatibility.assert_called_once_with(
        model_name="test-model",
        dim=3,
    )

    store.upsert.assert_called_once_with([], [])

    store.assert_has_calls(
        [
            call.recreate(),
            call.ensure_embedding_compatibility(
                model_name="test-model",
                dim=3,
            ),
            call.upsert([], []),
        ]
    )


def test_main_runs_ingest_with_settings(capsys: pytest.CaptureFixture[str]) -> None:
    settings = SimpleNamespace(
        source_path=Path("constitution.txt"),
        chroma_path=Path("chroma"),
        chroma_collection="test-collection",
        embedding_model="test-model",
    )

    with (
        patch(
            "scripts.ingest.get_settings",
            return_value=settings,
        ),
        patch(
            "scripts.ingest.run_ingest",
            return_value={
                "units": 384,
                "chunks": 383,
                "vectors": 383,
                "stored": 383,
            },
        ) as run_ingest_mock,
    ):
        main([])

    run_ingest_mock.assert_called_once_with(
        source_path=Path("constitution.txt"),
        chroma_path=Path("chroma"),
        collection_name="test-collection",
        embedding_model="test-model",
        recreate=False,
    )

    output = capsys.readouterr().out

    assert "Units: 384" in output
    assert "Chunks: 383" in output
    assert "Vectors: 383" in output
    assert "Stored: 383" in output


def test_main_passes_recreate_flag() -> None:
    settings = SimpleNamespace(
        source_path=Path("constitution.txt"),
        chroma_path=Path("chroma"),
        chroma_collection="test-collection",
        embedding_model="test-model",
    )

    with (
        patch(
            "scripts.ingest.get_settings",
            return_value=settings,
        ),
        patch(
            "scripts.ingest.run_ingest",
            return_value={
                "units": 384,
                "chunks": 383,
                "vectors": 383,
                "stored": 383,
            },
        ) as run_ingest_mock,
    ):
        main(["--recreate"])

    run_ingest_mock.assert_called_once_with(
        source_path=Path("constitution.txt"),
        chroma_path=Path("chroma"),
        collection_name="test-collection",
        embedding_model="test-model",
        recreate=True,
    )
