from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from app.ingest.chunker import Chunk
from app.ingest.parser import Unit
from app.search.store import ChromaStore
from scripts.ingest import build_index_revision, main, run_ingest


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
        patch(
            "scripts.ingest.calculate_checksum",
            return_value="a" * 64,
        ),
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
            call.clear_corpus_checksum(),
            call.clear_index_revision(),
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
        patch(
            "scripts.ingest.calculate_checksum",
            return_value="a" * 64,
        ),
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
            call.clear_corpus_checksum(),
            call.clear_index_revision(),
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


def test_run_ingest_removes_chunks_missing_from_new_snapshot(
    tmp_path: Path,
) -> None:
    first_chunks = [
        Chunk(
            id="art-1-p-1",
            embed_text="Статья 1, часть 1. Первый текст.",
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
            embed_text="Статья 1, часть 2. Устаревший текст.",
            quote="Устаревший текст.",
            ref="Статья 1, часть 2",
            chapter=1,
            chapter_title="ТЕСТОВАЯ ГЛАВА",
            article="1",
            part=2,
            part_label="2",
            kind="article",
        ),
    ]

    second_chunks = [
        first_chunks[0],
        Chunk(
            id="art-1-p-3",
            embed_text="Статья 1, часть 3. Новый текст.",
            quote="Новый текст.",
            ref="Статья 1, часть 3",
            chapter=1,
            chapter_title="ТЕСТОВАЯ ГЛАВА",
            article="1",
            part=3,
            part_label="3",
            kind="article",
        ),
    ]

    chroma_path = tmp_path / "chroma"
    collection_name = "test-ingest-stale-chunks"

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
            side_effect=[
                first_chunks,
                second_chunks,
            ],
        ),
        patch("scripts.ingest.Embedder") as embedder_class,
        patch(
            "scripts.ingest.calculate_checksum",
            return_value="a" * 64,
        ),
    ):
        embedder = MagicMock()
        embedder.model_name = "test-model"
        embedder.dim = 3
        embedder.embed_passages.side_effect = [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ]
        embedder_class.return_value = embedder

        run_ingest(
            source_path=Path("constitution.txt"),
            chroma_path=chroma_path,
            collection_name=collection_name,
            embedding_model="test-model",
        )

        run_ingest(
            source_path=Path("constitution.txt"),
            chroma_path=chroma_path,
            collection_name=collection_name,
            embedding_model="test-model",
        )

    store = ChromaStore(
        path=chroma_path,
        collection_name=collection_name,
    )

    assert {hit.id for hit in store.get_all()} == {
        "art-1-p-1",
        "art-1-p-3",
    }


def test_main_warns_when_stored_count_differs_from_chunks(
    capsys: pytest.CaptureFixture[str],
) -> None:
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
                "stored": 384,
            },
        ),
    ):
        main([])

    output = capsys.readouterr().out

    assert "WARNING" in output
    assert "384 != 383" in output


def test_run_ingest_updates_index_metadata_after_success(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "constitution.txt"
    source_path.write_text(
        "Новая редакция Конституции.",
        encoding="utf-8",
    )

    with (
        patch("scripts.ingest.parse_text", return_value=[]),
        patch("scripts.ingest.chunk_units", return_value=[]),
        patch("scripts.ingest.Embedder") as embedder_class,
        patch("scripts.ingest.ChromaStore") as store_class,
    ):
        embedder = MagicMock()
        embedder.model_name = "test-model"
        embedder.dim = 3
        embedder.embed_passages.return_value = []
        embedder_class.return_value = embedder

        store = MagicMock()
        store.get_ids.return_value = set()
        store.count.return_value = 0
        store_class.return_value = store

        run_ingest(
            source_path=source_path,
            chroma_path=tmp_path / "chroma",
            collection_name="test-collection",
            embedding_model="test-model",
        )

    expected_checksum = sha256(source_path.read_bytes()).hexdigest()

    expected_revision = build_index_revision(
        corpus_checksum=expected_checksum,
        embedding_model="test-model",
        embedding_dim=3,
        chunks=[],
    )

    store.set_index_revision.assert_called_once_with(expected_revision)

    store.set_corpus_checksum.assert_called_once_with(expected_checksum)

    store.assert_has_calls(
        [
            call.set_index_revision(expected_revision),
            call.set_corpus_checksum(expected_checksum),
        ]
    )


def test_run_ingest_does_not_update_corpus_checksum_when_ingest_fails(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "constitution.txt"
    source_path.write_text(
        "Новая редакция Конституции.",
        encoding="utf-8",
    )

    with (
        patch("scripts.ingest.parse_text", return_value=[]),
        patch("scripts.ingest.chunk_units", return_value=[]),
        patch("scripts.ingest.Embedder") as embedder_class,
        patch("scripts.ingest.ChromaStore") as store_class,
    ):
        embedder = MagicMock()
        embedder.model_name = "test-model"
        embedder.dim = 3
        embedder.embed_passages.side_effect = RuntimeError("embedding failed")
        embedder_class.return_value = embedder

        store = MagicMock()
        store_class.return_value = store

        with pytest.raises(RuntimeError, match="embedding failed"):
            run_ingest(
                source_path=source_path,
                chroma_path=tmp_path / "chroma",
                collection_name="test-collection",
                embedding_model="test-model",
            )

    store.set_corpus_checksum.assert_not_called()


def test_run_ingest_does_not_update_corpus_checksum_when_stored_count_mismatches(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "constitution.txt"
    source_path.write_text(
        "Новая редакция Конституции.",
        encoding="utf-8",
    )

    chunk = Chunk(
        id="art-1-p-1",
        embed_text="Статья 1, часть 1. Текст.",
        quote="Текст.",
        ref="Статья 1, часть 1",
        chapter=1,
        chapter_title="ТЕСТОВАЯ ГЛАВА",
        article="1",
        part=1,
        part_label="1",
        kind="article",
    )

    with (
        patch("scripts.ingest.parse_text", return_value=[]),
        patch("scripts.ingest.chunk_units", return_value=[chunk]),
        patch("scripts.ingest.Embedder") as embedder_class,
        patch("scripts.ingest.ChromaStore") as store_class,
    ):
        embedder = MagicMock()
        embedder.model_name = "test-model"
        embedder.dim = 3
        embedder.embed_passages.return_value = [[1.0, 0.0, 0.0]]
        embedder_class.return_value = embedder

        store = MagicMock()
        store.get_ids.return_value = {chunk.id}
        store.count.return_value = 2
        store_class.return_value = store

        stats = run_ingest(
            source_path=source_path,
            chroma_path=tmp_path / "chroma",
            collection_name="test-collection",
            embedding_model="test-model",
        )

    assert stats["chunks"] == 1
    assert stats["stored"] == 2
    store.set_corpus_checksum.assert_not_called()


def test_run_ingest_clears_index_metadata_before_upsert(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "constitution.txt"
    source_path.write_text(
        "Новая редакция Конституции.",
        encoding="utf-8",
    )

    chunk = Chunk(
        id="art-1-p-1",
        embed_text="Статья 1, часть 1. Текст.",
        quote="Текст.",
        ref="Статья 1, часть 1",
        chapter=1,
        chapter_title="ТЕСТОВАЯ ГЛАВА",
        article="1",
        part=1,
        part_label="1",
        kind="article",
    )

    with (
        patch("scripts.ingest.parse_text", return_value=[]),
        patch("scripts.ingest.chunk_units", return_value=[chunk]),
        patch("scripts.ingest.Embedder") as embedder_class,
        patch("scripts.ingest.ChromaStore") as store_class,
    ):
        embedder = MagicMock()
        embedder.model_name = "test-model"
        embedder.dim = 3
        embedder.embed_passages.return_value = [
            [1.0, 0.0, 0.0],
        ]
        embedder_class.return_value = embedder

        store = MagicMock()
        store.upsert.side_effect = RuntimeError("upsert failed")
        store_class.return_value = store

        with pytest.raises(RuntimeError, match="upsert failed"):
            run_ingest(
                source_path=source_path,
                chroma_path=tmp_path / "chroma",
                collection_name="test-collection",
                embedding_model="test-model",
            )

    store.assert_has_calls(
        [
            call.ensure_embedding_compatibility(
                model_name="test-model",
                dim=3,
            ),
            call.clear_corpus_checksum(),
            call.clear_index_revision(),
            call.upsert(
                [chunk],
                [[1.0, 0.0, 0.0]],
            ),
        ]
    )
    store.set_index_revision.assert_not_called()
    store.set_corpus_checksum.assert_not_called()


def test_build_index_revision_changes_when_index_inputs_change() -> None:
    chunk = Chunk(
        id="art-1-p-1",
        embed_text="Статья 1, часть 1. Текст.",
        quote="Текст.",
        ref="Статья 1, часть 1",
        chapter=1,
        chapter_title="ТЕСТОВАЯ ГЛАВА",
        article="1",
        part=1,
        part_label="1",
        kind="article",
    )

    revision = build_index_revision(
        corpus_checksum="a" * 64,
        embedding_model="test-model",
        embedding_dim=3,
        chunks=[chunk],
    )

    changed_model_revision = build_index_revision(
        corpus_checksum="a" * 64,
        embedding_model="other-model",
        embedding_dim=3,
        chunks=[chunk],
    )

    changed_chunk_revision = build_index_revision(
        corpus_checksum="a" * 64,
        embedding_model="test-model",
        embedding_dim=3,
        chunks=[
            Chunk(
                id=chunk.id,
                embed_text="Статья 1, часть 1. Изменённый текст.",
                quote="Изменённый текст.",
                ref=chunk.ref,
                chapter=chunk.chapter,
                chapter_title=chunk.chapter_title,
                article=chunk.article,
                part=chunk.part,
                part_label=chunk.part_label,
                kind=chunk.kind,
            )
        ],
    )

    assert len(revision) == 64
    assert revision != changed_model_revision
    assert revision != changed_chunk_revision
