from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.evaluate import build_components


def test_build_components_checks_embedding_compatibility() -> None:
    settings = SimpleNamespace(
        embedding_model="intfloat/multilingual-e5-small",
        chroma_path="data/chroma",
        chroma_collection="constitution_e5_small",
    )

    fake_embedder = MagicMock()
    fake_embedder.model_name = "intfloat/multilingual-e5-small"
    fake_embedder.dim = 384

    fake_store = MagicMock()
    fake_corpus = [MagicMock()]
    fake_store.get_all.return_value = fake_corpus

    fake_lexical_index = MagicMock()
    fake_retriever = MagicMock()

    with (
        patch(
            "scripts.evaluate.get_settings",
            return_value=settings,
        ),
        patch(
            "scripts.evaluate.Embedder",
            return_value=fake_embedder,
        ) as embedder_class,
        patch(
            "scripts.evaluate.ChromaStore",
            return_value=fake_store,
        ) as store_class,
        patch(
            "scripts.evaluate.LexicalIndex",
            return_value=fake_lexical_index,
        ) as lexical_index_class,
        patch(
            "scripts.evaluate.Retriever",
            return_value=fake_retriever,
        ) as retriever_class,
    ):
        retriever, embedder, store = build_components(min_score=0.833)

    embedder_class.assert_called_once_with("intfloat/multilingual-e5-small")

    store_class.assert_called_once_with(
        path="data/chroma",
        collection_name="constitution_e5_small",
    )

    fake_store.ensure_embedding_compatibility.assert_called_once_with(
        model_name="intfloat/multilingual-e5-small",
        dim=384,
    )

    fake_store.get_all.assert_called_once_with()
    lexical_index_class.assert_called_once_with(fake_corpus)

    retriever_class.assert_called_once_with(
        embedder=fake_embedder,
        store=fake_store,
        lexical_index=fake_lexical_index,
        min_score=0.833,
    )

    assert retriever is fake_retriever
    assert embedder is fake_embedder
    assert store is fake_store


def test_build_components_fails_fast_on_embedding_mismatch() -> None:
    settings = SimpleNamespace(
        embedding_model="model-b",
        chroma_path="data/chroma",
        chroma_collection="test-collection",
    )

    fake_embedder = MagicMock()
    fake_embedder.model_name = "model-b"
    fake_embedder.dim = 384

    fake_store = MagicMock()
    fake_store.ensure_embedding_compatibility.side_effect = RuntimeError(
        "Embedding model mismatch"
    )

    with (
        patch(
            "scripts.evaluate.get_settings",
            return_value=settings,
        ),
        patch(
            "scripts.evaluate.Embedder",
            return_value=fake_embedder,
        ),
        patch(
            "scripts.evaluate.ChromaStore",
            return_value=fake_store,
        ),
        patch("scripts.evaluate.LexicalIndex") as lexical_index_class,
        patch("scripts.evaluate.Retriever") as retriever_class,
    ):
        with pytest.raises(
            RuntimeError,
            match="Embedding model mismatch",
        ):
            build_components(min_score=0.833)

    fake_store.ensure_embedding_compatibility.assert_called_once_with(
        model_name="model-b",
        dim=384,
    )

    fake_store.get_all.assert_not_called()
    lexical_index_class.assert_not_called()
    retriever_class.assert_not_called()
