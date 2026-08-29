from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.search.embedder import Embedder


def test_embed_query_uses_e5_query_prefix() -> None:
    fake_embedding = np.array([0.1, 0.2, 0.3])

    with patch("app.search.embedder.SentenceTransformer") as model_class:
        model = MagicMock()
        model.encode.return_value = fake_embedding
        model_class.return_value = model

        embedder = Embedder("intfloat/multilingual-e5-small")
        result = embedder.embed_query("Что гарантирует Конституция?")

    model_class.assert_called_once_with("intfloat/multilingual-e5-small")
    model.encode.assert_called_once_with(
        "query: Что гарантирует Конституция?",
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    assert result == [0.1, 0.2, 0.3]


def test_embed_passages_uses_e5_passage_prefix() -> None:
    fake_embeddings = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )

    with patch("app.search.embedder.SentenceTransformer") as model_class:
        model = MagicMock()
        model.encode.return_value = fake_embeddings
        model_class.return_value = model

        embedder = Embedder("intfloat/multilingual-e5-small")
        result = embedder.embed_passages(
            [
                "Первый текст.",
                "Второй текст.",
            ]
        )

    model.encode.assert_called_once_with(
        [
            "passage: Первый текст.",
            "passage: Второй текст.",
        ],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    assert result == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]


def test_embed_query_does_not_use_e5_prefix_for_non_e5_model() -> None:
    fake_embedding = np.array([0.1, 0.2, 0.3])

    with patch("app.search.embedder.SentenceTransformer") as model_class:
        model = MagicMock()
        model.encode.return_value = fake_embedding
        model_class.return_value = model

        embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
        result = embedder.embed_query("Что гарантирует Конституция?")

    model.encode.assert_called_once_with(
        "Что гарантирует Конституция?",
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    assert result == [0.1, 0.2, 0.3]


def test_embed_passages_does_not_use_e5_prefix_for_non_e5_model() -> None:
    fake_embeddings = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )

    with patch("app.search.embedder.SentenceTransformer") as model_class:
        model = MagicMock()
        model.encode.return_value = fake_embeddings
        model_class.return_value = model

        embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
        result = embedder.embed_passages(
            [
                "Первый текст.",
                "Второй текст.",
            ]
        )

    model.encode.assert_called_once_with(
        [
            "Первый текст.",
            "Второй текст.",
        ],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    assert result == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]


def test_embed_query_caches_repeated_query() -> None:
    fake_embedding = np.array([0.1, 0.2, 0.3])

    with patch("app.search.embedder.SentenceTransformer") as model_class:
        model = MagicMock()
        model.encode.return_value = fake_embedding
        model_class.return_value = model

        embedder = Embedder("intfloat/multilingual-e5-small")

        first_result = embedder.embed_query("Одинаковый вопрос")
        second_result = embedder.embed_query("Одинаковый вопрос")
        third_result = embedder.embed_query("Другой вопрос")

    assert model.encode.call_count == 2

    model.encode.assert_any_call(
        "query: Одинаковый вопрос",
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    model.encode.assert_any_call(
        "query: Другой вопрос",
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    assert first_result == [0.1, 0.2, 0.3]
    assert second_result == [0.1, 0.2, 0.3]
    assert third_result == [0.1, 0.2, 0.3]


def test_dim_returns_model_embedding_dimension() -> None:
    with patch("app.search.embedder.SentenceTransformer") as model_class:
        model = MagicMock()
        model.get_embedding_dimension.return_value = 384
        model_class.return_value = model

        embedder = Embedder("intfloat/multilingual-e5-small")

        result = embedder.dim

    model.get_embedding_dimension.assert_called_once_with()
    assert result == 384


def test_embed_query_returns_independent_list_from_cache() -> None:
    fake_embedding = np.array([0.1, 0.2, 0.3])

    with patch("app.search.embedder.SentenceTransformer") as model_class:
        model = MagicMock()
        model.encode.return_value = fake_embedding
        model_class.return_value = model

        embedder = Embedder("intfloat/multilingual-e5-small")

        first_result = embedder.embed_query("Одинаковый вопрос")
        first_result[0] = 999.0

        second_result = embedder.embed_query("Одинаковый вопрос")

    model.encode.assert_called_once()
    assert second_result == [0.1, 0.2, 0.3]


def test_dim_raises_when_model_does_not_expose_dimension() -> None:
    with patch("app.search.embedder.SentenceTransformer") as model_class:
        model = MagicMock()
        model.get_embedding_dimension.return_value = None
        model_class.return_value = model

        embedder = Embedder("intfloat/multilingual-e5-small")

        with pytest.raises(
            ValueError,
            match="Embedding model does not expose its dimension",
        ):
            _ = embedder.dim
