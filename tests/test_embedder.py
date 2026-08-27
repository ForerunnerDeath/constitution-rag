from unittest.mock import MagicMock, patch

import numpy as np

from app.search.embedder import Embedder


def test_embed_query_uses_e5_query_prefix() -> None:
    fake_embedding = np.array([0.1, 0.2, 0.3])

    with patch("app.search.embedder.SentenceTransformer") as model_class:
        model = MagicMock()
        model.encode.return_value = fake_embedding
        model_class.return_value = model

        embedder = Embedder("test-model")
        result = embedder.embed_query("Что гарантирует Конституция?")

    model_class.assert_called_once_with("test-model")
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

        embedder = Embedder("test-model")
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
