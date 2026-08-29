from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._uses_e5_prefixes = self._is_e5_model(model_name)

    @staticmethod
    def _is_e5_model(model_name: str) -> bool:
        model_id = model_name.rsplit("/", maxsplit=1)[-1].lower()

        return model_id.startswith(("e5-", "multilingual-e5-"))

    @property
    def dim(self) -> int:
        dimension = self.model.get_embedding_dimension()

        if dimension is None:
            raise ValueError("Embedding model does not expose its dimension")

        return dimension

    def embed_query(self, text: str) -> list[float]:
        return list(self._embed_query_cached(text))

    @lru_cache(maxsize=256)
    def _embed_query_cached(self, text: str) -> tuple[float, ...]:
        input_text = f"query: {text}" if self._uses_e5_prefixes else text

        embedding: np.ndarray = self.model.encode(  # pyright: ignore[reportUnknownMemberType]
            input_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return tuple(embedding.tolist())

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if self._uses_e5_prefixes:
            input_texts = [f"passage: {text}" for text in texts]
        else:
            input_texts = texts

        embeddings: np.ndarray = self.model.encode(  # pyright: ignore[reportUnknownMemberType]
            input_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()
