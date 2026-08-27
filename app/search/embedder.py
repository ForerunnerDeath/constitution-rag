import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        prefixed_text = f"query: {text}"

        embedding: np.ndarray = self.model.encode(  # pyright: ignore[reportUnknownMemberType]
            prefixed_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed_texts = [f"passage: {text}" for text in texts]

        embeddings: np.ndarray = self.model.encode(  # pyright: ignore[reportUnknownMemberType]
            prefixed_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()
