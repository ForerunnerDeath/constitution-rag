from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb import Metadata, Where

from app.ingest.chunker import Chunk


@dataclass(frozen=True)
class Hit:
    id: str
    quote: str
    ref: str
    article: str | None
    part: int | None
    part_label: str | None
    score: float


def _chunk_to_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "ref": chunk.ref,
        "kind": chunk.kind,
    }

    if chunk.chapter is not None:
        metadata["chapter"] = chunk.chapter

    if chunk.chapter_title is not None:
        metadata["chapter_title"] = chunk.chapter_title

    if chunk.article is not None:
        metadata["article"] = chunk.article

    if chunk.part is not None:
        metadata["part"] = chunk.part

    if chunk.part_label is not None:
        metadata["part_label"] = chunk.part_label

    return metadata


def _build_hit(
    *,
    chunk_id: str,
    quote: str,
    metadata: Metadata,
    distance: float,
) -> Hit:
    ref = metadata.get("ref")
    if not isinstance(ref, str):
        raise ValueError("Chroma metadata must contain string field 'ref'")

    article_value = metadata.get("article")
    article = article_value if isinstance(article_value, str) else None

    part_value = metadata.get("part")
    part = (
        part_value
        if isinstance(part_value, int) and not isinstance(part_value, bool)
        else None
    )

    part_label_value = metadata.get("part_label")
    part_label = part_label_value if isinstance(part_label_value, str) else None

    return Hit(
        id=chunk_id,
        quote=quote,
        ref=ref,
        article=article,
        part=part,
        part_label=part_label,
        score=1.0 - distance,
    )


_UPSERT_BATCH_SIZE = 512


class ChromaStore:
    def __init__(self, path: Path, collection_name: str) -> None:
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection_name = collection_name

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        for start in range(0, len(chunks), _UPSERT_BATCH_SIZE):
            end = start + _UPSERT_BATCH_SIZE

            batch_chunks = chunks[start:end]
            batch_vectors = vectors[start:end]

            self.collection.upsert(
                ids=[chunk.id for chunk in batch_chunks],
                documents=[chunk.quote for chunk in batch_chunks],
                metadatas=[_chunk_to_metadata(chunk) for chunk in batch_chunks],
                embeddings=np.asarray(batch_vectors, dtype=np.float32),
            )

    def search(
        self, vector: list[float], k: int, where: Where | None = None
    ) -> list[Hit]:
        if k <= 0:
            raise ValueError("k must be greater than zero")

        result = self.collection.query(
            query_embeddings=np.asarray([vector], dtype=np.float32),
            n_results=k,
            where=where,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = result["documents"]
        metadatas = result["metadatas"]
        distances = result["distances"]

        if documents is None or metadatas is None or distances is None:
            raise RuntimeError("Chroma query did not return required fields")

        ids = result["ids"][0]
        quotes = documents[0]
        metadata_items = metadatas[0]
        distance_items = distances[0]

        hits: list[Hit] = []

        for chunk_id, quote, metadata, distance in zip(
            ids,
            quotes,
            metadata_items,
            distance_items,
        ):
            hits.append(
                _build_hit(
                    chunk_id=chunk_id,
                    quote=quote,
                    metadata=metadata,
                    distance=distance,
                )
            )

        return hits

    def get_by_article(self, article: str) -> list[Hit]:
        result = self.collection.get(
            where={"article": article}, include=["documents", "metadatas"]
        )

        documents = result["documents"]
        metadatas = result["metadatas"]

        if documents is None or metadatas is None:
            raise RuntimeError("Chroma get did not return required fields")

        hits: list[Hit] = []

        for chunk_id, quote, metadata in zip(
            result["ids"],
            documents,
            metadatas,
        ):
            hits.append(
                _build_hit(
                    chunk_id=chunk_id,
                    quote=quote,
                    metadata=metadata,
                    distance=0.0,
                )
            )

        return hits

    def count(self) -> int:
        return self.collection.count()

    def recreate(self) -> None:
        self.client.delete_collection(name=self.collection_name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
        )
