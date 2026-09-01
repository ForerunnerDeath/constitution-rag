from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb import Metadata, Where

from app.ingest.chunker import Chunk
from app.ingest.loader import calculate_checksum


@dataclass(frozen=True)
class Hit:
    id: str
    quote: str
    ref: str
    article: str | None
    part: int | None
    part_label: str | None
    score: float | None


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
    distance: float | None,
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

    score = None if distance is None else 1.0 - distance

    return Hit(
        id=chunk_id,
        quote=quote,
        ref=ref,
        article=article,
        part=part,
        part_label=part_label,
        score=score,
    )


_UPSERT_BATCH_SIZE = 512


def _part_order(part_label: str | None) -> tuple[int, ...]:
    if part_label is None:
        return ()

    start_label = part_label.split("-", 1)[0]

    try:
        return tuple(int(value) for value in start_label.split("."))
    except ValueError as exc:
        raise ValueError(f"Invalid part label: {part_label}") from exc


def _fragment_order(chunk_id: str) -> int:
    marker = "-c-"

    if marker not in chunk_id:
        return 0

    fragment = chunk_id.rsplit(marker, 1)[1]

    if not fragment.isdigit():
        raise ValueError(f"Invalid chunk fragment id: {chunk_id}")

    return int(fragment)


def _article_hit_sort_key(hit: Hit) -> tuple[bool, tuple[int, ...], int, str]:
    return (
        hit.part_label is not None,
        _part_order(hit.part_label),
        _fragment_order(hit.id),
        hit.id,
    )


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
                    distance=None,
                )
            )

        hits.sort(key=_article_hit_sort_key)
        return hits

    def count(self) -> int:
        return self.collection.count()

    def get_ids(self) -> set[str]:
        result = self.collection.get(include=[])

        return set(result["ids"])

    def get_corpus_checksum(self) -> str | None:
        metadata = dict(self.collection.metadata or {})
        checksum = metadata.get("corpus_checksum")

        if checksum is None:
            return None

        if not isinstance(checksum, str):
            raise RuntimeError("Chroma collection has invalid corpus checksum metadata")

        return checksum

    def set_corpus_checksum(self, checksum: str) -> None:
        metadata = dict(self.collection.metadata or {})
        metadata["corpus_checksum"] = checksum

        self.collection.modify(metadata=metadata)

    def get_index_revision(self) -> str | None:
        metadata = dict(self.collection.metadata or {})
        revision = metadata.get("index_revision")

        if revision is None:
            return None

        if not isinstance(revision, str):
            raise RuntimeError("Chroma collection has invalid index revision metadata")

        return revision

    def set_index_revision(self, revision: str) -> None:
        metadata = dict(self.collection.metadata or {})
        metadata["index_revision"] = revision

        self.collection.modify(metadata=metadata)

    def clear_corpus_checksum(self) -> None:
        metadata = dict(self.collection.metadata or {})
        metadata.pop("corpus_checksum", None)

        self.collection.modify(metadata=metadata)

    def clear_index_revision(self) -> None:
        metadata = dict(self.collection.metadata or {})
        metadata.pop("index_revision", None)

        self.collection.modify(metadata=metadata)

    def ensure_corpus_compatibility(self, source_path: Path) -> None:
        if self.count() == 0:
            return

        stored_checksum = self.get_corpus_checksum()

        if stored_checksum is None:
            raise RuntimeError(
                "Chroma collection has no corpus checksum metadata; "
                "run ingest to rebuild the index"
            )

        source_checksum = calculate_checksum(source_path)

        if stored_checksum != source_checksum:
            raise RuntimeError(
                "corpus checksum mismatch: source corpus differs from indexed corpus; "
                "run ingest to rebuild the index"
            )

    def ensure_index_revision(self) -> None:
        if self.count() == 0:
            return

        if self.get_index_revision() is None:
            raise RuntimeError(
                "Chroma collection has no index revision metadata; "
                "run ingest to rebuild the index"
            )

    def delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return

        self.collection.delete(ids=ids)

    def ensure_embedding_compatibility(self, *, model_name: str, dim: int) -> None:
        stored = self.count()
        metadata = dict(self.collection.metadata or {})

        stored_model = metadata.get("embedding_model")
        stored_dim = metadata.get("embedding_dim")

        if stored == 0:
            metadata["embedding_model"] = model_name
            metadata["embedding_dim"] = dim

            self.collection.modify(metadata=metadata)
            return

        if not isinstance(stored_model, str) or not (
            isinstance(stored_dim, int) and not isinstance(stored_dim, bool)
        ):
            raise RuntimeError(
                "Chroma collection has no embedding provenance metadata; "
                "recreate the collection"
            )

        if stored_model != model_name:
            raise RuntimeError(
                "Embedding model mismatch: "
                f"index uses {stored_model!r}, runtime uses {model_name!r}"
            )

        if stored_dim != dim:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"index uses {stored_dim}, runtime uses {dim}"
            )

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

    def get_all(self) -> list[Hit]:
        result = self.collection.get(
            include=[
                "documents",
                "metadatas",
            ]
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
                    distance=None,
                )
            )

        hits.sort(key=lambda hit: hit.id)

        return hits
