import argparse
import json
from hashlib import sha256
from pathlib import Path

from app.config import get_settings
from app.ingest.chunker import Chunk, chunk_units
from app.ingest.loader import calculate_checksum, load_text
from app.ingest.parser import parse_text
from app.search.embedder import Embedder
from app.search.store import ChromaStore


def build_index_revision(
    *,
    corpus_checksum: str,
    embedding_model: str,
    embedding_dim: int,
    chunks: list[Chunk],
) -> str:
    payload = {
        "corpus_checksum": corpus_checksum,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "chunks": [
            {
                "id": chunk.id,
                "embed_text": chunk.embed_text,
                "quote": chunk.quote,
                "ref": chunk.ref,
                "chapter": chunk.chapter,
                "chapter_title": chunk.chapter_title,
                "article": chunk.article,
                "part": chunk.part,
                "part_label": chunk.part_label,
                "kind": chunk.kind,
            }
            for chunk in sorted(chunks, key=lambda item: item.id)
        ],
    }

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return sha256(serialized.encode("utf-8")).hexdigest()


def run_ingest(
    *,
    source_path: Path,
    chroma_path: Path,
    collection_name: str,
    embedding_model: str,
    recreate: bool = False,
) -> dict[str, int]:
    source_checksum = calculate_checksum(source_path)
    text = load_text(source_path)
    units = parse_text(text)
    chunks = chunk_units(units)

    embedder = Embedder(embedding_model)

    index_revision = build_index_revision(
        corpus_checksum=source_checksum,
        embedding_model=embedder.model_name,
        embedding_dim=embedder.dim,
        chunks=chunks,
    )

    store = ChromaStore(path=chroma_path, collection_name=collection_name)

    if recreate:
        store.recreate()

    store.ensure_embedding_compatibility(
        model_name=embedder.model_name, dim=embedder.dim
    )

    vectors = embedder.embed_passages([chunk.embed_text for chunk in chunks])

    store.clear_corpus_checksum()
    store.clear_index_revision()
    store.upsert(chunks, vectors)

    new_ids = {chunk.id for chunk in chunks}
    stored_ids = store.get_ids()

    stale_ids = stored_ids - new_ids

    store.delete_ids(sorted(stale_ids))

    stored = store.count()

    if stored == len(chunks):
        store.set_index_revision(index_revision)
        store.set_corpus_checksum(source_checksum)

    return {
        "units": len(units),
        "chunks": len(chunks),
        "vectors": len(vectors),
        "stored": stored,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Constitution into Chroma vector store."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate Chroma collection before ingest.",
    )

    args = parser.parse_args(argv)
    settings = get_settings()

    stats = run_ingest(
        source_path=settings.source_path,
        chroma_path=settings.chroma_path,
        collection_name=settings.chroma_collection,
        embedding_model=settings.embedding_model,
        recreate=args.recreate,
    )

    print(f"Units: {stats['units']}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Vectors: {stats['vectors']}")
    print(f"Stored: {stats['stored']}")
    if stats["stored"] != stats["chunks"]:
        print(
            "WARNING: stored chunk count does not match generated chunk count "
            f"({stats['stored']} != {stats['chunks']})"
        )


if __name__ == "__main__":
    main()
