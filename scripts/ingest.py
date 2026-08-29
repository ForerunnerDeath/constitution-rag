import argparse
from pathlib import Path

from app.config import get_settings
from app.ingest.chunker import chunk_units
from app.ingest.loader import load_text
from app.ingest.parser import parse_text
from app.search.embedder import Embedder
from app.search.store import ChromaStore


def run_ingest(
    *,
    source_path: Path,
    chroma_path: Path,
    collection_name: str,
    embedding_model: str,
    recreate: bool = False,
) -> dict[str, int]:
    text = load_text(source_path)
    units = parse_text(text)
    chunks = chunk_units(units)

    embedder = Embedder(embedding_model)

    store = ChromaStore(path=chroma_path, collection_name=collection_name)

    if recreate:
        store.recreate()

    store.ensure_embedding_compatibility(
        model_name=embedder.model_name, dim=embedder.dim
    )

    vectors = embedder.embed_passages([chunk.embed_text for chunk in chunks])

    store.upsert(chunks, vectors)

    stored = store.count()

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


if __name__ == "__main__":
    main()
