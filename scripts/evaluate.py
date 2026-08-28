from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from app.config import get_settings
from app.search.embedder import Embedder
from app.search.lexical import LexicalIndex
from app.search.retriever import Retriever
from app.search.store import ChromaStore

QUESTIONS_PATH = Path("eval/questions.csv")


@dataclass(frozen=True)
class Metrics:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    refusal_accuracy: float
    false_refusal_rate: float


@dataclass(frozen=True)
class ScoreStats:
    minimum: float
    median: float
    maximum: float


@dataclass(frozen=True)
class EvaluationResult:
    metrics: Metrics
    positive_scores: ScoreStats
    negative_scores: ScoreStats
    positive_count: int
    negative_count: int


def load_questions() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    with QUESTIONS_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"question", "expected_article"}

        if reader.fieldnames is None:
            raise ValueError("CSV file has no header")

        if not required_columns.issubset(reader.fieldnames):
            raise ValueError("CSV must contain question and expected_article columns")

        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Invalid CSV row {row_number}: too many columns")

            question = row["question"].strip()
            expected_article = row["expected_article"].strip()

            if not question or not expected_article:
                raise ValueError(f"Invalid CSV row {row_number}: empty field")

            rows.append((question, expected_article))

    return rows


def build_components(*, min_score: float) -> tuple[Retriever, Embedder, ChromaStore]:
    settings = get_settings()

    embedder = Embedder(settings.embedding_model)

    store = ChromaStore(
        path=settings.chroma_path,
        collection_name=settings.chroma_collection,
    )

    lexical_index = LexicalIndex(store.get_all())

    retriever = Retriever(
        embedder=embedder,
        store=store,
        lexical_index=lexical_index,
        min_score=min_score,
    )

    return retriever, embedder, store


def get_rank(found_articles: list[str | None], expected_article: str) -> int | None:
    for rank, article in enumerate(found_articles, start=1):
        if article == expected_article:
            return rank

    return None


def reciprocal_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0

    return 1 / rank


def get_raw_top_score(question: str, embedder: Embedder, store: ChromaStore) -> float:
    vector = embedder.embed_query(question)
    hits = store.search(vector, k=1)

    if not hits:
        raise RuntimeError("Vector search returned no hits")

    score = hits[0].score

    if score is None:
        raise RuntimeError("Vector search hit must have a cosine score")

    return score


def calculate_score_stats(scores: list[float]) -> ScoreStats:
    if not scores:
        raise ValueError("Score list must not be empty")

    return ScoreStats(
        minimum=min(scores),
        median=median(scores),
        maximum=max(scores),
    )


def evaluate(
    questions: list[tuple[str, str]],
    retriever: Retriever,
    embedder: Embedder,
    store: ChromaStore,
    *,
    use_hybrid: bool,
) -> EvaluationResult:
    positive_count = 0
    negative_count = 0

    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0

    reciprocal_rank_sum = 0.0

    correct_refusals = 0
    false_refusals = 0

    positive_scores: list[float] = []
    negative_scores: list[float] = []

    for question, expected_article in questions:
        raw_top_score = get_raw_top_score(question, embedder, store)

        hits = retriever.retrieve(question, k=5, use_hybrid=use_hybrid)

        found_articles = [hit.article for hit in hits]

        if expected_article == "NONE":
            negative_count += 1
            negative_scores.append(raw_top_score)

            if not hits:
                correct_refusals += 1

            continue

        positive_count += 1
        positive_scores.append(raw_top_score)

        if not hits:
            false_refusals += 1

        rank = get_rank(found_articles, expected_article)

        if rank is not None:
            if rank <= 1:
                recall_at_1 += 1

            if rank <= 3:
                recall_at_3 += 1

            if rank <= 5:
                recall_at_5 += 1

        reciprocal_rank_sum += reciprocal_rank(rank)

    if positive_count == 0:
        raise ValueError("Evaluation dataset must contain positive questions")

    if negative_count == 0:
        raise ValueError("Evaluation dataset must contain negative questions")

    metrics = Metrics(
        recall_at_1=recall_at_1 / positive_count,
        recall_at_3=recall_at_3 / positive_count,
        recall_at_5=recall_at_5 / positive_count,
        mrr=reciprocal_rank_sum / positive_count,
        refusal_accuracy=correct_refusals / negative_count,
        false_refusal_rate=false_refusals / positive_count,
    )

    return EvaluationResult(
        metrics=metrics,
        positive_scores=calculate_score_stats(positive_scores),
        negative_scores=calculate_score_stats(negative_scores),
        positive_count=positive_count,
        negative_count=negative_count,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Constitution retrieval quality."
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Override configured retrieval min_score.",
    )

    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Evaluate hybrid vector + BM25 retrieval.",
    )

    args = parser.parse_args(argv)

    settings = get_settings()

    min_score = args.min_score if args.min_score is not None else settings.min_score

    questions = load_questions()

    retriever, embedder, store = build_components(min_score=min_score)

    result = evaluate(questions, retriever, embedder, store, use_hybrid=args.hybrid)

    metrics = result.metrics

    print("Evaluation configuration:")
    print(f"Questions:       {len(questions)}")
    print(f"Positive:        {result.positive_count}")
    print(f"Negative:        {result.negative_count}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Min score:       {min_score:.3f}")
    print(f"Hybrid:          {args.hybrid}")

    print()
    print("Metrics:")
    print(f"Recall@1:         {metrics.recall_at_1:.3f}")
    print(f"Recall@3:         {metrics.recall_at_3:.3f}")
    print(f"Recall@5:         {metrics.recall_at_5:.3f}")
    print(f"MRR:              {metrics.mrr:.3f}")
    print(f"Refusal accuracy: {metrics.refusal_accuracy:.3f}")
    print(f"False refusal:    {metrics.false_refusal_rate:.3f}")

    print()
    print("Raw TOP-1 vector scores (before threshold):")
    print(
        "Positive: "
        f"min={result.positive_scores.minimum:.4f}, "
        f"median={result.positive_scores.median:.4f}, "
        f"max={result.positive_scores.maximum:.4f}"
    )
    print(
        "Negative: "
        f"min={result.negative_scores.minimum:.4f}, "
        f"median={result.negative_scores.median:.4f}, "
        f"max={result.negative_scores.maximum:.4f}"
    )


if __name__ == "__main__":
    main()
