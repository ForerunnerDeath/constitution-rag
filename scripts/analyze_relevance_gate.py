from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev

from app.config import get_settings
from app.search.embedder import Embedder
from app.search.store import ChromaStore
from scripts.evaluate import load_questions

DEV_DATASET_PATH = Path("eval/dev.csv")


@dataclass(frozen=True)
class GateObservation:
    question: str
    expected_article: str
    top1_score: float
    top2_score: float
    margin: float
    top1_z_score: float


@dataclass(frozen=True)
class DistributionStats:
    minimum: float
    median: float
    maximum: float


def calculate_distribution_stats(values: list[float]) -> DistributionStats:
    if not values:
        raise ValueError("Values must not be empty")

    return DistributionStats(
        minimum=min(values),
        median=median(values),
        maximum=max(values),
    )


def calculate_top1_z_score(scores: list[float]) -> float:
    if len(scores) < 2:
        raise ValueError("At least two scores are required")

    standard_deviation = pstdev(scores)

    if standard_deviation == 0:
        raise ValueError("Score distribution must have non-zero variance")

    return (scores[0] - mean(scores)) / standard_deviation


def collect_observations(
    questions: list[tuple[str, str]],
    *,
    embedder: Embedder,
    store: ChromaStore,
) -> list[GateObservation]:
    observations: list[GateObservation] = []

    corpus_size = store.count()

    if corpus_size < 2:
        raise RuntimeError("Vector store must contain at least two chunks")

    for question, expected_article in questions:
        vector = embedder.embed_query(question)
        hits = store.search(vector, k=corpus_size)

        if len(hits) < 2:
            raise RuntimeError("Vector search must return at least two hits")

        top1_score = hits[0].score
        top2_score = hits[1].score

        if top1_score is None or top2_score is None:
            raise RuntimeError("Vector search hits must have cosine scores")

        scores = [hit.score for hit in hits]

        if any(score is None for score in scores):
            raise RuntimeError("Vector search hits must have cosine scores")

        numeric_scores = [score for score in scores if score is not None]

        top1_z_score = calculate_top1_z_score(numeric_scores)

        observations.append(
            GateObservation(
                question=question,
                expected_article=expected_article,
                top1_score=top1_score,
                top2_score=top2_score,
                margin=top1_score - top2_score,
                top1_z_score=top1_z_score,
            )
        )

    return observations


def main() -> None:
    settings = get_settings()

    questions = load_questions(DEV_DATASET_PATH)

    embedder = Embedder(settings.embedding_model)

    store = ChromaStore(
        path=settings.chroma_path,
        collection_name=settings.chroma_collection,
    )

    store.ensure_embedding_compatibility(
        model_name=embedder.model_name,
        dim=embedder.dim,
    )

    observations = collect_observations(
        questions,
        embedder=embedder,
        store=store,
    )

    positives = [
        observation
        for observation in observations
        if observation.expected_article != "NONE"
    ]
    negatives = [
        observation
        for observation in observations
        if observation.expected_article == "NONE"
    ]

    positive_top1 = [observation.top1_score for observation in positives]
    negative_top1 = [observation.top1_score for observation in negatives]

    positive_margins = [observation.margin for observation in positives]
    negative_margins = [observation.margin for observation in negatives]

    positive_z_scores = [observation.top1_z_score for observation in positives]
    negative_z_scores = [observation.top1_z_score for observation in negatives]

    print("Per-question observations:")
    print()

    for observation in observations:
        label = "NEGATIVE" if observation.expected_article == "NONE" else "POSITIVE"

        print(
            f"{label:8} "
            f"top1={observation.top1_score:.4f} "
            f"top2={observation.top2_score:.4f} "
            f"margin={observation.margin:.4f} "
            f"z={observation.top1_z_score:.4f} "
            f"| {observation.question}"
        )

    print()
    print("Summary:")
    print()

    for name, values in (
        ("Positive TOP-1", positive_top1),
        ("Negative TOP-1", negative_top1),
        ("Positive margin", positive_margins),
        ("Negative margin", negative_margins),
        ("Positive z-score", positive_z_scores),
        ("Negative z-score", negative_z_scores),
    ):
        stats = calculate_distribution_stats(values)

        print(
            f"{name:16} "
            f"min={stats.minimum:.4f} "
            f"median={stats.median:.4f} "
            f"max={stats.maximum:.4f}"
        )


if __name__ == "__main__":
    main()
