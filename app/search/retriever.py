from dataclasses import dataclass

from app.search.embedder import Embedder
from app.search.lexical import LexicalHit, LexicalIndex
from app.search.store import ChromaStore, Hit


def _deduplicate_hits(hits: list[Hit]) -> list[Hit]:
    seen_ids: set[str] = set()
    result: list[Hit] = []

    for hit in hits:
        if hit.id in seen_ids:
            continue

        seen_ids.add(hit.id)
        result.append(hit)

    return result


@dataclass(frozen=True)
class FusedHit:
    hit: Hit
    rrf_score: float


def reciprocal_rank_fusion(
    vector_hits: list[Hit],
    lexical_hits: list[LexicalHit],
    *,
    rrf_constant: int = 60,
) -> list[FusedHit]:
    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be greater than zero")

    scores: dict[str, float] = {}
    hits_by_id: dict[str, Hit] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (rrf_constant + rank)

        hits_by_id[hit.id] = hit

    for rank, lexical_hit in enumerate(lexical_hits, start=1):
        hit = lexical_hit.hit

        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (rrf_constant + rank)

        hits_by_id.setdefault(hit.id, hit)

    result = [
        FusedHit(hit=hits_by_id[hit_id], rrf_score=rrf_score)
        for hit_id, rrf_score in scores.items()
    ]

    result.sort(
        key=lambda item: (
            -item.rrf_score,
            item.hit.id,
        )
    )

    return result


class Retriever:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: ChromaStore,
        min_score: float,
        lexical_index: LexicalIndex | None = None,
        candidate_k: int = 20,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")

        self.embedder = embedder
        self.store = store
        self.min_score = min_score
        self.lexical_index = lexical_index
        self.candidate_k = candidate_k

    def retrieve(self, query: str, k: int = 5, use_hybrid: bool = False) -> list[Hit]:
        if k <= 0:
            raise ValueError("k must be greater than zero")

        if k > self.candidate_k:
            raise ValueError("k must not exceed candidate_k")

        vector = self.embedder.embed_query(query)

        vector_candidates = _deduplicate_hits(
            self.store.search(
                vector,
                self.candidate_k,
            )
        )

        relevant_vector_hits: list[Hit] = []

        for hit in vector_candidates:
            if hit.score is None:
                raise RuntimeError("Vector search hit must have cosine score")

            if hit.score >= self.min_score:
                relevant_vector_hits.append(hit)

        if not relevant_vector_hits:
            return []

        if not use_hybrid:
            return relevant_vector_hits[:k]

        if self.lexical_index is None:
            raise RuntimeError("Lexical index is not initialized")

        lexical_candidates = self.lexical_index.search(
            query,
            k=self.candidate_k,
        )

        fused_hits = reciprocal_rank_fusion(
            relevant_vector_hits,
            lexical_candidates,
        )

        return [fused_hit.hit for fused_hit in fused_hits[:k]]

    @property
    def collection_name(self) -> str:
        return self.store.collection_name
