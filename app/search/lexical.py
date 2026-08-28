import re
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi  # pyright: ignore[reportMissingTypeStubs]

from app.search.store import Hit

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:\.[0-9]+)?")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


@dataclass(frozen=True)
class LexicalHit:
    hit: Hit
    score: float


class LexicalIndex:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits

        tokenized_corpus = [_tokenize(f"{hit.ref} {hit.quote}") for hit in hits]

        self.index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def search(self, query: str, k: int = 20) -> list[LexicalHit]:
        if k <= 0:
            raise ValueError("k must be greater than zero")

        if self.index is None:
            return []

        query_tokens = _tokenize(query)

        if not query_tokens:
            return []

        scores = cast(NDArray[np.float64], self.index.get_scores(query_tokens))  # pyright: ignore[reportUnknownMemberType]

        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        results: list[LexicalHit] = []

        for index, raw_score in ranked:
            score = float(raw_score)

            if score <= 0:
                continue

            results.append(
                LexicalHit(
                    hit=self.hits[index],
                    score=score,
                )
            )

            if len(results) == k:
                break

        return results
