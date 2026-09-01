from unittest.mock import MagicMock, patch

import pytest

from app.search.embedder import Embedder
from app.search.lexical import LexicalHit, LexicalIndex
from app.search.retriever import (
    Retriever,
    _deduplicate_hits,
    reciprocal_rank_fusion,
)
from app.search.store import ChromaStore, Hit


def _make_hit(
    *,
    hit_id: str,
    score: float | None,
    article: str = "1",
) -> Hit:
    return Hit(
        id=hit_id,
        quote=f"Текст {hit_id}.",
        ref=f"Статья {article}",
        article=article,
        part=None,
        part_label=None,
        score=score,
    )


def test_retrieve_uses_candidate_k_for_vector_search() -> None:
    embedder = MagicMock(spec=Embedder)
    store = MagicMock(spec=ChromaStore)

    vector = [0.1, 0.2, 0.3]

    embedder.embed_query.return_value = vector
    store.search.return_value = []

    retriever = Retriever(
        embedder=embedder,
        store=store,
        min_score=0.70,
        candidate_k=20,
    )

    hits = retriever.retrieve(
        "Кто является источником власти?",
        k=5,
    )

    assert hits == []

    embedder.embed_query.assert_called_once_with("Кто является источником власти?")

    store.search.assert_called_once_with(
        vector,
        20,
    )


def test_retrieve_filters_by_min_score_and_limits_to_k() -> None:
    embedder = MagicMock(spec=Embedder)
    store = MagicMock(spec=ChromaStore)

    embedder.embed_query.return_value = [0.1, 0.2]

    store.search.return_value = [
        _make_hit(hit_id="a", score=0.90),
        _make_hit(hit_id="b", score=0.82),
        _make_hit(hit_id="c", score=0.75),
        _make_hit(hit_id="d", score=0.69),
    ]

    retriever = Retriever(
        embedder=embedder,
        store=store,
        min_score=0.70,
        candidate_k=20,
    )

    hits = retriever.retrieve(
        "Тестовый вопрос",
        k=2,
    )

    assert [hit.id for hit in hits] == [
        "a",
        "b",
    ]


def test_retrieve_returns_empty_when_all_scores_below_threshold() -> None:
    embedder = MagicMock(spec=Embedder)
    store = MagicMock(spec=ChromaStore)

    embedder.embed_query.return_value = [0.1]

    store.search.return_value = [
        _make_hit(hit_id="a", score=0.69),
        _make_hit(hit_id="b", score=0.65),
    ]

    retriever = Retriever(
        embedder=embedder,
        store=store,
        min_score=0.70,
    )

    hits = retriever.retrieve("Как приготовить борщ?")

    assert hits == []


def test_deduplicate_hits_keeps_first_hit_for_each_id() -> None:
    first = _make_hit(hit_id="same", score=0.90)
    duplicate = _make_hit(hit_id="same", score=0.80)
    other = _make_hit(hit_id="other", score=0.75)

    result = _deduplicate_hits(
        [
            first,
            duplicate,
            other,
        ]
    )

    assert result == [
        first,
        other,
    ]


@pytest.mark.parametrize("k", [0, -1])
def test_retrieve_rejects_non_positive_k(k: int) -> None:
    retriever = Retriever(
        embedder=MagicMock(spec=Embedder),
        store=MagicMock(spec=ChromaStore),
        min_score=0.70,
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            "Тестовый вопрос",
            k=k,
        )


def test_retrieve_rejects_k_greater_than_candidate_k() -> None:
    retriever = Retriever(
        embedder=MagicMock(spec=Embedder),
        store=MagicMock(spec=ChromaStore),
        min_score=0.70,
        candidate_k=5,
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            "Тестовый вопрос",
            k=6,
        )


def test_rrf_promotes_hit_present_in_both_rankings() -> None:
    vector_a = _make_hit(
        hit_id="a",
        score=0.90,
    )
    vector_b = _make_hit(
        hit_id="b",
        score=0.85,
    )

    lexical_b = LexicalHit(
        hit=_make_hit(
            hit_id="b",
            score=None,
        ),
        score=10.0,
    )
    lexical_c = LexicalHit(
        hit=_make_hit(
            hit_id="c",
            score=None,
        ),
        score=8.0,
    )

    result = reciprocal_rank_fusion(
        [vector_a, vector_b],
        [lexical_b, lexical_c],
    )

    assert [item.hit.id for item in result] == [
        "b",
        "a",
        "c",
    ]


def test_rrf_preserves_vector_score_for_shared_hit() -> None:
    vector_hit = _make_hit(
        hit_id="shared",
        score=0.84,
    )

    lexical_hit = LexicalHit(
        hit=_make_hit(
            hit_id="shared",
            score=None,
        ),
        score=12.0,
    )

    result = reciprocal_rank_fusion(
        [vector_hit],
        [lexical_hit],
    )

    assert len(result) == 1
    assert result[0].hit.id == "shared"
    assert result[0].hit.score == pytest.approx(0.84)


def test_rrf_keeps_lexical_only_hit_without_cosine_score() -> None:
    lexical_hit = LexicalHit(
        hit=_make_hit(
            hit_id="lexical-only",
            score=None,
        ),
        score=7.0,
    )

    result = reciprocal_rank_fusion(
        [],
        [lexical_hit],
    )

    assert len(result) == 1
    assert result[0].hit.id == "lexical-only"
    assert result[0].hit.score is None


def test_rrf_rejects_non_positive_constant() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            [],
            [],
            rrf_constant=0,
        )


def test_retrieve_uses_hybrid_search_and_rrf() -> None:
    embedder = MagicMock(spec=Embedder)
    store = MagicMock(spec=ChromaStore)
    lexical_index = MagicMock(spec=LexicalIndex)

    vector = [0.1, 0.2]

    embedder.embed_query.return_value = vector

    vector_wrong = _make_hit(
        hit_id="art-65",
        score=0.86,
        article="65",
    )
    vector_target = _make_hit(
        hit_id="art-15",
        score=0.85,
        article="15",
    )

    store.search.return_value = [
        vector_wrong,
        vector_target,
    ]

    lexical_index.search.return_value = [
        LexicalHit(
            hit=_make_hit(
                hit_id="art-15",
                score=None,
                article="15",
            ),
            score=7.0,
        )
    ]

    retriever = Retriever(
        embedder=embedder,
        store=store,
        lexical_index=lexical_index,
        min_score=0.80,
        candidate_k=20,
    )

    hits = retriever.retrieve(
        "статья 15",
        k=2,
        use_hybrid=True,
    )

    assert [hit.id for hit in hits] == [
        "art-15",
        "art-65",
    ]

    lexical_index.search.assert_called_once_with(
        "статья 15",
        k=20,
    )


def test_retrieve_does_not_run_lexical_search_when_vector_threshold_fails() -> None:
    embedder = MagicMock(spec=Embedder)
    store = MagicMock(spec=ChromaStore)
    lexical_index = MagicMock(spec=LexicalIndex)

    embedder.embed_query.return_value = [0.1]

    store.search.return_value = [
        _make_hit(
            hit_id="irrelevant",
            score=0.79,
        )
    ]

    retriever = Retriever(
        embedder=embedder,
        store=store,
        lexical_index=lexical_index,
        min_score=0.80,
    )

    hits = retriever.retrieve(
        "Как приготовить борщ?",
        use_hybrid=True,
    )

    assert hits == []
    lexical_index.search.assert_not_called()


def test_retrieve_does_not_use_lexical_index_when_hybrid_disabled() -> None:
    embedder = MagicMock(spec=Embedder)
    store = MagicMock(spec=ChromaStore)
    lexical_index = MagicMock(spec=LexicalIndex)

    embedder.embed_query.return_value = [0.1]

    store.search.return_value = [
        _make_hit(
            hit_id="vector-hit",
            score=0.85,
        )
    ]

    retriever = Retriever(
        embedder=embedder,
        store=store,
        lexical_index=lexical_index,
        min_score=0.80,
    )

    hits = retriever.retrieve(
        "Тестовый вопрос",
        use_hybrid=False,
    )

    assert [hit.id for hit in hits] == ["vector-hit"]
    lexical_index.search.assert_not_called()


def test_retrieve_with_metrics_returns_hits_and_timings() -> None:
    embedder = MagicMock(spec=Embedder)
    store = MagicMock(spec=ChromaStore)

    vector = [0.1, 0.2]

    embedder.embed_query.return_value = vector

    expected_hit = _make_hit(
        hit_id="art-3",
        score=0.90,
        article="3",
    )

    store.search.return_value = [expected_hit]

    retriever = Retriever(
        embedder=embedder,
        store=store,
        min_score=0.80,
    )

    with patch(
        "app.search.retriever.perf_counter",
        side_effect=[
            10.000,
            10.025,
            20.000,
            20.075,
        ],
    ):
        result = retriever.retrieve_with_metrics(
            "Кто является источником власти?",
            k=5,
        )

    assert result.hits == [expected_hit]

    assert result.embed_ms == pytest.approx(25.0)
    assert result.search_ms == pytest.approx(75.0)

    embedder.embed_query.assert_called_once_with("Кто является источником власти?")

    store.search.assert_called_once_with(
        vector,
        20,
    )


def test_retrieve_with_metrics_returns_timings_when_no_hits_found() -> None:
    embedder = MagicMock(spec=Embedder)
    store = MagicMock(spec=ChromaStore)

    embedder.embed_query.return_value = [0.1]

    store.search.return_value = [
        _make_hit(
            hit_id="irrelevant",
            score=0.70,
        )
    ]

    retriever = Retriever(
        embedder=embedder,
        store=store,
        min_score=0.80,
    )

    with patch(
        "app.search.retriever.perf_counter",
        side_effect=[
            1.000,
            1.010,
            2.000,
            2.020,
        ],
    ):
        result = retriever.retrieve_with_metrics("Как приготовить борщ?")

    assert result.hits == []
    assert result.embed_ms == pytest.approx(10.0)
    assert result.search_ms == pytest.approx(20.0)


def test_retrieve_uses_min_score_as_query_level_gate() -> None:
    embedder = MagicMock(spec=Embedder)
    store = MagicMock(spec=ChromaStore)

    embedder.embed_query.return_value = [0.1, 0.2]

    store.search.return_value = [
        _make_hit(hit_id="a", score=0.90),
        _make_hit(hit_id="b", score=0.82),
    ]

    retriever = Retriever(
        embedder=embedder,
        store=store,
        min_score=0.833,
        candidate_k=20,
    )

    hits = retriever.retrieve(
        "Тестовый вопрос",
        k=2,
    )

    assert [hit.id for hit in hits] == [
        "a",
        "b",
    ]


def test_hybrid_keeps_vector_candidate_below_query_threshold_after_gate_passes() -> (
    None
):
    embedder = MagicMock(spec=Embedder)
    store = MagicMock(spec=ChromaStore)
    lexical_index = MagicMock(spec=LexicalIndex)

    embedder.embed_query.return_value = [0.1, 0.2]

    vector_top = _make_hit(
        hit_id="a",
        score=0.90,
    )
    vector_below_threshold = _make_hit(
        hit_id="b",
        score=0.82,
    )

    store.search.return_value = [
        vector_top,
        vector_below_threshold,
    ]

    lexical_index.search.return_value = [
        LexicalHit(
            hit=_make_hit(
                hit_id="b",
                score=None,
            ),
            score=10.0,
        )
    ]

    retriever = Retriever(
        embedder=embedder,
        store=store,
        lexical_index=lexical_index,
        min_score=0.833,
        candidate_k=20,
    )

    hits = retriever.retrieve(
        "Тестовый вопрос",
        k=2,
        use_hybrid=True,
    )

    assert [hit.id for hit in hits] == [
        "b",
        "a",
    ]
