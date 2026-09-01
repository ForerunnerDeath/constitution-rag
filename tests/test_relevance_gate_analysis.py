from unittest.mock import MagicMock

import pytest

from app.search.store import Hit
from scripts.analyze_relevance_gate import (
    calculate_distribution_stats,
    calculate_top1_z_score,
    collect_observations,
)


def test_calculate_distribution_stats() -> None:
    stats = calculate_distribution_stats(
        [0.1, 0.3, 0.2],
    )

    assert stats.minimum == pytest.approx(0.1)
    assert stats.median == pytest.approx(0.2)
    assert stats.maximum == pytest.approx(0.3)


def test_collect_observations_calculates_top1_top2_margin() -> None:
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]

    store = MagicMock()
    store.search.return_value = [
        Hit(
            id="art-3",
            quote="Первый результат",
            ref="Статья 3",
            article="3",
            part=None,
            part_label=None,
            score=0.91,
        ),
        Hit(
            id="art-4",
            quote="Второй результат",
            ref="Статья 4",
            article="4",
            part=None,
            part_label=None,
            score=0.86,
        ),
        Hit(
            id="art-5",
            quote="Третий результат",
            ref="Статья 5",
            article="5",
            part=None,
            part_label=None,
            score=0.80,
        ),
    ]
    store.count.return_value = 3

    observations = collect_observations(
        [("Кто является источником власти?", "3")],
        embedder=embedder,
        store=store,
    )

    assert len(observations) == 1

    observation = observations[0]

    assert observation.question == "Кто является источником власти?"
    assert observation.expected_article == "3"
    assert observation.top1_score == pytest.approx(0.91)
    assert observation.top2_score == pytest.approx(0.86)
    assert observation.margin == pytest.approx(0.05)
    assert observation.top1_z_score > 0

    embedder.embed_query.assert_called_once_with("Кто является источником власти?")
    store.count.assert_called_once_with()

    store.search.assert_called_once_with(
        [0.1, 0.2, 0.3],
        k=3,
    )


def test_calculate_top1_z_score() -> None:
    z_score = calculate_top1_z_score(
        [0.9, 0.8, 0.7],
    )

    assert z_score == pytest.approx(1.2247448714)
