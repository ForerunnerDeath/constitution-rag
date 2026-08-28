import pytest

from app.search.lexical import LexicalIndex, _tokenize
from app.search.store import Hit


def _make_hit(
    *,
    hit_id: str,
    article: str,
    quote: str,
) -> Hit:
    return Hit(
        id=hit_id,
        quote=quote,
        ref=f"Статья {article}",
        article=article,
        part=None,
        part_label=None,
        score=1.0,
    )


def test_tokenize_normalizes_text() -> None:
    assert _tokenize("Статья 67.1, ЧАСТЬ 2!") == [
        "статья",
        "67.1",
        "часть",
        "2",
    ]


def test_search_ranks_exact_article_number_first() -> None:
    hits = [
        _make_hit(
            hit_id="art-3",
            article="3",
            quote="Носителем суверенитета является народ.",
        ),
        _make_hit(
            hit_id="art-15",
            article="15",
            quote="Конституция Российской Федерации имеет высшую юридическую силу.",
        ),
        _make_hit(
            hit_id="art-28",
            article="28",
            quote="Каждому гарантируется свобода совести.",
        ),
    ]

    index = LexicalIndex(hits)

    results = index.search(
        "что говорит статья 15",
        k=3,
    )

    assert results
    assert results[0].hit.id == "art-15"
    assert results[0].hit.article == "15"
    assert results[0].score > 0


def test_search_uses_ref_not_only_quote() -> None:
    hits = [
        _make_hit(
            hit_id="art-3",
            article="3",
            quote="Носителем суверенитета является народ.",
        ),
        _make_hit(
            hit_id="art-15",
            article="15",
            quote="Конституция имеет высшую юридическую силу.",
        ),
        _make_hit(
            hit_id="art-28",
            article="28",
            quote="Каждому гарантируется свобода совести.",
        ),
    ]

    index = LexicalIndex(hits)

    results = index.search(
        "15",
        k=3,
    )

    assert results
    assert results[0].hit.id == "art-15"


def test_search_returns_empty_for_unmatched_query() -> None:
    hits = [
        _make_hit(
            hit_id="art-28",
            article="28",
            quote="Свобода совести и вероисповедания.",
        )
    ]

    index = LexicalIndex(hits)

    assert index.search("борщ картошка свекла") == []


def test_search_returns_empty_for_empty_corpus() -> None:
    index = LexicalIndex([])

    assert index.search("статья 15") == []


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_non_positive_k(k: int) -> None:
    index = LexicalIndex([])

    with pytest.raises(ValueError):
        index.search(
            "статья 15",
            k=k,
        )
