from app.schemas import (
    ArticleChunkResponse,
    ArticleResponse,
    SearchHitResponse,
    SearchResponse,
)
from app.search.store import Hit


def test_search_hit_response_from_hit() -> None:
    hit = Hit(
        id="art-29-p-3-4",
        quote="Текст Конституции.",
        ref="Статья 29, части 3-4",
        article="29",
        part=None,
        part_label="3-4",
        score=0.8396,
    )

    response = SearchHitResponse.model_validate(hit)

    assert response.model_dump() == {
        "id": "art-29-p-3-4",
        "quote": "Текст Конституции.",
        "ref": "Статья 29, части 3-4",
        "article": "29",
        "part": None,
        "part_label": "3-4",
        "score": 0.8396,
    }


def test_search_response() -> None:
    response = SearchResponse(
        hits=[
            SearchHitResponse(
                id="art-29",
                quote="Текст.",
                ref="Статья 29",
                article="29",
                part=None,
                part_label=None,
                score=0.84,
            )
        ],
        took_ms=12.5,
        collection_version="constitution_e5_small",
    )

    assert response.model_dump() == {
        "hits": [
            {
                "id": "art-29",
                "quote": "Текст.",
                "ref": "Статья 29",
                "article": "29",
                "part": None,
                "part_label": None,
                "score": 0.84,
            }
        ],
        "took_ms": 12.5,
        "collection_version": "constitution_e5_small",
    }


def test_article_chunk_response_from_hit() -> None:
    hit = Hit(
        id="art-81-p-2-c-1",
        quote="Текст части 2.",
        ref="Статья 81, часть 2",
        article="81",
        part=2,
        part_label="2",
        score=1.0,
    )

    response = ArticleChunkResponse.model_validate(hit)

    assert response.model_dump() == {
        "id": "art-81-p-2-c-1",
        "quote": "Текст части 2.",
        "ref": "Статья 81, часть 2",
        "part": 2,
        "part_label": "2",
    }


def test_article_response() -> None:
    response = ArticleResponse(
        article="81",
        chunks=[
            ArticleChunkResponse(
                id="art-81-p-1",
                quote="Текст части 1.",
                ref="Статья 81, часть 1",
                part=1,
                part_label="1",
            )
        ],
    )

    assert response.model_dump() == {
        "article": "81",
        "chunks": [
            {
                "id": "art-81-p-1",
                "quote": "Текст части 1.",
                "ref": "Статья 81, часть 1",
                "part": 1,
                "part_label": "1",
            }
        ],
    }
