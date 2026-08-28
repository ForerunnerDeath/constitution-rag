from app.schemas import (
    ArticleChunkResponse,
    ArticleResponse,
    AskRequest,
    AskResponse,
    CitationResponse,
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


def test_search_hit_response_accepts_missing_cosine_score() -> None:
    hit = Hit(
        id="art-15-p-3",
        quote="Текст статьи.",
        ref="Статья 15, часть 3",
        article="15",
        part=3,
        part_label="3",
        score=None,
    )

    response = SearchHitResponse.model_validate(hit)

    assert response.score is None


def test_citation_response_from_hit() -> None:
    hit = Hit(
        id="art-3-p-1",
        quote="Носителем суверенитета является многонациональный народ.",
        ref="Статья 3, часть 1",
        article="3",
        part=1,
        part_label="1",
        score=0.84,
    )

    response = CitationResponse.model_validate(hit)

    assert response.model_dump() == {
        "id": "art-3-p-1",
        "quote": "Носителем суверенитета является многонациональный народ.",
        "ref": "Статья 3, часть 1",
        "article": "3",
        "part": 1,
        "part_label": "1",
    }


def test_ask_response_without_answer() -> None:
    response = AskResponse(
        found=False,
        answer=None,
        message="В тексте Конституции прямого ответа не нашлось.",
        citations=[],
        llm_used=False,
    )

    assert response.model_dump() == {
        "found": False,
        "answer": None,
        "message": "В тексте Конституции прямого ответа не нашлось.",
        "citations": [],
        "llm_used": False,
    }
