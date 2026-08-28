from app.schemas import SearchHitResponse
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


from app.schemas import SearchHitResponse, SearchResponse


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
