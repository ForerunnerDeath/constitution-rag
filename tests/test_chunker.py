import pytest

from app.ingest.chunker import (
    _build_chunk_id,
    _build_ref,
    _create_chunk,
    _create_chunk_from_group,
    _group_short_units,
    _split_chunk,
    _split_long_text,
    chunk_units,
)
from app.ingest.parser import Unit


def _article_unit(
    *,
    article: str,
    part: int | None,
    part_label: str | None,
    text: str,
) -> Unit:
    return Unit(
        section="первый",
        chapter=1,
        chapter_title="ТЕСТОВАЯ ГЛАВА",
        article=article,
        part=part,
        part_label=part_label,
        text=text,
        kind="article",
    )


def test_create_article_chunk() -> None:
    quote = "Носителем суверенитета и единственным источником власти является народ."

    chunk = _create_chunk(
        quote=quote,
        chapter=1,
        chapter_title="ОСНОВЫ КОНСТИТУЦИОННОГО СТРОЯ",
        article="3",
        part=1,
        part_label="1",
        kind="article",
    )

    assert chunk.id == "art-3-p-1"
    assert chunk.ref == "Статья 3, часть 1"

    assert chunk.quote == quote

    assert chunk.embed_text == (
        f"Глава 1. ОСНОВЫ КОНСТИТУЦИОННОГО СТРОЯ. Статья 3, часть 1. {quote}"
    )

    assert "Глава 1." not in chunk.quote


def test_create_article_without_part() -> None:
    chunk = _create_chunk(
        quote="Государственная власть осуществляется на основе разделения властей.",
        chapter=1,
        chapter_title="ОСНОВЫ КОНСТИТУЦИОННОГО СТРОЯ",
        article="10",
        part=None,
        part_label=None,
        kind="article",
    )

    assert chunk.id == "art-10"
    assert chunk.ref == "Статья 10"
    assert chunk.part is None
    assert chunk.part_label is None


def test_create_decimal_part_chunk() -> None:
    chunk = _create_chunk(
        quote="Российская Федерация обеспечивает защиту своего суверенитета.",
        chapter=3,
        chapter_title="ФЕДЕРАТИВНОЕ УСТРОЙСТВО",
        article="67",
        part=None,
        part_label="2.1",
        kind="article",
    )

    assert chunk.id == "art-67-p-2.1"
    assert chunk.ref == "Статья 67, часть 2.1"
    assert chunk.part is None
    assert chunk.part_label == "2.1"


def test_create_preamble_chunk() -> None:
    chunk = _create_chunk(
        quote="Мы, многонациональный народ Российской Федерации...",
        chapter=None,
        chapter_title=None,
        article=None,
        part=None,
        part_label=None,
        kind="preamble",
    )

    assert chunk.id == "preamble"
    assert chunk.ref == "Преамбула"

    assert chunk.embed_text.startswith("Преамбула Конституции Российской Федерации.")


def test_create_transitional_chunk() -> None:
    chunk = _create_chunk(
        quote="Конституция Российской Федерации вступает в силу...",
        chapter=None,
        chapter_title=None,
        article=None,
        part=1,
        part_label="1",
        kind="transitional",
    )

    assert chunk.id == "transitional-1"

    assert chunk.ref == ("Заключительные и переходные положения, пункт 1")


def test_build_merged_part_ref_and_id() -> None:
    ref = _build_ref(
        kind="article",
        article="65",
        part_label="1-2",
    )

    chunk_id = _build_chunk_id(
        kind="article",
        article="65",
        part_label="1-2",
    )

    assert ref == "Статья 65, части 1-2"
    assert chunk_id == "art-65-p-1-2"


def test_group_short_part_with_next_part() -> None:
    units = [
        _article_unit(
            article="5",
            part=1,
            part_label="1",
            text="A" * 40,
        ),
        _article_unit(
            article="5",
            part=2,
            part_label="2",
            text="B" * 80,
        ),
    ]

    groups = _group_short_units(
        units,
        min_chunk_chars=100,
    )

    assert len(groups) == 1
    assert groups[0] == units


def test_group_short_parts_until_minimum_is_reached() -> None:
    units = [
        _article_unit(
            article="5",
            part=1,
            part_label="1",
            text="A" * 30,
        ),
        _article_unit(
            article="5",
            part=2,
            part_label="2",
            text="B" * 30,
        ),
        _article_unit(
            article="5",
            part=3,
            part_label="3",
            text="C" * 60,
        ),
    ]

    groups = _group_short_units(
        units,
        min_chunk_chars=100,
    )

    assert len(groups) == 1
    assert groups[0] == units


def test_short_part_is_not_merged_with_next_article() -> None:
    first = _article_unit(
        article="5",
        part=2,
        part_label="2",
        text="A" * 40,
    )

    second = _article_unit(
        article="6",
        part=1,
        part_label="1",
        text="B" * 200,
    )

    groups = _group_short_units(
        [first, second],
        min_chunk_chars=100,
    )

    assert groups == [
        [first],
        [second],
    ]


def test_last_short_part_can_remain_alone() -> None:
    first = _article_unit(
        article="5",
        part=1,
        part_label="1",
        text="A" * 150,
    )

    last = _article_unit(
        article="5",
        part=2,
        part_label="2",
        text="B" * 40,
    )

    groups = _group_short_units(
        [first, last],
        min_chunk_chars=100,
    )

    assert groups == [
        [first],
        [last],
    ]


def test_article_without_parts_is_not_merged() -> None:
    first = _article_unit(
        article="10",
        part=None,
        part_label=None,
        text="A" * 40,
    )

    second = _article_unit(
        article="11",
        part=1,
        part_label="1",
        text="B" * 200,
    )

    groups = _group_short_units(
        [first, second],
        min_chunk_chars=100,
    )

    assert groups == [
        [first],
        [second],
    ]


def test_decimal_part_can_be_merged() -> None:
    first = _article_unit(
        article="67",
        part=2,
        part_label="2",
        text="A" * 40,
    )

    decimal = _article_unit(
        article="67",
        part=None,
        part_label="2.1",
        text="B" * 100,
    )

    groups = _group_short_units(
        [first, decimal],
        min_chunk_chars=100,
    )

    assert groups == [
        [first, decimal],
    ]


def test_create_chunk_from_single_unit() -> None:
    unit = _article_unit(
        article="3",
        part=1,
        part_label="1",
        text="Народ является источником власти.",
    )

    chunk = _create_chunk_from_group([unit])

    assert chunk.id == "art-3-p-1"
    assert chunk.ref == "Статья 3, часть 1"
    assert chunk.part == 1
    assert chunk.part_label == "1"
    assert chunk.quote == unit.text


def test_create_chunk_from_merged_parts() -> None:
    first = _article_unit(
        article="5",
        part=1,
        part_label="1",
        text="Первая часть.",
    )

    second = _article_unit(
        article="5",
        part=2,
        part_label="2",
        text="Вторая часть.",
    )

    chunk = _create_chunk_from_group([first, second])

    assert chunk.id == "art-5-p-1-2"
    assert chunk.ref == "Статья 5, части 1-2"

    assert chunk.part is None
    assert chunk.part_label == "1-2"

    assert chunk.quote == ("Первая часть.\nВторая часть.")

    assert "Статья 5, части 1-2" in chunk.embed_text


def test_create_chunk_from_regular_and_decimal_parts() -> None:
    regular = _article_unit(
        article="67",
        part=2,
        part_label="2",
        text="Обычная часть.",
    )

    decimal = _article_unit(
        article="67",
        part=None,
        part_label="2.1",
        text="Дробная часть.",
    )

    chunk = _create_chunk_from_group([regular, decimal])

    assert chunk.id == "art-67-p-2-2.1"
    assert chunk.ref == "Статья 67, части 2-2.1"
    assert chunk.part is None
    assert chunk.part_label == "2-2.1"


def test_chunk_units_merges_short_parts() -> None:
    units = [
        _article_unit(
            article="5",
            part=1,
            part_label="1",
            text="A" * 40,
        ),
        _article_unit(
            article="5",
            part=2,
            part_label="2",
            text="B" * 80,
        ),
        _article_unit(
            article="5",
            part=3,
            part_label="3",
            text="C" * 200,
        ),
    ]

    chunks = chunk_units(
        units,
        min_chunk_chars=100,
    )

    assert len(chunks) == 2

    assert chunks[0].id == "art-5-p-1-2"
    assert chunks[0].part_label == "1-2"

    assert chunks[1].id == "art-5-p-3"
    assert chunks[1].part == 3


def test_chunk_units_is_deterministic() -> None:
    units = [
        _article_unit(
            article="5",
            part=1,
            part_label="1",
            text="A" * 40,
        ),
        _article_unit(
            article="5",
            part=2,
            part_label="2",
            text="B" * 80,
        ),
    ]

    first_result = chunk_units(
        units,
        min_chunk_chars=100,
    )

    second_result = chunk_units(
        units,
        min_chunk_chars=100,
    )

    assert first_result == second_result


def test_split_long_text_with_overlap() -> None:
    first = "A" * 40 + "."
    second = "B" * 40 + "."
    third = "C" * 40 + "."

    text = "\n".join([first, second, third])

    chunks = _split_long_text(
        text,
        max_chunk_chars=90,
    )

    assert chunks == [
        f"{first}\n{second}",
        f"{second}\n{third}",
    ]


def test_split_chunk_gets_stable_fragment_ids() -> None:
    first = "A" * 40 + "."
    second = "B" * 40 + "."
    third = "C" * 40 + "."

    unit = _article_unit(
        article="83",
        part=None,
        part_label=None,
        text="\n".join([first, second, third]),
    )

    chunk = _create_chunk_from_group([unit])

    result = _split_chunk(
        chunk,
        max_chunk_chars=90,
    )

    assert len(result) == 2

    assert result[0].id == "art-83-c-1"
    assert result[1].id == "art-83-c-2"

    assert result[0].ref == "Статья 83"
    assert result[1].ref == "Статья 83"


def test_split_chunk_preserves_overlap() -> None:
    first = "A" * 40 + "."
    second = "B" * 40 + "."
    third = "C" * 40 + "."

    unit = _article_unit(
        article="83",
        part=None,
        part_label=None,
        text="\n".join([first, second, third]),
    )

    chunk = _create_chunk_from_group([unit])

    result = _split_chunk(
        chunk,
        max_chunk_chars=90,
    )

    assert second in result[0].quote
    assert second in result[1].quote


def test_indivisible_segment_can_exceed_max_size() -> None:
    text = "A" * 150

    chunks = _split_long_text(
        text,
        max_chunk_chars=100,
    )

    assert chunks == [text]


def test_chunk_units_splits_long_chunk() -> None:
    first = "A" * 40 + "."
    second = "B" * 40 + "."
    third = "C" * 40 + "."

    units = [
        _article_unit(
            article="83",
            part=None,
            part_label=None,
            text="\n".join([first, second, third]),
        )
    ]

    chunks = chunk_units(
        units,
        min_chunk_chars=20,
        max_chunk_chars=90,
    )

    assert len(chunks) == 2

    assert [chunk.id for chunk in chunks] == [
        "art-83-c-1",
        "art-83-c-2",
    ]


def test_split_chunk_preserves_metadata() -> None:
    first = "A" * 40 + "."
    second = "B" * 40 + "."
    third = "C" * 40 + "."

    unit = _article_unit(
        article="102",
        part=1,
        part_label="1",
        text="\n".join([first, second, third]),
    )

    chunk = _create_chunk_from_group([unit])

    result = _split_chunk(
        chunk,
        max_chunk_chars=90,
    )

    assert len(result) == 2

    for fragment in result:
        assert fragment.article == "102"
        assert fragment.part == 1
        assert fragment.part_label == "1"
        assert fragment.chapter == 1
        assert fragment.chapter_title == "ТЕСТОВАЯ ГЛАВА"
        assert fragment.kind == "article"
        assert fragment.ref == "Статья 102, часть 1"


def test_chunk_units_produces_unique_ids() -> None:
    first = "A" * 60 + "."
    second = "B" * 60 + "."
    third = "C" * 60 + "."

    units = [
        _article_unit(
            article="5",
            part=1,
            part_label="1",
            text="X" * 40,
        ),
        _article_unit(
            article="5",
            part=2,
            part_label="2",
            text="Y" * 80,
        ),
        _article_unit(
            article="83",
            part=None,
            part_label=None,
            text="\n".join([first, second, third]),
        ),
    ]

    chunks = chunk_units(
        units,
        min_chunk_chars=50,
        max_chunk_chars=130,
    )

    ids = [chunk.id for chunk in chunks]

    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    ("min_chars", "max_chars"),
    [
        (0, 900),
        (100, 0),
        (-1, 900),
        (100, -1),
        (901, 900),
    ],
)
def test_chunk_units_rejects_invalid_size_limits(
    min_chars: int,
    max_chars: int,
) -> None:
    unit = _article_unit(
        article="1",
        part=1,
        part_label="1",
        text="Текст.",
    )

    with pytest.raises(ValueError):
        chunk_units(
            [unit],
            min_chunk_chars=min_chars,
            max_chunk_chars=max_chars,
        )
