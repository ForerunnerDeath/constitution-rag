from app.ingest.parser import (
    _clean_source_line,
    _parse_article,
    _parse_chapter,
    _parse_part,
    _parse_section,
    parse_text,
)


def test_parse_section() -> None:
    assert _parse_section("РАЗДЕЛ ПЕРВЫЙ") == "первый"
    assert _parse_section("РАЗДЕЛ ВТОРОЙ") == "второй"
    assert _parse_section("Статья 1") is None


def test_parse_chapter_with_title_on_same_line() -> None:
    result = _parse_chapter("Глава 1. ОСНОВЫ КОНСТИТУЦИОННОГО СТРОЯ")

    assert result == (
        1,
        "ОСНОВЫ КОНСТИТУЦИОННОГО СТРОЯ",
    )


def test_parse_chapter_without_inline_title() -> None:
    result = _parse_chapter("ГЛАВА 3.")

    assert result == (3, None)


def test_parse_article() -> None:
    assert _parse_article("Статья 1") == "1"
    assert _parse_article("Статья 67.1 <*>") == "67.1"
    assert _parse_article("Статья 103.1") == "103.1"
    assert _parse_article("Российская Федерация") is None


def test_parse_regular_part() -> None:
    result = _parse_part("2. Народ осуществляет свою власть непосредственно")

    assert result == (
        2,
        "2",
        "Народ осуществляет свою власть непосредственно",
    )


def test_subpoints_are_not_parts() -> None:
    assert _parse_part("а) принятие Конституции") is None
    assert _parse_part("ж.1) защита семьи") is None


def test_parse_article_parts() -> None:
    text = """
РАЗДЕЛ ПЕРВЫЙ

ГЛАВА 1.
ТЕСТОВАЯ ГЛАВА

Статья 3

1. Первый текст.
2. Второй текст.
3. Третий текст.
4. Четвертый текст.
"""

    units = parse_text(text)

    assert len(units) == 4

    assert [unit.article for unit in units] == [
        "3",
        "3",
        "3",
        "3",
    ]

    assert [unit.part for unit in units] == [1, 2, 3, 4]
    assert [unit.part_label for unit in units] == [
        "1",
        "2",
        "3",
        "4",
    ]

    assert all(unit.chapter == 1 for unit in units)
    assert all(unit.chapter_title == "ТЕСТОВАЯ ГЛАВА" for unit in units)


def test_parse_decimal_part_line() -> None:
    result = _parse_part(
        "2.1. Российская Федерация обеспечивает защиту своего суверенитета"
    )

    assert result == (
        None,
        "2.1",
        "Российская Федерация обеспечивает защиту своего суверенитета",
    )


def test_parse_article_without_parts() -> None:
    text = """
РАЗДЕЛ ПЕРВЫЙ

ГЛАВА 1.
ТЕСТОВАЯ ГЛАВА

Статья 2

Человек, его права и свободы являются высшей ценностью.
"""

    units = parse_text(text)

    assert len(units) == 1

    unit = units[0]

    assert unit.article == "2"
    assert unit.part is None
    assert unit.part_label is None
    assert unit.text == ("Человек, его права и свободы являются высшей ценностью.")
    assert unit.kind == "article"


def test_parse_text_decimal_part() -> None:
    text = """
РАЗДЕЛ ПЕРВЫЙ

ГЛАВА 3.
ФЕДЕРАТИВНОЕ УСТРОЙСТВО

Статья 67

2. Обычная часть.
2.1. Дробная часть.
3. Следующая часть.
"""

    units = parse_text(text)

    assert len(units) == 3

    decimal_unit = units[1]

    assert decimal_unit.article == "67"
    assert decimal_unit.part is None
    assert decimal_unit.part_label == "2.1"
    assert decimal_unit.text == "Дробная часть."


def test_subpoints_remain_inside_part() -> None:
    text = """
РАЗДЕЛ ПЕРВЫЙ

ГЛАВА 3.
ФЕДЕРАТИВНОЕ УСТРОЙСТВО

Статья 72

1. В совместном ведении находятся:
а) первый подпункт;
б) второй подпункт;
ж.1) дополнительный подпункт.
2. Следующая часть.
"""

    units = parse_text(text)

    assert len(units) == 2

    first_part = units[0]

    assert first_part.part == 1
    assert "а) первый подпункт;" in first_part.text
    assert "б) второй подпункт;" in first_part.text
    assert "ж.1) дополнительный подпункт." in first_part.text


def test_parse_inline_chapter_title() -> None:
    text = """
РАЗДЕЛ ПЕРВЫЙ

Глава 4. ПРЕЗИДЕНТ РОССИЙСКОЙ ФЕДЕРАЦИИ

Статья 80

Президент Российской Федерации является главой государства.
"""

    units = parse_text(text)

    assert len(units) == 1

    unit = units[0]

    assert unit.chapter == 4
    assert unit.chapter_title == "ПРЕЗИДЕНТ РОССИЙСКОЙ ФЕДЕРАЦИИ"
    assert unit.article == "80"


def test_parse_preamble() -> None:
    text = """
Принята всенародным голосованием 12 декабря 1993 года

КОНСТИТУЦИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ

Мы, многонациональный народ Российской Федерации,
соединенные общей судьбой на своей земле,
принимаем КОНСТИТУЦИЮ РОССИЙСКОЙ ФЕДЕРАЦИИ.

РАЗДЕЛ ПЕРВЫЙ

ГЛАВА 1.
ТЕСТОВАЯ ГЛАВА

Статья 1

Текст статьи.
"""

    units = parse_text(text)

    preamble = units[0]

    assert preamble.kind == "preamble"
    assert preamble.section is None
    assert preamble.chapter is None
    assert preamble.article is None
    assert preamble.part is None
    assert preamble.part_label is None

    assert preamble.text == (
        "Мы, многонациональный народ Российской Федерации,\n"
        "соединенные общей судьбой на своей земле,\n"
        "принимаем КОНСТИТУЦИЮ РОССИЙСКОЙ ФЕДЕРАЦИИ."
    )

    assert "Принята всенародным голосованием" not in preamble.text


def test_parse_transitional_units() -> None:
    text = """
РАЗДЕЛ ВТОРОЙ

ЗАКЛЮЧИТЕЛЬНЫЕ И ПЕРЕХОДНЫЕ ПОЛОЖЕНИЯ

1. Первый переходный пункт.
Продолжение первого пункта.
2. Второй переходный пункт.
"""

    units = parse_text(text)

    assert len(units) == 2

    first = units[0]
    second = units[1]

    assert first.kind == "transitional"
    assert first.section == "второй"
    assert first.chapter is None
    assert first.article is None
    assert first.part == 1
    assert first.part_label == "1"
    assert first.text == ("Первый переходный пункт.\nПродолжение первого пункта.")

    assert second.kind == "transitional"
    assert second.article is None
    assert second.part == 2
    assert second.part_label == "2"


def test_section_two_is_not_attached_to_article_137() -> None:
    text = """
РАЗДЕЛ ПЕРВЫЙ

ГЛАВА 9.
КОНСТИТУЦИОННЫЕ ПОПРАВКИ

Статья 137

1. Последняя статья раздела первого.

РАЗДЕЛ ВТОРОЙ

ЗАКЛЮЧИТЕЛЬНЫЕ И ПЕРЕХОДНЫЕ ПОЛОЖЕНИЯ

1. Первый переходный пункт.
"""

    units = parse_text(text)

    assert len(units) == 2

    article = units[0]
    transitional = units[1]

    assert article.article == "137"
    assert article.kind == "article"

    assert transitional.section == "второй"
    assert transitional.chapter is None
    assert transitional.article is None
    assert transitional.part == 1
    assert transitional.kind == "transitional"


def test_source_markers_are_removed() -> None:
    text = """
РАЗДЕЛ ПЕРВЫЙ

ГЛАВА 3.
ФЕДЕРАТИВНОЕ УСТРОЙСТВО

Статья 67.1 <*>

1. Российская Федерация является правопреемником Союза ССР. <*>
"""

    units = parse_text(text)

    assert len(units) == 1

    unit = units[0]

    assert unit.article == "67.1"
    assert "<*>" not in unit.text
    assert unit.text == ("Российская Федерация является правопреемником Союза ССР.")


def test_editorial_footnotes_are_skipped() -> None:
    text = """
РАЗДЕЛ ПЕРВЫЙ

ГЛАВА 3.
ФЕДЕРАТИВНОЕ УСТРОЙСТВО

Статья 71

Текст статьи.
--------------------------------
<18> Редакция пункта приведена в соответствии с законом.
<19> Еще одна редакционная сноска.

Статья 72

Следующая статья.
"""

    units = parse_text(text)

    assert len(units) == 2

    assert units[0].article == "71"
    assert units[0].text == "Текст статьи."

    assert units[1].article == "72"
    assert units[1].text == "Следующая статья."

    assert all("Редакция пункта" not in unit.text for unit in units)


def test_consultant_note_is_skipped_inside_article() -> None:
    text = """
РАЗДЕЛ ПЕРВЫЙ

ГЛАВА 4.
ПРЕЗИДЕНТ РОССИЙСКОЙ ФЕДЕРАЦИИ

Статья 81

2. Вторая часть.

КонсультантПлюс: примечание.
О возможности участия кандидата см. специальный закон.

3. Третья часть.
4. Четвертая часть.
"""

    units = parse_text(text)

    assert len(units) == 3

    assert [unit.part for unit in units] == [2, 3, 4]

    assert all("КонсультантПлюс" not in unit.text for unit in units)

    assert all("О возможности участия" not in unit.text for unit in units)

    assert units[1].text == "Третья часть."


def test_numeric_source_markers_are_removed() -> None:
    text = """
РАЗДЕЛ ПЕРВЫЙ

ГЛАВА 7.
СУДЕБНАЯ ВЛАСТЬ И ПРОКУРАТУРА <23>

Статья 96

1. Государственная Дума избирается сроком на пять лет <20>.
"""

    units = parse_text(text)

    assert len(units) == 1

    unit = units[0]

    assert unit.chapter_title == "СУДЕБНАЯ ВЛАСТЬ И ПРОКУРАТУРА"
    assert unit.text == "Государственная Дума избирается сроком на пять лет."

    assert "<20>" not in unit.text
    assert "<23>" not in unit.chapter_title


def test_source_marker_removal_does_not_leave_extra_spaces() -> None:
    result = _clean_source_line(
        "Москва, Санкт-Петербург, Севастополь <16> - города федерального значения;"
    )

    assert result == (
        "Москва, Санкт-Петербург, Севастополь - города федерального значения;"
    )
