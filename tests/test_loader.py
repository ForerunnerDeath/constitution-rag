from pathlib import Path

from app.ingest.loader import load_text, normalize_text


def test_normalize_special_characters() -> None:
    raw = (
        "\ufeffСтатья\xa067.1\u00ad\n"
        "“Российская Федерация” – государство — текст − пример"
    )

    result = normalize_text(raw)

    assert result == (
        'Статья 67.1\n"Российская Федерация" - государство - текст - пример'
    )


def test_normalize_whitespace() -> None:
    raw = "   Статья     1   \n\t1.   Российская\tФедерация   "

    result = normalize_text(raw)

    assert result == "Статья 1\n1. Российская Федерация"


def test_limit_consecutive_blank_lines() -> None:
    raw = "Статья 1\n\n\n\n\n1. Текст"

    result = normalize_text(raw)

    assert result == "Статья 1\n\n\n1. Текст"


def test_load_text(tmp_path: Path) -> None:
    path = tmp_path / "constitution.txt"
    path.write_text("  Статья\xa0  1  ", encoding="utf-8")

    result = load_text(path)

    assert result == "Статья 1"
