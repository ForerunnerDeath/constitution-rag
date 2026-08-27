import re
from pathlib import Path

_CHAR_TRANSLATION = str.maketrans(
    {
        "\xa0": " ",
        "\u00ad": "",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u00ab": '"',
        "\u00bb": '"',
    }
)

_HORIZONTAL_WHITESPACE_RE = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    text = text.lstrip("\ufeff")
    text = text.translate(_CHAR_TRANSLATION)

    normalized_lines: list[str] = []
    blank_lines = 0

    for raw_line in text.splitlines():
        line = _HORIZONTAL_WHITESPACE_RE.sub(" ", raw_line).strip()

        if not line:
            blank_lines += 1

            if blank_lines <= 2:
                normalized_lines.append("")

            continue

        blank_lines = 0
        normalized_lines.append(line)

    return "\n".join(normalized_lines).strip("\n")


def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return normalize_text(text)
