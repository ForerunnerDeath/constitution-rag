import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    section: str | None
    chapter: int | None
    chapter_title: str | None
    article: str | None
    part: int | None
    part_label: str | None
    text: str
    kind: str


_SECTION_RE = re.compile(
    r"^РАЗДЕЛ\s+(ПЕРВЫЙ|ВТОРОЙ)$",
    re.IGNORECASE,
)

_CHAPTER_RE = re.compile(
    r"^ГЛАВА\s+(\d+)\.\s*(.*)$",
    re.IGNORECASE,
)

_ARTICLE_RE = re.compile(
    r"^Статья\s+(\d+(?:\.\d+)?)\s*(?:<\*>)?$",
    re.IGNORECASE,
)

_PART_RE = re.compile(r"^(\d+(?:\.\d+)*)\.\s*(.*)$")

_SOURCE_MARKER_RE = re.compile(r"<(?:\*|\d+)>")


def _clean_source_line(line: str) -> str:
    line = _SOURCE_MARKER_RE.sub("", line)
    line = " ".join(line.split())
    line = re.sub(r"\s+([.,;:])", r"\1", line)
    return line.strip()


def _parse_section(line: str) -> str | None:
    match = _SECTION_RE.match(line)

    if match is None:
        return None

    return match.group(1).lower()


def _parse_chapter(line: str) -> tuple[int, str | None] | None:
    match = _CHAPTER_RE.match(line)

    if match is None:
        return None

    chapter = int(match.group(1))
    title = match.group(2).strip() or None

    return chapter, title


def _parse_article(line: str) -> str | None:
    match = _ARTICLE_RE.match(line)

    if match is None:
        return None

    return match.group(1)


def _parse_part(
    line: str,
) -> tuple[int | None, str, str] | None:
    match = _PART_RE.match(line)

    if match is None:
        return None

    part_label = match.group(1)
    content = match.group(2).strip()

    part = int(part_label) if part_label.isdigit() else None

    return part, part_label, content


def parse_text(text: str) -> list[Unit]:
    units: list[Unit] = []

    current_section: str | None = None
    current_chapter: int | None = None
    current_chapter_title: str | None = None
    current_article: str | None = None
    current_part: int | None = None
    current_part_label: str | None = None
    current_kind: str | None = None

    buffer: list[str] = []
    awaiting_chapter_title = False
    skipping_footnotes = False
    skipping_consultant_note = False

    def flush() -> None:
        nonlocal buffer

        if current_kind is None:
            buffer = []
            return

        content = "\n".join(buffer).strip()

        if not content:
            buffer = []
            return

        units.append(
            Unit(
                section=current_section,
                chapter=current_chapter,
                chapter_title=current_chapter_title,
                article=current_article,
                part=current_part,
                part_label=current_part_label,
                text=content,
                kind=current_kind,
            )
        )

        buffer = []

    for raw_line in text.splitlines():
        line = _clean_source_line(raw_line)

        if not line:
            continue

        if line == "--------------------------------":
            skipping_footnotes = True
            continue

        if skipping_footnotes:
            if (
                _parse_section(line) is not None
                or _parse_chapter(line) is not None
                or _parse_article(line) is not None
            ):
                skipping_footnotes = False
            else:
                continue

        if line == "КонсультантПлюс: примечание.":
            skipping_consultant_note = True
            continue

        if skipping_consultant_note:
            if (
                _parse_section(line) is not None
                or _parse_chapter(line) is not None
                or _parse_article(line) is not None
                or _parse_part(line) is not None
            ):
                skipping_consultant_note = False
            else:
                continue

        section = _parse_section(line)

        if section is not None:
            flush()

            current_section = section
            current_chapter = None
            current_chapter_title = None
            current_article = None
            current_part = None
            current_part_label = None
            current_kind = None
            awaiting_chapter_title = False

            continue

        if current_section is None:
            if current_kind == "preamble":
                buffer.append(line)
                continue

            if line.startswith("Мы, многонациональный народ Российской Федерации"):
                current_kind = "preamble"
                buffer.append(line)

            continue

        if current_section == "второй":
            part = _parse_part(line)

            if part is not None:
                flush()

                current_part, current_part_label, content = part
                current_kind = "transitional"

                if content:
                    buffer.append(content)

                continue

            if current_kind == "transitional":
                buffer.append(line)

            continue

        # Пока реализуем только раздел I.
        if current_section != "первый":
            continue

        chapter = _parse_chapter(line)

        if chapter is not None:
            flush()

            current_chapter, current_chapter_title = chapter
            current_article = None
            current_part = None
            current_part_label = None
            current_kind = None

            awaiting_chapter_title = current_chapter_title is None

            continue

        if awaiting_chapter_title:
            current_chapter_title = line
            awaiting_chapter_title = False
            continue

        article = _parse_article(line)

        if article is not None:
            flush()

            current_article = article
            current_part = None
            current_part_label = None
            current_kind = "article"

            continue

        if current_article is None:
            continue

        part = _parse_part(line)

        if part is not None:
            flush()

            current_part, current_part_label, content = part
            current_kind = "article"

            if content:
                buffer.append(content)

            continue

        buffer.append(line)

    flush()

    return units
