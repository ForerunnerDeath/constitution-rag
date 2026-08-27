import re
from dataclasses import dataclass

from app.ingest.parser import Unit


@dataclass(frozen=True)
class Chunk:
    id: str
    embed_text: str
    quote: str
    ref: str
    chapter: int | None
    chapter_title: str | None
    article: str | None
    part: int | None
    part_label: str | None
    kind: str


def _build_ref(*, kind: str, article: str | None, part_label: str | None) -> str:
    if kind == "preamble":
        return "Преамбула"

    if kind == "transitional":
        if part_label is None:
            return "Заключительные и переходные положения"

        return f"Заключительные и переходные положения, пункт {part_label}"

    if kind == "article":
        if article is None:
            raise ValueError("Article chunk must have article number")

        if part_label is None:
            return f"Статья {article}"

        if "-" in part_label:
            return f"Статья {article}, части {part_label}"

        return f"Статья {article}, часть {part_label}"

    raise ValueError(f"Unknown chunk kind: {kind}")


def _build_chunk_id(*, kind: str, article: str | None, part_label: str | None) -> str:
    if kind == "preamble":
        return "preamble"

    if kind == "transitional":
        if part_label is None:
            raise ValueError("Transitional chunk must have part label")

        return f"transitional-{part_label}"

    if kind == "article":
        if article is None:
            raise ValueError("Article chunk must have article number")

        chunk_id = f"art-{article}"

        if part_label is not None:
            chunk_id += f"-p-{part_label}"

        return chunk_id

    raise ValueError(f"Unknown chunk kind: {kind}")


def _build_embed_text(
    *, kind: str, chapter: int | None, chapter_title: str | None, ref: str, quote: str
) -> str:
    if kind == "article":
        if chapter is None or chapter_title is None:
            raise ValueError("Article chunk must have chapter metadata")

        prefix = f"Глава {chapter}. {chapter_title}. {ref}."

    elif kind == "preamble":
        prefix = "Преамбула Конституции Российской Федерации."

    elif kind == "transitional":
        prefix = f"{ref}."

    else:
        raise ValueError(f"Unknown chunk kind: {kind}")

    return f"{prefix} {quote}"


def _create_chunk(
    *,
    quote: str,
    chapter: int | None,
    chapter_title: str | None,
    article: str | None,
    part: int | None,
    part_label: str | None,
    kind: str,
) -> Chunk:
    ref = _build_ref(kind=kind, article=article, part_label=part_label)

    chunk_id = _build_chunk_id(kind=kind, article=article, part_label=part_label)

    embed_text = _build_embed_text(
        kind=kind, chapter=chapter, chapter_title=chapter_title, ref=ref, quote=quote
    )

    return Chunk(
        id=chunk_id,
        embed_text=embed_text,
        quote=quote,
        ref=ref,
        chapter=chapter,
        chapter_title=chapter_title,
        article=article,
        part=part,
        part_label=part_label,
        kind=kind,
    )


def _group_short_units(units: list[Unit], min_chunk_chars: int) -> list[list[Unit]]:
    if min_chunk_chars <= 0:
        raise ValueError("min_chunk_chars must be greater than zero")

    groups: list[list[Unit]] = []
    index = 0

    while index < len(units):
        unit = units[index]
        group = [unit]

        if (
            unit.kind == "article"
            and unit.article is not None
            and unit.part_label is not None
            and len(unit.text) < min_chunk_chars
        ):
            current_length = len(unit.text)
            next_index = index + 1

            while current_length < min_chunk_chars and next_index < len(units):
                next_unit = units[next_index]

                if (
                    next_unit.kind != "article"
                    or next_unit.article != unit.article
                    or next_unit.part_label is None
                ):
                    break

                group.append(next_unit)

                current_length += 1 + len(next_unit.text)
                next_index += 1

        groups.append(group)
        index += len(group)

    return groups


def _create_chunk_from_group(group: list[Unit]) -> Chunk:
    if not group:
        raise ValueError("Unit group must not be empty")

    first = group[0]

    if len(group) == 1:
        part = first.part
        part_label = first.part_label

    else:
        if first.kind != "article" or first.article is None:
            raise ValueError("Only article parts can be merged into one chunk")

        for unit in group[1:]:
            if (
                unit.kind != first.kind
                or unit.article != first.article
                or unit.chapter != first.chapter
                or unit.chapter_title != first.chapter_title
            ):
                raise ValueError("Merged units must belong to the same article")

        first_label = group[0].part_label
        last_label = group[-1].part_label

        if first_label is None or last_label is None:
            raise ValueError("Merged article parts must have part labels")

        part = None
        part_label = f"{first_label}-{last_label}"

    quote = "\n".join(unit.text for unit in group)

    return _create_chunk(
        quote=quote,
        chapter=first.chapter,
        chapter_title=first.chapter_title,
        article=first.article,
        part=part,
        part_label=part_label,
        kind=first.kind,
    )


def chunk_units(
    units: list[Unit], min_chunk_chars: int = 100, max_chunk_chars: int = 900
) -> list[Chunk]:
    if min_chunk_chars <= 0:
        raise ValueError("min_chunk_chars must be greater than zero")

    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be greater than zero")

    if min_chunk_chars > max_chunk_chars:
        raise ValueError("min_chunk_chars must not exceed max_chunk_chars")

    groups = _group_short_units(units, min_chunk_chars=min_chunk_chars)

    chunks: list[Chunk] = []

    for group in groups:
        chunk = _create_chunk_from_group(group)

        chunks.extend(_split_chunk(chunk, max_chunk_chars=max_chunk_chars))

    return chunks


_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+(?=[А-ЯЁ"«])')


def _split_into_segments(text: str) -> list[str]:
    segments: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        sentences = _SENTENCE_BOUNDARY_RE.split(line)

        segments.extend(sentence.strip() for sentence in sentences if sentence.strip())

    return segments


def _split_long_text(text: str, max_chunk_chars: int) -> list[str]:
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be greater than zero")

    if len(text) <= max_chunk_chars:
        return [text]

    segments = _split_into_segments(text)

    if not segments:
        return [text]

    chunks: list[str] = []
    current: list[str] = []

    for segment in segments:
        if not current:
            current.append(segment)
            continue

        candidate = "\n".join([*current, segment])

        if len(candidate) <= max_chunk_chars:
            current.append(segment)
            continue

        chunks.append("\n".join(current))

        overlap = current[-1]

        overlap_candidate = "\n".join([overlap, segment])

        if len(overlap_candidate) <= max_chunk_chars:
            current = [overlap, segment]
        else:
            current = [segment]

    if current:
        chunks.append("\n".join(current))

    return chunks


def _split_chunk(chunk: Chunk, max_chunk_chars: int) -> list[Chunk]:
    fragments = _split_long_text(chunk.quote, max_chunk_chars=max_chunk_chars)

    if len(fragments) == 1:
        return [chunk]

    result: list[Chunk] = []

    for index, fragment in enumerate(fragments, start=1):
        result.append(
            Chunk(
                id=f"{chunk.id}-c-{index}",
                embed_text=_build_embed_text(
                    kind=chunk.kind,
                    chapter=chunk.chapter,
                    chapter_title=chunk.chapter_title,
                    ref=chunk.ref,
                    quote=fragment,
                ),
                quote=fragment,
                ref=chunk.ref,
                chapter=chunk.chapter,
                chapter_title=chunk.chapter_title,
                article=chunk.article,
                part=chunk.part,
                part_label=chunk.part_label,
                kind=chunk.kind,
            )
        )

    return result
