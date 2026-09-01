from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import get_settings
from app.llm.client import OpenAICompatibleLLMClient
from app.llm.rag import PROMPT_VERSION, RAGResult, RAGService
from app.search.embedder import Embedder
from app.search.lexical import LexicalIndex
from app.search.retriever import Retriever
from app.search.store import ChromaStore

DEFAULT_DATASET_PATH = Path("eval/generation_dev.csv")
DEFAULT_OUTPUT_PATH = Path("eval/generation_dev_results.json")


@dataclass(frozen=True)
class GenerationQuestion:
    id: str
    question: str
    category: str


@dataclass(frozen=True)
class CitationCheck:
    inline_citations: tuple[str, ...]
    valid_citations: tuple[str, ...]
    invalid_citations: tuple[str, ...]
    citation_present: bool
    all_citations_valid: bool


@dataclass(frozen=True)
class GenerationHit:
    id: str
    ref: str
    quote: str
    score: float | None


@dataclass(frozen=True)
class GenerationEvaluationResult:
    id: str
    question: str
    category: str

    found: bool
    answer: str | None
    message: str | None

    llm_used: bool
    llm_called: bool
    llm_ms: float | None

    hits: tuple[GenerationHit, ...]

    inline_citations: tuple[str, ...]
    valid_citations: tuple[str, ...]
    invalid_citations: tuple[str, ...]
    citation_present: bool
    all_citations_valid: bool


@dataclass(frozen=True)
class GenerationMetrics:
    questions: int
    generated_answers: int

    answers_with_citations: int
    answers_with_only_valid_citations: int

    citation_refs_total: int
    valid_citation_refs: int

    citation_presence_rate: float
    citation_validity_answer_rate: float
    citation_validity_reference_rate: float


@dataclass(frozen=True)
class GenerationEvaluationRun:
    results: tuple[GenerationEvaluationResult, ...]
    metrics: GenerationMetrics


@dataclass(frozen=True)
class GenerationReport:
    dataset: str
    llm_model: str
    embedding_model: str
    min_score: float
    k: int
    prompt_version: str
    run: GenerationEvaluationRun


@dataclass(frozen=True)
class GenerationComponents:
    rag_service: RAGService
    llm_client: OpenAICompatibleLLMClient


_BRACKET_CONTENT_RE = re.compile(r"\[([^\[\]\r\n]+)\]")

_CITATION_DASH_RE = re.compile(r"[‐-‒–—−]")


def normalize_citation(value: str) -> str:
    return _CITATION_DASH_RE.sub("-", value).strip()


def extract_inline_citations(answer: str) -> tuple[str, ...]:
    citations: list[str] = []

    for match in _BRACKET_CONTENT_RE.finditer(answer):
        for raw_value in match.group(1).split(";"):
            value = raw_value.strip()

            if (
                value == "Преамбула"
                or value.startswith("Статья ")
                or value.startswith("Заключительные и переходные положения")
            ):
                citations.append(value)

    return tuple(citations)


def check_citations(
    answer: str,
    available_refs: Iterable[str],
) -> CitationCheck:
    inline_citations = extract_inline_citations(answer)

    available = {normalize_citation(ref) for ref in available_refs}

    valid_citations = tuple(
        citation
        for citation in inline_citations
        if normalize_citation(citation) in available
    )

    invalid_citations = tuple(
        citation
        for citation in inline_citations
        if normalize_citation(citation) not in available
    )

    return CitationCheck(
        inline_citations=inline_citations,
        valid_citations=valid_citations,
        invalid_citations=invalid_citations,
        citation_present=bool(inline_citations),
        all_citations_valid=not invalid_citations,
    )


def build_evaluation_result(
    question: GenerationQuestion,
    rag_result: RAGResult,
) -> GenerationEvaluationResult:
    hits = tuple(
        GenerationHit(
            id=hit.id,
            ref=hit.ref,
            quote=hit.quote,
            score=hit.score,
        )
        for hit in rag_result.hits
    )

    citation_check = check_citations(
        answer=rag_result.answer or "",
        available_refs=(hit.ref for hit in rag_result.hits),
    )

    return GenerationEvaluationResult(
        id=question.id,
        question=question.question,
        category=question.category,
        found=rag_result.found,
        answer=rag_result.answer,
        message=rag_result.message,
        llm_used=rag_result.llm_used,
        llm_called=rag_result.llm_called,
        llm_ms=rag_result.llm_ms,
        hits=hits,
        inline_citations=citation_check.inline_citations,
        valid_citations=citation_check.valid_citations,
        invalid_citations=citation_check.invalid_citations,
        citation_present=citation_check.citation_present,
        all_citations_valid=citation_check.all_citations_valid,
    )


def load_generation_questions(path: Path) -> list[GenerationQuestion]:
    rows: list[GenerationQuestion] = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required_columns = {"id", "question", "category"}

        if reader.fieldnames is None:
            raise ValueError("CSV file has no header")

        if not required_columns.issubset(reader.fieldnames):
            raise ValueError("CSV must contain id, question and category columns")

        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Invalid CSV row {row_number}: too many columns")

            question_id = row["id"]
            question = row["question"]
            category = row["category"]

            if question_id is None or question is None or category is None:
                raise ValueError(f"Invalid CSV row {row_number}: missing field")

            question_id = question_id.strip()
            question = question.strip()
            category = category.strip()

            if not question_id or not question or not category:
                raise ValueError(f"Invalid CSV row {row_number}: empty field")

            rows.append(
                GenerationQuestion(
                    id=question_id,
                    question=question,
                    category=category,
                )
            )

    return rows


def calculate_metrics(
    results: list[GenerationEvaluationResult],
) -> GenerationMetrics:
    questions = len(results)

    generated = [result for result in results if result.answer is not None]
    generated_answers = len(generated)

    answers_with_citations = sum(result.citation_present for result in generated)

    answers_with_only_valid_citations = sum(
        result.citation_present and result.all_citations_valid for result in generated
    )

    citation_refs_total = sum(len(result.inline_citations) for result in generated)

    valid_citation_refs = sum(len(result.valid_citations) for result in generated)

    citation_presence_rate = (
        answers_with_citations / generated_answers if generated_answers else 0.0
    )

    citation_validity_answer_rate = (
        answers_with_only_valid_citations / generated_answers
        if generated_answers
        else 0.0
    )

    citation_validity_reference_rate = (
        valid_citation_refs / citation_refs_total if citation_refs_total else 0.0
    )

    return GenerationMetrics(
        questions=questions,
        generated_answers=generated_answers,
        answers_with_citations=answers_with_citations,
        answers_with_only_valid_citations=answers_with_only_valid_citations,
        citation_refs_total=citation_refs_total,
        valid_citation_refs=valid_citation_refs,
        citation_presence_rate=citation_presence_rate,
        citation_validity_answer_rate=citation_validity_answer_rate,
        citation_validity_reference_rate=citation_validity_reference_rate,
    )


async def evaluate_generation(
    questions: list[GenerationQuestion],
    rag_service: RAGService,
    *,
    k: int = 5,
) -> GenerationEvaluationRun:
    results: list[GenerationEvaluationResult] = []

    for question in questions:
        rag_result = await rag_service.ask(
            question.question,
            k=k,
        )

        results.append(
            build_evaluation_result(
                question,
                rag_result,
            )
        )

    return GenerationEvaluationRun(
        results=tuple(results),
        metrics=calculate_metrics(results),
    )


def write_report(
    report: GenerationReport,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            asdict(report),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


async def run_generation_evaluation(
    *,
    dataset_path: Path,
    output_path: Path,
    k: int = 5,
) -> GenerationReport:
    if not 1 <= k <= 20:
        raise ValueError("k must be between 1 and 20")

    questions = load_generation_questions(dataset_path)
    components = build_generation_components()
    settings = get_settings()

    try:
        run = await evaluate_generation(
            questions,
            components.rag_service,
            k=k,
        )

        report = GenerationReport(
            dataset=str(dataset_path),
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model,
            min_score=settings.min_score,
            k=k,
            prompt_version=PROMPT_VERSION,
            run=run,
        )

        write_report(report, output_path)

        return report

    finally:
        await components.llm_client.close()


def build_generation_components() -> GenerationComponents:
    settings = get_settings()

    if not settings.llm_enabled:
        raise RuntimeError("Generation evaluation requires llm_enabled=true")

    if not settings.llm_model.strip():
        raise RuntimeError("Generation evaluation requires llm_model")

    embedder = Embedder(settings.embedding_model)

    store = ChromaStore(
        path=settings.chroma_path,
        collection_name=settings.chroma_collection,
    )

    store.ensure_embedding_compatibility(
        model_name=embedder.model_name,
        dim=embedder.dim,
    )

    lexical_index = LexicalIndex(store.get_all())

    retriever = Retriever(
        embedder=embedder,
        store=store,
        lexical_index=lexical_index,
        min_score=settings.min_score,
    )

    llm_client = OpenAICompatibleLLMClient(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        max_tokens=settings.llm_max_tokens,
        timeout_seconds=settings.llm_timeout_seconds,
    )

    rag_service = RAGService(
        retriever=retriever,
        llm_client=llm_client,
    )

    return GenerationComponents(
        rag_service=rag_service,
        llm_client=llm_client,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Constitution RAG generation quality."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Generation evaluation dataset CSV path.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSON report output path.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of retrieval hits passed to RAG.",
    )

    args = parser.parse_args(argv)

    report = asyncio.run(
        run_generation_evaluation(
            dataset_path=args.dataset,
            output_path=args.output,
            k=args.k,
        )
    )

    metrics = report.run.metrics

    print("Generation evaluation configuration:")
    print(f"Dataset:         {report.dataset}")
    print(f"Output:          {args.output}")
    print(f"Questions:       {metrics.questions}")
    print(f"Embedding model: {report.embedding_model}")
    print(f"LLM model:       {report.llm_model}")
    print(f"Min score:       {report.min_score:.3f}")
    print(f"k:               {report.k}")
    print(f"Prompt version:  {report.prompt_version}")

    print()
    print("Automatic metrics:")
    print(f"Generated answers:          {metrics.generated_answers}")
    print(f"Citation presence:         {metrics.citation_presence_rate:.3f}")
    print(f"Citation validity/answer:  {metrics.citation_validity_answer_rate:.3f}")
    print(f"Citation validity/ref:     {metrics.citation_validity_reference_rate:.3f}")

    print()
    print(
        "Groundedness is not scored automatically; "
        "review answer against hits[].quote in the JSON report."
    )


if __name__ == "__main__":
    main()
