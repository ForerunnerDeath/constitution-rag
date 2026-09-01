import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.rag import RAGResult
from app.search.store import Hit
from scripts.evaluate_generation import (
    GenerationComponents,
    GenerationEvaluationRun,
    GenerationMetrics,
    GenerationQuestion,
    build_evaluation_result,
    build_generation_components,
    calculate_metrics,
    check_citations,
    evaluate_generation,
    extract_inline_citations,
    load_generation_questions,
    run_generation_evaluation,
)


def test_load_generation_questions_reads_valid_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "generation.csv"
    dataset.write_text(
        "id,question,category\n"
        "gen-001,Какова форма правления РФ?,single_fact\n"
        'gen-002,"Какие права гарантирует Конституция?",synthesis\n',
        encoding="utf-8",
    )

    questions = load_generation_questions(dataset)

    assert questions == [
        GenerationQuestion(
            id="gen-001",
            question="Какова форма правления РФ?",
            category="single_fact",
        ),
        GenerationQuestion(
            id="gen-002",
            question="Какие права гарантирует Конституция?",
            category="synthesis",
        ),
    ]


def test_load_generation_questions_rejects_missing_required_column(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "generation.csv"
    dataset.write_text(
        "id,question\ngen-001,Какова форма правления РФ?\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="CSV must contain id, question and category columns",
    ):
        load_generation_questions(dataset)


def test_load_generation_questions_rejects_empty_field(tmp_path: Path) -> None:
    dataset = tmp_path / "generation.csv"
    dataset.write_text(
        "id,question,category\ngen-001,,single_fact\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid CSV row 2: empty field",
    ):
        load_generation_questions(dataset)


def test_load_generation_questions_rejects_missing_field(tmp_path: Path) -> None:
    dataset = tmp_path / "generation.csv"
    dataset.write_text(
        "id,question,category\ngen-001,Какова форма правления РФ?\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid CSV row 2: missing field",
    ):
        load_generation_questions(dataset)


def test_load_generation_questions_rejects_extra_field(tmp_path: Path) -> None:
    dataset = tmp_path / "generation.csv"
    dataset.write_text(
        "id,question,category\n"
        "gen-001,Какова форма правления РФ?,single_fact,unexpected\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid CSV row 2: too many columns",
    ):
        load_generation_questions(dataset)


def test_extract_inline_citations_supports_real_source_formats() -> None:
    answer = (
        "Первое утверждение [Статья 3, часть 1]. "
        "Второе [Статья 5, части 1-2]. "
        "Также [Преамбула]. "
        "Служебная метка [Документ 1]."
    )

    citations = extract_inline_citations(answer)

    assert citations == (
        "Статья 3, часть 1",
        "Статья 5, части 1-2",
        "Преамбула",
    )


def test_check_citations_accepts_refs_from_retrieved_context() -> None:
    result = check_citations(
        answer=("Народ является источником власти [Статья 3, часть 1]."),
        available_refs=[
            "Статья 3, часть 1",
            "Статья 3, часть 2",
        ],
    )

    assert result.citation_present is True
    assert result.all_citations_valid is True
    assert result.valid_citations == ("Статья 3, часть 1",)
    assert result.invalid_citations == ()


def test_check_citations_detects_hallucinated_ref() -> None:
    result = check_citations(
        answer=(
            "Народ является источником власти "
            "[Статья 3, часть 1]. "
            "Дополнительное утверждение "
            "[Статья 999, часть 7]."
        ),
        available_refs=[
            "Статья 3, часть 1",
            "Статья 3, часть 2",
        ],
    )

    assert result.citation_present is True
    assert result.all_citations_valid is False
    assert result.valid_citations == ("Статья 3, часть 1",)
    assert result.invalid_citations == ("Статья 999, часть 7",)


def test_check_citations_reports_missing_citation() -> None:
    result = check_citations(
        answer="Народ является источником власти.",
        available_refs=["Статья 3, часть 1"],
    )

    assert result.citation_present is False
    assert result.inline_citations == ()
    assert result.valid_citations == ()
    assert result.invalid_citations == ()


def test_build_evaluation_result_preserves_generation_and_context() -> None:
    question = GenerationQuestion(
        id="gen-001",
        question="Кто является источником власти?",
        category="single_fact",
    )
    hits = [
        Hit(
            id="art-3-p-1",
            quote="Носителем суверенитета и единственным источником власти...",
            ref="Статья 3, часть 1",
            article="3",
            part=1,
            part_label="1",
            score=0.91,
        ),
        Hit(
            id="art-3-p-2",
            quote="Народ осуществляет свою власть непосредственно...",
            ref="Статья 3, часть 2",
            article="3",
            part=2,
            part_label="2",
            score=0.88,
        ),
    ]
    rag_result = RAGResult(
        found=True,
        answer=(
            "Источником власти является многонациональный народ [Статья 3, часть 1]."
        ),
        message=None,
        hits=hits,
        llm_used=True,
        llm_called=True,
        llm_ms=125.0,
    )

    result = build_evaluation_result(question, rag_result)

    assert result.id == "gen-001"
    assert result.category == "single_fact"
    assert result.answer == rag_result.answer
    assert result.llm_used is True
    assert result.llm_called is True
    assert result.llm_ms == 125.0

    assert len(result.hits) == 2
    assert result.hits[0].ref == "Статья 3, часть 1"
    assert result.hits[0].quote == hits[0].quote
    assert result.hits[0].score == 0.91

    assert result.citation_present is True
    assert result.all_citations_valid is True
    assert result.valid_citations == ("Статья 3, часть 1",)


def test_build_evaluation_result_preserves_llm_not_found_context() -> None:
    question = GenerationQuestion(
        id="gen-020",
        question="Какие документы нужны для альтернативной службы?",
        category="insufficient_context",
    )
    hits = [
        Hit(
            id="art-59-p-3",
            quote="Гражданин Российской Федерации в случае...",
            ref="Статья 59, часть 3",
            article="59",
            part=3,
            part_label="3",
            score=0.87,
        )
    ]
    rag_result = RAGResult(
        found=False,
        answer=None,
        message="В тексте Конституции прямого ответа не нашлось.",
        hits=hits,
        llm_used=True,
        llm_called=True,
        llm_ms=100.0,
    )

    result = build_evaluation_result(question, rag_result)

    assert result.found is False
    assert result.answer is None
    assert result.llm_used is True
    assert result.llm_called is True

    assert len(result.hits) == 1
    assert result.hits[0].ref == "Статья 59, часть 3"

    assert result.citation_present is False
    assert result.inline_citations == ()
    assert result.invalid_citations == ()


def test_calculate_metrics_uses_only_generated_answers_for_citation_rates() -> None:
    base_question = GenerationQuestion(
        id="gen-001",
        question="Вопрос",
        category="single_fact",
    )

    valid = build_evaluation_result(
        base_question,
        RAGResult(
            found=True,
            answer="Ответ [Статья 1, часть 1].",
            message=None,
            hits=[
                Hit(
                    id="art-1-p-1",
                    quote="Текст",
                    ref="Статья 1, часть 1",
                    article="1",
                    part=1,
                    part_label="1",
                    score=0.9,
                )
            ],
            llm_used=True,
            llm_called=True,
        ),
    )

    invalid = build_evaluation_result(
        GenerationQuestion(
            id="gen-002",
            question="Вопрос 2",
            category="single_fact",
        ),
        RAGResult(
            found=True,
            answer=("Ответ [Статья 1, часть 1] [Статья 999, часть 7]."),
            message=None,
            hits=[
                Hit(
                    id="art-1-p-1",
                    quote="Текст",
                    ref="Статья 1, часть 1",
                    article="1",
                    part=1,
                    part_label="1",
                    score=0.9,
                )
            ],
            llm_used=True,
            llm_called=True,
        ),
    )

    without_citation = build_evaluation_result(
        GenerationQuestion(
            id="gen-003",
            question="Вопрос 3",
            category="single_fact",
        ),
        RAGResult(
            found=True,
            answer="Ответ без ссылки.",
            message=None,
            hits=[],
            llm_used=True,
            llm_called=True,
        ),
    )

    not_found = build_evaluation_result(
        GenerationQuestion(
            id="gen-004",
            question="Вопрос 4",
            category="insufficient_context",
        ),
        RAGResult(
            found=False,
            answer=None,
            message="NOT FOUND",
            hits=[],
            llm_used=True,
            llm_called=True,
        ),
    )

    metrics = calculate_metrics([valid, invalid, without_citation, not_found])

    assert metrics.questions == 4
    assert metrics.generated_answers == 3

    assert metrics.answers_with_citations == 2
    assert metrics.answers_with_only_valid_citations == 1

    assert metrics.citation_refs_total == 3
    assert metrics.valid_citation_refs == 2

    assert metrics.citation_presence_rate == pytest.approx(2 / 3)
    assert metrics.citation_validity_answer_rate == pytest.approx(1 / 3)
    assert metrics.citation_validity_reference_rate == pytest.approx(2 / 3)


def test_calculate_metrics_handles_no_generated_answers() -> None:
    result = build_evaluation_result(
        GenerationQuestion(
            id="gen-020",
            question="Недостаточно контекста?",
            category="insufficient_context",
        ),
        RAGResult(
            found=False,
            answer=None,
            message="NOT FOUND",
            hits=[],
            llm_used=True,
            llm_called=True,
        ),
    )

    metrics = calculate_metrics([result])

    assert metrics.questions == 1
    assert metrics.generated_answers == 0

    assert metrics.citation_presence_rate == 0.0
    assert metrics.citation_validity_answer_rate == 0.0
    assert metrics.citation_validity_reference_rate == 0.0


@pytest.mark.asyncio
async def test_evaluate_generation_runs_rag_for_every_question() -> None:
    questions = [
        GenerationQuestion(
            id="gen-001",
            question="Первый вопрос",
            category="single_fact",
        ),
        GenerationQuestion(
            id="gen-002",
            question="Второй вопрос",
            category="insufficient_context",
        ),
    ]

    first_hit = Hit(
        id="art-1-p-1",
        quote="Первый текст",
        ref="Статья 1, часть 1",
        article="1",
        part=1,
        part_label="1",
        score=0.9,
    )

    rag_service = MagicMock()
    rag_service.ask = AsyncMock(
        side_effect=[
            RAGResult(
                found=True,
                answer="Первый ответ [Статья 1, часть 1].",
                message=None,
                hits=[first_hit],
                llm_used=True,
                llm_called=True,
            ),
            RAGResult(
                found=False,
                answer=None,
                message="В тексте Конституции прямого ответа не нашлось.",
                hits=[],
                llm_used=True,
                llm_called=True,
            ),
        ]
    )

    run = await evaluate_generation(
        questions,
        rag_service,
        k=5,
    )

    assert rag_service.ask.await_count == 2
    assert rag_service.ask.await_args_list[0].args == ("Первый вопрос",)
    assert rag_service.ask.await_args_list[0].kwargs == {"k": 5}
    assert rag_service.ask.await_args_list[1].args == ("Второй вопрос",)
    assert rag_service.ask.await_args_list[1].kwargs == {"k": 5}

    assert len(run.results) == 2

    assert run.results[0].id == "gen-001"
    assert run.results[0].answer == ("Первый ответ [Статья 1, часть 1].")
    assert run.results[0].citation_present is True

    assert run.results[1].id == "gen-002"
    assert run.results[1].answer is None
    assert run.results[1].found is False

    assert run.metrics.questions == 2
    assert run.metrics.generated_answers == 1
    assert run.metrics.answers_with_citations == 1
    assert run.metrics.citation_presence_rate == 1.0


def test_build_generation_components_requires_enabled_llm() -> None:
    settings = SimpleNamespace(
        llm_enabled=False,
        llm_model="test-model",
    )

    with patch(
        "scripts.evaluate_generation.get_settings",
        return_value=settings,
    ):
        with pytest.raises(
            RuntimeError,
            match="Generation evaluation requires llm_enabled=true",
        ):
            build_generation_components()


def test_build_generation_components_requires_llm_model() -> None:
    settings = SimpleNamespace(
        llm_enabled=True,
        llm_model="   ",
    )

    with patch(
        "scripts.evaluate_generation.get_settings",
        return_value=settings,
    ):
        with pytest.raises(
            RuntimeError,
            match="Generation evaluation requires llm_model",
        ):
            build_generation_components()


def test_build_generation_components_uses_project_configuration() -> None:
    settings = SimpleNamespace(
        llm_enabled=True,
        llm_model="test-model",
        llm_api_key="test-key",
        llm_base_url="http://localhost:1234/v1",
        llm_max_tokens=512,
        llm_timeout_seconds=20.0,
        embedding_model="intfloat/multilingual-e5-small",
        chroma_path="data/chroma",
        chroma_collection="constitution_e5_small",
        min_score=0.833,
    )

    fake_embedder = MagicMock()
    fake_embedder.model_name = settings.embedding_model
    fake_embedder.dim = 384

    fake_store = MagicMock()
    fake_corpus = [MagicMock()]
    fake_store.get_all.return_value = fake_corpus

    fake_lexical_index = MagicMock()
    fake_retriever = MagicMock()
    fake_llm_client = MagicMock()
    fake_rag_service = MagicMock()

    with (
        patch(
            "scripts.evaluate_generation.get_settings",
            return_value=settings,
        ),
        patch(
            "scripts.evaluate_generation.Embedder",
            return_value=fake_embedder,
        ),
        patch(
            "scripts.evaluate_generation.ChromaStore",
            return_value=fake_store,
        ),
        patch(
            "scripts.evaluate_generation.LexicalIndex",
            return_value=fake_lexical_index,
        ),
        patch(
            "scripts.evaluate_generation.Retriever",
            return_value=fake_retriever,
        ),
        patch(
            "scripts.evaluate_generation.OpenAICompatibleLLMClient",
            return_value=fake_llm_client,
        ) as llm_client_class,
        patch(
            "scripts.evaluate_generation.RAGService",
            return_value=fake_rag_service,
        ) as rag_service_class,
    ):
        components = build_generation_components()

    fake_store.ensure_embedding_compatibility.assert_called_once_with(
        model_name=settings.embedding_model,
        dim=384,
    )

    fake_store.get_all.assert_called_once_with()

    llm_client_class.assert_called_once_with(
        model="test-model",
        api_key="test-key",
        base_url="http://localhost:1234/v1",
        max_tokens=512,
        timeout_seconds=20.0,
    )

    rag_service_class.assert_called_once_with(
        retriever=fake_retriever,
        llm_client=fake_llm_client,
    )

    assert components.rag_service is fake_rag_service
    assert components.llm_client is fake_llm_client


@pytest.mark.asyncio
async def test_run_generation_evaluation_writes_report_and_closes_client(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "generation.csv"
    dataset.write_text(
        "id,question,category\ngen-001,Вопрос?,single_fact\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    fake_llm_client = MagicMock()
    fake_llm_client.close = AsyncMock()

    fake_components = GenerationComponents(
        rag_service=MagicMock(),
        llm_client=fake_llm_client,
    )

    fake_run = GenerationEvaluationRun(
        results=(),
        metrics=GenerationMetrics(
            questions=1,
            generated_answers=1,
            answers_with_citations=1,
            answers_with_only_valid_citations=1,
            citation_refs_total=1,
            valid_citation_refs=1,
            citation_presence_rate=1.0,
            citation_validity_answer_rate=1.0,
            citation_validity_reference_rate=1.0,
        ),
    )

    settings = SimpleNamespace(
        llm_model="test-model",
        embedding_model="test-embedding",
        min_score=0.833,
    )

    with (
        patch(
            "scripts.evaluate_generation.build_generation_components",
            return_value=fake_components,
        ),
        patch(
            "scripts.evaluate_generation.get_settings",
            return_value=settings,
        ),
        patch(
            "scripts.evaluate_generation.evaluate_generation",
            new=AsyncMock(return_value=fake_run),
        ),
    ):
        report = await run_generation_evaluation(
            dataset_path=dataset,
            output_path=output,
            k=5,
        )

    fake_llm_client.close.assert_awaited_once_with()

    assert report.llm_model == "test-model"
    assert report.k == 5
    assert output.exists()

    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["llm_model"] == "test-model"
    assert saved["k"] == 5
    assert saved["run"]["metrics"]["citation_presence_rate"] == 1.0


@pytest.mark.asyncio
async def test_run_generation_evaluation_closes_client_on_failure(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "generation.csv"
    dataset.write_text(
        "id,question,category\ngen-001,Вопрос?,single_fact\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    fake_llm_client = MagicMock()
    fake_llm_client.close = AsyncMock()

    fake_components = GenerationComponents(
        rag_service=MagicMock(),
        llm_client=fake_llm_client,
    )

    settings = SimpleNamespace(
        llm_model="test-model",
        embedding_model="test-embedding",
        min_score=0.833,
    )

    with (
        patch(
            "scripts.evaluate_generation.build_generation_components",
            return_value=fake_components,
        ),
        patch(
            "scripts.evaluate_generation.get_settings",
            return_value=settings,
        ),
        patch(
            "scripts.evaluate_generation.evaluate_generation",
            new=AsyncMock(side_effect=RuntimeError("evaluation failed")),
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="evaluation failed",
        ):
            await run_generation_evaluation(
                dataset_path=dataset,
                output_path=output,
            )

    fake_llm_client.close.assert_awaited_once_with()
    assert not output.exists()


def test_check_citations_accepts_unicode_dash_variant() -> None:
    result = check_citations(
        answer=("Никто не может быть повторно осужден [Статья 50, части 1–2]."),
        available_refs=["Статья 50, части 1-2"],
    )

    assert result.inline_citations == ("Статья 50, части 1–2",)
    assert result.valid_citations == ("Статья 50, части 1–2",)
    assert result.invalid_citations == ()
    assert result.all_citations_valid is True


def test_check_citations_splits_multiple_refs_inside_one_bracket() -> None:
    result = check_citations(
        answer=(
            "Гарантируется судебная защита [Статья 46, части 1–2; Статья 47, часть 2]."
        ),
        available_refs=[
            "Статья 46, части 1-2",
            "Статья 47, часть 2",
        ],
    )

    assert result.inline_citations == (
        "Статья 46, части 1–2",
        "Статья 47, часть 2",
    )
    assert result.valid_citations == (
        "Статья 46, части 1–2",
        "Статья 47, часть 2",
    )
    assert result.invalid_citations == ()
    assert result.all_citations_valid is True
