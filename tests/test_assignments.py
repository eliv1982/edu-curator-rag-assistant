"""
Deterministic assignment/deadline routing: query detection, module-number
extraction, assignments.csv loading and the context built from it.
"""

import csv

import pytest

from src import rag_pipeline

ROWS = [
    ("Модуль 1", "Рефлексия об основах AI", "2026-06-14", "Открыто", "Описание один"),
    ("Модуль 2", "Портфолио промптов", "2026-06-21", "Открыто", "Описание два"),
    ("Модуль 10", "Десятое задание", "2026-09-01", "Открыто", "Описание десять"),
]


# ---------------------------------------------------------------------------
# detect_assignment_query
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question",
    [
        "Какой дедлайн у модуля 3?",
        "Когда нужно сдать задание модуля 2?",
        "Какие задания есть в курсе?",
        "Какой срок сдачи?",
        "До когда можно отправить работу?",
    ],
)
def test_assignment_questions_are_detected(question):
    assert rag_pipeline.detect_assignment_query(question)


@pytest.mark.parametrize(
    "question",
    [
        "Что изучается в модуле 3?",
        "Поставь мне зачёт за модуль 2",
        "Расскажи про модуль 5 подробнее",
        "Что такое RAG?",
    ],
)
def test_word_module_alone_does_not_trigger_deadline_routing(question):
    assert not rag_pipeline.detect_assignment_query(question)


# ---------------------------------------------------------------------------
# extract_module_number
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Какой дедлайн у модуля 3?", 3),
        ("Дедлайн по модулю 4", 4),
        ("Что в модуле 6?", 6),
        ("модуль 1", 1),
        ("МОДУЛЬ 2", 2),
        ("Какой дедлайн у модуля 10?", 10),
        ("Когда сдать задание модуль 12", 12),
    ],
)
def test_explicit_module_number_is_extracted(question, expected):
    assert rag_pipeline.extract_module_number(question) == expected


@pytest.mark.parametrize(
    "question",
    ["Какие у меня дедлайны?", "Расскажи про модуль", "Модуль без номера, 3 задания"],
)
def test_no_module_number_means_none(question):
    assert rag_pipeline.extract_module_number(question) is None


# ---------------------------------------------------------------------------
# get_assignment_context
# ---------------------------------------------------------------------------
def test_specific_module_returns_only_that_row(assignments_csv):
    assignments_csv(ROWS)

    (chunk,) = rag_pipeline.get_assignment_context("Какой дедлайн у модуля 2?")

    assert chunk["metadata"]["module"] == "Модуль 2"
    assert "Дедлайн: 2026-06-21" in chunk["text"]
    assert "Рефлексия" not in chunk["text"]


def test_module_1_does_not_match_module_10(assignments_csv):
    # Module 10 is listed before module 1: a substring match would pick it.
    assignments_csv([ROWS[2], ROWS[0]])

    (chunk,) = rag_pipeline.get_assignment_context("Дедлайн модуля 1")

    assert chunk["metadata"]["module"] == "Модуль 1"
    assert "2026-06-14" in chunk["text"]
    assert "2026-09-01" not in chunk["text"]


def test_multi_digit_module_matches_its_own_row(assignments_csv):
    assignments_csv(ROWS)

    (chunk,) = rag_pipeline.get_assignment_context("Дедлайн модуля 10")

    assert chunk["metadata"]["module"] == "Модуль 10"
    assert "Дедлайн: 2026-09-01" in chunk["text"]


def test_no_module_in_question_returns_summary_of_all(assignments_csv):
    assignments_csv(ROWS)

    (chunk,) = rag_pipeline.get_assignment_context("Какие дедлайны в курсе?")

    assert chunk["metadata"]["module"] == "all"
    for row in ROWS:
        assert row[0] in chunk["text"]
        assert row[2] in chunk["text"]


def test_unknown_module_returns_a_list_and_invents_no_deadline(assignments_csv):
    assignments_csv(ROWS)

    result = rag_pipeline.get_assignment_context("Какой дедлайн у модуля 3?")

    # Documented contract: always a list of chunks, never None.
    assert isinstance(result, list)
    # No row-specific chunk for the missing module: falls back to the summary
    # of the modules that do exist.
    assert [c["metadata"]["module"] for c in result] == ["all"]
    assert "Модуль 3" not in result[0]["text"]


def test_unknown_module_can_be_prepended_to_retrieved_chunks(assignments_csv):
    # Regression: the caller does `priority + chunks`; None used to raise TypeError.
    assignments_csv(ROWS)
    retrieved = [{"text": "chunk", "metadata": {"source": "faq.md"}}]

    combined = rag_pipeline.get_assignment_context("Дедлайн модуля 3") + retrieved

    assert combined[-1] == retrieved[0]


def test_only_module_10_in_data_does_not_answer_for_module_1(assignments_csv):
    assignments_csv([ROWS[2]])

    result = rag_pipeline.get_assignment_context("Дедлайн модуля 1")

    assert [c["metadata"]["module"] for c in result] == ["all"]


def test_missing_csv_gives_empty_list(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_pipeline, "ASSIGNMENTS_CSV", tmp_path / "absent.csv")

    assert rag_pipeline.get_assignment_context("Дедлайн модуля 1") == []
    assert rag_pipeline.get_assignment_context("Какие дедлайны?") == []


# ---------------------------------------------------------------------------
# CSV loading: same column handling for index building and direct lookup
# ---------------------------------------------------------------------------
RUSSIAN_HEADER = "Модуль,Задание,Дедлайн,Статус,Описание"


@pytest.mark.parametrize(
    ("header", "encoding"),
    [
        ("module,assignment,deadline,status,description", "utf-8"),
        (RUSSIAN_HEADER, "utf-8"),
        (RUSSIAN_HEADER, "utf-8-sig"),  # Excel-style BOM
    ],
    ids=["english", "russian", "russian-with-bom"],
)
def test_index_documents_and_direct_lookup_agree(assignments_csv, header, encoding):
    assignments_csv(ROWS, header=header, encoding=encoding)

    docs = rag_pipeline._load_assignments_as_documents()
    (chunk,) = rag_pipeline.get_assignment_context("Дедлайн модуля 2")

    assert len(docs) == len(ROWS)
    doc = next(d for d in docs if d.metadata["module"] == "Модуль 2")
    assert "Дедлайн: 2026-06-21" in doc.page_content
    assert doc.metadata["assignment"] == "Портфолио промптов"
    # The direct lookup reads the same values from the same columns.
    assert chunk["metadata"]["module"] == "Модуль 2"
    assert "Дедлайн: 2026-06-21" in chunk["text"]
    assert "Задание: Портфолио промптов" in chunk["text"]


def test_short_row_does_not_crash_loading(tmp_path, monkeypatch):
    path = tmp_path / "assignments.csv"
    path.write_text(
        "module,assignment,deadline,status,description\nМодуль 1,Только название\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag_pipeline, "ASSIGNMENTS_CSV", path)

    docs = rag_pipeline._load_assignments_as_documents()
    (chunk,) = rag_pipeline.get_assignment_context("Дедлайн модуля 1")

    assert len(docs) == 1
    assert chunk["metadata"]["module"] == "Модуль 1"


def test_shipped_assignments_csv_resolves_every_module(repo_root, monkeypatch):
    """The real demo data has the expected schema and each module finds its own row."""
    csv_path = repo_root / "data" / "assignments.csv"
    monkeypatch.setattr(rag_pipeline, "ASSIGNMENTS_CSV", csv_path)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["module", "assignment", "deadline", "status", "description"]
        expected = list(reader)

    assert expected
    for row in expected:
        number = int(row["module"].split()[-1])
        (chunk,) = rag_pipeline.get_assignment_context(f"Какой дедлайн у модуля {number}?")
        assert chunk["metadata"]["module"] == row["module"]
        assert f"Дедлайн: {row['deadline']}" in chunk["text"]
