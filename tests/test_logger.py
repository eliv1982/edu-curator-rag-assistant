"""Query logging: row format and spreadsheet-formula neutralization."""

import csv
import re

import pytest

from src import logger


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "query_log.csv"
    monkeypatch.setattr(logger, "LOG_DIR", path.parent)
    monkeypatch.setattr(logger, "LOG_FILE", path)
    return path


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_normal_row_is_written_with_header(log_file):
    logger.log_query(
        "Когда дедлайн по модулю 2?",
        "beginner",
        ["assignments.csv (deadlines)", "faq.md (general)"],
        "Дедлайн 2026-06-21.",
    )

    with open(log_file, newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == logger.LOG_COLUMNS
    (row,) = read_rows(log_file)
    assert row["question"] == "Когда дедлайн по модулю 2?"
    assert row["learner_level"] == "beginner"
    assert row["retrieved_sources"] == "assignments.csv (deadlines); faq.md (general)"
    assert row["answer_preview"] == "Дедлайн 2026-06-21."
    assert row["topic_guess"] == "deadlines"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", row["timestamp"])


def test_rows_are_appended_without_repeating_the_header(log_file):
    logger.log_query("один", "beginner", [], "a")
    logger.log_query("два", "advanced", [], "b")

    assert [r["question"] for r in read_rows(log_file)] == ["один", "два"]


def test_no_sources_are_recorded_as_none(log_file):
    logger.log_query("вопрос", "beginner", [], "ответ")

    assert read_rows(log_file)[0]["retrieved_sources"] == "none"


def test_answer_preview_is_single_line_and_truncated(log_file):
    logger.log_query("вопрос", "beginner", [], "строка\n" + "x" * 500)

    preview = read_rows(log_file)[0]["answer_preview"]
    assert "\n" not in preview
    assert len(preview) == 200


def test_commas_quotes_and_newlines_in_question_survive_round_trip(log_file):
    question = 'Привет, "мир"\nвторая строка'

    logger.log_query(question, "beginner", [], "ответ")

    assert read_rows(log_file)[0]["question"] == question


@pytest.mark.parametrize("prefix", ["=", "+", "-", "@"])
def test_formula_prefixes_are_neutralized_in_question_and_answer(log_file, prefix):
    payload = f'{prefix}HYPERLINK("http://evil.example","x")'

    logger.log_query(payload, "beginner", [], payload)

    (row,) = read_rows(log_file)
    assert row["question"] == "'" + payload
    assert row["answer_preview"] == "'" + payload
    assert not row["question"].startswith(("=", "+", "-", "@"))


def test_safe_values_are_not_modified(log_file):
    logger.log_query("Что такое 2+2=4?", "beginner", ["faq.md"], "ответ - без изменений")

    (row,) = read_rows(log_file)
    assert row["question"] == "Что такое 2+2=4?"
    assert row["answer_preview"] == "ответ - без изменений"
