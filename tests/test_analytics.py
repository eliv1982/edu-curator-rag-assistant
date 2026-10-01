"""Analytics: loading the query log and the aggregations the dashboard uses."""

import pytest

from src import analytics, logger


@pytest.fixture
def log_file(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "query_log.csv"
    monkeypatch.setattr(logger, "LOG_DIR", path.parent)
    monkeypatch.setattr(logger, "LOG_FILE", path)
    monkeypatch.setattr(analytics, "LOG_FILE", path)
    return path


def test_missing_log_loads_as_none(log_file):
    assert analytics.load_logs() is None


@pytest.mark.parametrize("content", ["", "timestamp,question\n"], ids=["empty", "header-only"])
def test_empty_log_loads_as_none(log_file, content):
    log_file.parent.mkdir(parents=True)
    log_file.write_text(content, encoding="utf-8")

    assert analytics.load_logs() is None


def test_logger_output_is_readable_and_aggregates_correctly(log_file):
    logger.log_query("Когда дедлайн?", "beginner", [], "a")
    logger.log_query("Что такое RAG?", "advanced", [], "b")
    logger.log_query("Какой срок сдачи?", "beginner", [], "c")

    df = analytics.load_logs()

    assert analytics.total_questions(df) == 3
    by_level = analytics.questions_by_level(df)
    assert by_level["beginner"] == 2
    assert by_level["advanced"] == 1
    by_topic = analytics.questions_by_topic(df)
    assert by_topic["deadlines"] == 2
    assert by_topic["rag"] == 1


def test_recent_questions_are_newest_first_and_limited(log_file):
    for n in range(5):
        logger.log_query(f"вопрос {n}", "beginner", [], "ответ")

    recent = analytics.recent_questions(analytics.load_logs(), n=3)

    assert list(recent["question"]) == ["вопрос 4", "вопрос 3", "вопрос 2"]
    assert list(recent.columns) == ["timestamp", "question", "learner_level", "topic_guess"]
