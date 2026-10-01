"""
Streamlit smoke tests via AppTest. Nothing here calls OpenAI: the retrieval and
LLM entry points are replaced by recorders/fakes.
"""

import logging

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import analytics, logger, prompt_builder, rag_pipeline

FAKE_KEY = "sk-test-not-a-real-key"
LEAKY_ERROR = "secret-token-123 at C:\\private\\path"


class Calls:
    """Records calls to anything that would reach OpenAI or the vector index."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError("unexpected call to an OpenAI/index entry point")


@pytest.fixture
def make_app(monkeypatch, tmp_path, repo_root):
    """Factory for an AppTest on app.py, isolated from the developer's machine."""
    monkeypatch.chdir(repo_root)  # app.py reads data/ relative to the working directory
    # The developer's real .env must not leak a real key into "no key" tests.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    # Never touch the real logs/ directory.
    log_file = tmp_path / "logs" / "query_log.csv"
    for module in (logger, analytics):
        monkeypatch.setattr(module, "LOG_FILE", log_file)
    monkeypatch.setattr(logger, "LOG_DIR", log_file.parent)
    st.cache_resource.clear()

    def make(api_key=None):
        if api_key:
            monkeypatch.setenv("OPENAI_API_KEY", api_key)
        return AppTest.from_file(str(repo_root / "app.py"), default_timeout=60)

    yield make
    st.cache_resource.clear()


@pytest.fixture
def fake_answering(monkeypatch):
    """Replace index/retrieval/LLM with fakes; return the dict of captured LLM inputs."""
    captured = {}
    retrieved = [{"text": "Текст из faq", "metadata": {"source": "faq.md", "topic": "rag"}}]

    monkeypatch.setattr(rag_pipeline, "get_or_build_index", lambda: object())
    monkeypatch.setattr(rag_pipeline, "retrieve", lambda question, vs, top_k=4: list(retrieved))

    def fake_ask_llm(question, context, learner_level, model="gpt-4o-mini"):
        captured.update(question=question, context=context, level=learner_level)
        return "Готовый ответ ассистента"

    monkeypatch.setattr(prompt_builder, "ask_llm", fake_ask_llm)
    return captured


def submit(at, question, level=None):
    at.text_area[0].input(question)
    if level:
        at.radio[0].set_value(level)
    at.button[0].click()
    at.run()


def all_text(at):
    values = [el.value for el in at.markdown] + [el.value for el in at.subheader]
    values += [el.value for el in at.error] + [el.value for el in at.warning]
    values += [el.value for el in at.info] + [el.value for el in at.caption]
    return "\n".join(str(v) for v in values)


# ---------------------------------------------------------------------------
# No OpenAI key: Q&A explains itself, the other tabs keep working
# ---------------------------------------------------------------------------
def test_app_starts_without_key_and_explains_missing_key(make_app):
    at = make_app().run()

    assert not at.exception
    assert len(at.tabs) == 3
    assert any("OPENAI_API_KEY" in err.value for err in at.error)
    assert len(at.text_area) == 0  # no question form without a key


def test_course_tab_renders_without_key(make_app):
    at = make_app().run()

    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert any("Задания и дедлайны" in s for s in subheaders)
    assert any("Модули курса" in s for s in subheaders)
    assert len(at.dataframe) >= 1  # assignments table
    assert len(at.expander) >= 6  # per-assignment and per-module details


def test_analytics_tab_without_key_and_without_logs(make_app):
    at = make_app().run()

    assert not at.exception
    assert any("Аналитика" in t.value for t in at.title)
    assert any("Данных пока нет" in info.value for info in at.info)


def test_analytics_tab_without_key_shows_logged_questions(make_app):
    logger.log_query("Когда дедлайн?", "beginner", [], "ответ")
    logger.log_query("Что такое RAG?", "advanced", [], "ответ")

    at = make_app().run()

    assert not at.exception
    assert [m.value for m in at.metric] == ["2", "1", "1"]


# ---------------------------------------------------------------------------
# Opening the app is free: no embedding, index or LLM call
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("api_key", [None, FAKE_KEY], ids=["no-key", "with-key"])
def test_opening_the_app_makes_no_api_calls(make_app, monkeypatch, api_key):
    recorders = {}
    for module, name in [
        (rag_pipeline, "OpenAIEmbeddings"),
        (rag_pipeline, "get_or_build_index"),
        (rag_pipeline, "retrieve"),
        (prompt_builder, "OpenAI"),
        (prompt_builder, "ask_llm"),
    ]:
        recorders[name] = Calls()
        monkeypatch.setattr(module, name, recorders[name])

    at = make_app(api_key).run()

    assert not at.exception
    assert {name: r.calls for name, r in recorders.items() if r.calls} == {}


def dataframe_text(at):
    return "\n".join(df.value.to_string() for df in at.dataframe)


SECRET_QUESTION = "Меня зовут Иван Петров, мой телефон 8-900-123-45-67"


def test_analytics_hides_raw_questions_by_default_but_keeps_aggregates(make_app):
    logger.log_query(SECRET_QUESTION, "beginner", [], "ответ")
    logger.log_query("Что такое RAG?", "advanced", [], "ответ")

    at = make_app().run()

    assert not at.exception
    # Aggregates still render ...
    assert [m.value for m in at.metric] == ["2", "1", "1"]
    subheaders = [s.value for s in at.subheader]
    assert "По уровню студента" in subheaders
    assert "По теме вопроса" in subheaders
    assert len(at.get("vega_lite_chart")) == 2  # st.bar_chart: by level, by topic
    # ... but no raw question text is rendered anywhere, and the table is absent.
    assert "Последние 10 вопросов" not in subheaders
    assert any("скрыты по умолчанию" in c.value for c in at.caption)
    assert "Иван Петров" not in all_text(at)
    assert "Иван Петров" not in dataframe_text(at)
    assert "Что такое RAG?" not in dataframe_text(at)


@pytest.mark.parametrize("value", ["false", "0", "off"])
def test_analytics_stays_hidden_for_non_opt_in_values(make_app, monkeypatch, value):
    monkeypatch.setenv("SHOW_RECENT_QUESTIONS", value)
    logger.log_query(SECRET_QUESTION, "beginner", [], "ответ")

    at = make_app().run()

    assert not at.exception
    assert "Иван Петров" not in dataframe_text(at)


def test_analytics_shows_recent_questions_on_explicit_opt_in(make_app, monkeypatch):
    monkeypatch.setenv("SHOW_RECENT_QUESTIONS", "true")
    logger.log_query(SECRET_QUESTION, "beginner", [], "ответ")
    logger.log_query("Что такое RAG?", "advanced", [], "ответ")

    at = make_app().run()

    assert not at.exception
    assert [m.value for m in at.metric] == ["2", "1", "1"]  # aggregates unchanged
    assert "Последние 10 вопросов" in [s.value for s in at.subheader]
    table = dataframe_text(at)
    assert "Иван Петров" in table
    assert "Что такое RAG?" in table
    assert not any("скрыты по умолчанию" in c.value for c in at.caption)


def test_app_uses_no_deprecated_streamlit_apis(make_app, monkeypatch):
    # Streamlit reports deprecations (e.g. use_container_width) on a logger that
    # does not propagate, so attach a handler to it directly.
    monkeypatch.setenv("SHOW_RECENT_QUESTIONS", "true")  # render the recent-questions table too
    messages = []

    class Collect(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    handler = Collect()
    deprecations = logging.getLogger("streamlit.deprecation_util")
    deprecations.addHandler(handler)
    try:
        logger.log_query("Когда дедлайн?", "beginner", [], "ответ")  # renders the analytics table
        at = make_app(FAKE_KEY).run()  # with a key: question form is rendered too
    finally:
        deprecations.removeHandler(handler)

    assert not at.exception
    assert messages == []


# ---------------------------------------------------------------------------
# Query path (with a key, everything behind it faked)
# ---------------------------------------------------------------------------
def test_answer_and_retrieved_materials_are_shown_with_truthful_label(make_app, fake_answering):
    at = make_app(FAKE_KEY).run()

    submit(at, "Что такое RAG?")

    assert not at.exception
    assert not at.error
    subheaders = [s.value for s in at.subheader]
    assert "Ответ" in subheaders
    assert "Найденные материалы" in subheaders
    assert "Источники" not in subheaders  # not verified citations
    assert any("Готовый ответ ассистента" in m.value for m in at.markdown)
    assert any("faq.md" in m.value for m in at.markdown)
    assert fake_answering["level"] == "beginner"


def test_level_selection_reaches_the_model(make_app, fake_answering):
    at = make_app(FAKE_KEY).run()

    submit(at, "Что такое RAG?", level="Продвинутый")

    assert fake_answering["level"] == "advanced"


def test_empty_question_asks_for_input_and_calls_nothing(make_app, fake_answering):
    at = make_app(FAKE_KEY).run()

    submit(at, "   ")

    assert any("введите вопрос" in w.value for w in at.warning)
    assert fake_answering == {}


def test_log_failure_does_not_hide_a_successful_answer(make_app, fake_answering, monkeypatch):
    def failing_log(*args, **kwargs):
        raise OSError(LEAKY_ERROR)

    monkeypatch.setattr(logger, "log_query", failing_log)
    at = make_app(FAKE_KEY).run()

    submit(at, "Что такое RAG?")

    assert not at.exception
    assert not at.error
    assert any("Готовый ответ ассистента" in m.value for m in at.markdown)
    assert "secret-token-123" not in all_text(at)


def test_llm_failure_shows_generic_message_without_exception_text(
    make_app, fake_answering, monkeypatch, caplog
):
    def failing_llm(*args, **kwargs):
        raise RuntimeError(LEAKY_ERROR)

    monkeypatch.setattr(prompt_builder, "ask_llm", failing_llm)
    at = make_app(FAKE_KEY).run()

    with caplog.at_level(logging.ERROR):
        submit(at, "Что такое RAG?")

    assert not at.exception
    assert len(at.error) == 1
    assert "secret-token-123" not in all_text(at)
    assert "private" not in all_text(at)
    # The detail stays available locally in the log.
    assert any("secret-token-123" in str(r.exc_info[1]) for r in caplog.records if r.exc_info)


# ---------------------------------------------------------------------------
# Deadline routing through the real app flow
# ---------------------------------------------------------------------------
def test_deadline_question_puts_exact_assignment_row_first(make_app, fake_answering):
    at = make_app(FAKE_KEY).run()

    submit(at, "Какой дедлайн у модуля 3?")

    assert not at.error
    context = fake_answering["context"]
    assert context.startswith("[assignments.csv]")
    assert "Модуль: Модуль 3" in context
    assert context.index("assignments.csv") < context.index("Текст из faq")


def test_conceptual_module_question_does_not_pull_assignment_data(make_app, fake_answering):
    at = make_app(FAKE_KEY).run()

    submit(at, "Что изучается в модуле 3?")

    assert not at.error
    assert "assignments.csv" not in fake_answering["context"]


def test_deadline_question_for_module_missing_from_data_is_a_normal_answer(
    make_app, fake_answering, assignments_csv
):
    # Regression: this path used to raise TypeError (None + list) and show an error.
    assignments_csv(
        [
            ("Модуль 1", "Первое", "2026-06-14", "Открыто", "Описание"),
            ("Модуль 2", "Второе", "2026-06-21", "Открыто", "Описание"),
        ]
    )
    at = make_app(FAKE_KEY).run()

    submit(at, "Какой дедлайн у модуля 3?")

    assert not at.exception
    assert not at.error
    assert any("Готовый ответ ассистента" in m.value for m in at.markdown)
    assert "Модуль: Модуль 3" not in fake_answering["context"]
