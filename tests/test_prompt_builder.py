"""
Prompt/context helpers. These tests pin the contracts (what goes where), not the
exact wording of the system prompt, so documentation-style prompt edits do not
break them.
"""

from types import SimpleNamespace

from src import prompt_builder
from src.prompt_builder import ask_llm, build_system_prompt, format_context


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------
def test_format_context_with_no_chunks_is_empty():
    assert format_context([]) == ("", [])


def test_format_context_keeps_order_and_labels_each_chunk():
    chunks = [
        {"text": "  Первый  ", "metadata": {"source": "faq.md", "topic": "deadlines"}},
        {"text": "Второй", "metadata": {"source": "modules.md"}},
    ]

    context, sources = format_context(chunks)

    first, second = context.split("---")
    assert "[faq.md]" in first and "Первый" in first
    assert "[modules.md]" in second and "Второй" in second
    assert context.index("Первый") < context.index("Второй")
    assert sources == ["faq.md (deadlines)", "modules.md"]


def test_format_context_lists_each_source_once():
    chunks = [
        {"text": "a", "metadata": {"source": "faq.md", "topic": "rag"}},
        {"text": "b", "metadata": {"source": "faq.md", "topic": "rag"}},
        {"text": "c", "metadata": {"source": "faq.md", "topic": "prompts"}},
    ]

    _, sources = format_context(chunks)

    assert sources == ["faq.md (rag)", "faq.md (prompts)"]


def test_format_context_tolerates_missing_metadata():
    context, sources = format_context([{"text": "голый текст"}])

    assert "голый текст" in context
    assert sources == ["unknown"]


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------
def test_context_is_inserted_verbatim_even_with_braces():
    context = "[faq.md]\nПример кода: def f(): return {'a': 1}  {не_плейсхолдер}"

    prompt = build_system_prompt(context, "beginner")

    assert context in prompt


def test_learner_level_changes_exactly_the_level_line():
    beginner = build_system_prompt("ctx", "beginner").splitlines()
    advanced = build_system_prompt("ctx", "advanced").splitlines()

    assert len(beginner) == len(advanced)
    differing = [(b, a) for b, a in zip(beginner, advanced, strict=True) if b != a]
    assert len(differing) == 1
    assert "beginner" in differing[0][0]
    assert "advanced" in differing[0][1]


def test_empty_context_gets_an_explicit_no_materials_notice():
    prompt = build_system_prompt("", "beginner")

    assert "не найдены" in prompt
    assert "{context}" not in prompt
    assert "{learner_level}" not in prompt


# ---------------------------------------------------------------------------
# ask_llm (OpenAI client replaced by a fake)
# ---------------------------------------------------------------------------
INSTANCES = []


class FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        INSTANCES.append(self)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content="ответ ассистента")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_ask_llm_sends_system_prompt_and_question_and_returns_answer(monkeypatch):
    INSTANCES.clear()
    monkeypatch.setattr(prompt_builder, "OpenAI", FakeOpenAI)

    answer = ask_llm("Что такое RAG?", "КОНТЕКСТ-123", "advanced", model="test-model")

    assert answer == "ответ ассистента"
    (request,) = INSTANCES[0].requests
    assert request["model"] == "test-model"
    system, user = request["messages"]
    assert system["role"] == "system"
    assert "КОНТЕКСТ-123" in system["content"]
    assert user == {"role": "user", "content": "Что такое RAG?"}
