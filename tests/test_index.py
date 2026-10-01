"""
Index build/load/retrieve wiring, using a deterministic fake embedding so no
OpenAI call is made. LangChain/FAISS internals are not tested.
"""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from src import rag_pipeline


@pytest.fixture
def kb(tmp_path, monkeypatch):
    """A tiny knowledge base in tmp_path with fake embeddings."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "faq.md").write_text(
        "# FAQ\n\n**topic:** deadlines\n\nКак сдать задание? Загрузите файл на платформу.\n",
        encoding="utf-8",
    )
    (data / "modules.md").write_text(
        "# Модули\n\nМодуль 1 посвящён основам работы языковых моделей.\n",
        encoding="utf-8",
    )
    (data / "few_shot_examples.md").write_text("Справочные примеры, не знания курса.\n", encoding="utf-8")
    (data / "assignments.csv").write_text(
        "module,assignment,deadline,status,description\n"
        "Модуль 1,Рефлексия,2026-06-14,Открыто,Напишите рефлексию\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag_pipeline, "DATA_DIR", data)
    monkeypatch.setattr(rag_pipeline, "ASSIGNMENTS_CSV", data / "assignments.csv")
    monkeypatch.setattr(rag_pipeline, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(
        rag_pipeline, "OpenAIEmbeddings", lambda **kwargs: DeterministicFakeEmbedding(size=32)
    )
    return data


def everything(vectorstore):
    return rag_pipeline.retrieve("любой запрос", vectorstore, top_k=100)


def test_build_index_covers_markdown_and_assignments_but_skips_few_shot(kb):
    vectorstore = rag_pipeline.build_index()

    chunks = everything(vectorstore)
    sources = {c["metadata"]["source"] for c in chunks}
    assert sources == {"faq.md", "modules.md", "assignments.csv"}
    assert (rag_pipeline.INDEX_DIR / "index.faiss").exists()


def test_chunk_metadata_comes_from_header_and_csv_rows(kb):
    chunks = everything(rag_pipeline.build_index())

    faq = next(c for c in chunks if c["metadata"]["source"] == "faq.md")
    assert faq["metadata"]["topic"] == "deadlines"
    assignment = next(c for c in chunks if c["metadata"]["source"] == "assignments.csv")
    assert assignment["metadata"]["module"] == "Модуль 1"
    assert "Дедлайн: 2026-06-14" in assignment["text"]


def test_load_index_is_none_until_built_then_round_trips(kb):
    assert rag_pipeline.load_index() is None

    built = rag_pipeline.build_index()
    loaded = rag_pipeline.load_index()

    assert loaded is not None
    assert len(everything(loaded)) == len(everything(built))


def test_get_or_build_index_builds_once_then_loads(kb, monkeypatch):
    first = rag_pipeline.get_or_build_index()
    monkeypatch.setattr(
        rag_pipeline, "build_index", lambda: pytest.fail("index should be loaded, not rebuilt")
    )

    second = rag_pipeline.get_or_build_index()

    assert len(everything(second)) == len(everything(first))


def test_retrieve_respects_top_k_and_returns_text_with_metadata(kb):
    vectorstore = rag_pipeline.build_index()

    results = rag_pipeline.retrieve("Как сдать задание?", vectorstore, top_k=2)

    assert len(results) == 2
    assert all(set(r) == {"text", "metadata"} for r in results)


def test_build_index_without_markdown_files_fails_clearly(kb):
    for md in kb.glob("*.md"):
        md.unlink()

    with pytest.raises(FileNotFoundError):
        rag_pipeline.build_index()
