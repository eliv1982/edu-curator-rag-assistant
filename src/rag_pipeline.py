"""
RAG pipeline: index building and document retrieval via FAISS.
"""

import csv
import re
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data")
INDEX_DIR = Path("storage/faiss_index")
ASSIGNMENTS_CSV = DATA_DIR / "assignments.csv"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Файлы, которые не индексируются в FAISS.
# few_shot_examples.md содержит примеры поведения ассистента, а не знания курса —
# включать его в retrieval нецелесообразно.
SKIP_FILES = {"few_shot_examples.md"}


def _get_value(row: dict, *keys: str) -> str:
    """Return the first non-empty value found for any of the given keys."""
    # Strip a stray BOM from header names; missing cells (short rows) are None.
    clean = {(k or "").lstrip("\ufeff").strip(): v for k, v in row.items()}
    for key in keys:
        val = clean.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _read_assignments() -> list[dict]:
    """
    Read assignments.csv into rows with the canonical English keys
    (module, assignment, deadline, status, description).

    Uses utf-8-sig to safely strip the BOM that Windows/Excel adds, and accepts
    both English and Russian column names. Returns [] if the file is missing.
    """
    if not ASSIGNMENTS_CSV.exists():
        return []
    rows = []
    with open(ASSIGNMENTS_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append({
                "module": _get_value(row, "module", "Модуль"),
                "assignment": _get_value(row, "assignment", "Задание"),
                "deadline": _get_value(row, "deadline", "Дедлайн"),
                "status": _get_value(row, "status", "Статус"),
                "description": _get_value(row, "description", "Описание"),
            })
    return rows


def _load_assignments_as_documents() -> list[Document]:
    """Convert each row of assignments.csv into a LangChain Document."""
    docs = []
    for row in _read_assignments():
        module = row["module"]
        assignment = row["assignment"]
        deadline = row["deadline"]
        text = (
            "Источник: LMS-заглушка assignments.csv\n"
            f"Модуль: {module}\n"
            f"Задание: {assignment}\n"
            f"Дедлайн: {deadline}\n"
            f"Статус: {row['status']}\n"
            f"Описание: {row['description']}"
        )
        metadata = {
            "source": "assignments.csv",
            "resource_type": "lms_stub",
            "topic": "deadlines",
            "module": module,
            "assignment": assignment,
        }
        docs.append(Document(page_content=text, metadata=metadata))
        print(f"  [assignments] {module} | deadline={deadline} | {assignment}")

    return docs


def _parse_metadata_from_text(text: str, filename: str) -> dict:
    """Extract metadata fields from the top of a Markdown document."""
    metadata = {
        "source": filename,
        "resource_type": "general",
        "module": "general",
        "topic": "general",
    }
    pattern = r"\*\*(\w+):\*\*\s*(.+)"
    for line in text.splitlines()[:15]:
        match = re.search(pattern, line)
        if match:
            key = match.group(1).strip().lower()
            value = match.group(2).strip()
            if key in metadata:
                metadata[key] = value
    return metadata


def build_index() -> FAISS:
    """Load all .md files from data/, chunk them, embed, and save FAISS index."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    md_files = [f for f in DATA_DIR.glob("*.md") if f.name not in SKIP_FILES]
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {DATA_DIR}")

    documents = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    for md_path in md_files:
        raw_text = md_path.read_text(encoding="utf-8")
        meta = _parse_metadata_from_text(raw_text, md_path.name)

        chunks = splitter.create_documents(
            texts=[raw_text],
            metadatas=[meta],
        )
        documents.extend(chunks)

    # Добавляем строки assignments.csv как отдельные документы.
    # Каждая строка — один Document: модуль, задание, дедлайн, статус, описание.
    # Это позволяет ассистенту отвечать на вопросы о дедлайнах по точным данным.
    print("[build_index] Loading assignments.csv ...")
    assignment_docs = _load_assignments_as_documents()
    documents.extend(assignment_docs)
    print(f"[build_index] Total documents: {len(documents)} "
          f"(md chunks + {len(assignment_docs)} assignment rows)")

    vectorstore = FAISS.from_documents(documents, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    return vectorstore


def load_index() -> FAISS | None:
    """Load FAISS index from disk. Returns None if index does not exist."""
    index_file = INDEX_DIR / "index.faiss"
    if not index_file.exists():
        return None
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_or_build_index() -> FAISS:
    """Return existing FAISS index or build it if missing."""
    vectorstore = load_index()
    if vectorstore is None:
        vectorstore = build_index()
    return vectorstore


def retrieve(query: str, vectorstore: FAISS, top_k: int = 4) -> list[dict]:
    """
    Retrieve top_k relevant chunks for a query.

    Returns a list of dicts with 'text' and 'metadata' keys.
    """
    results = vectorstore.similarity_search(query, k=top_k)
    return [
        {"text": doc.page_content, "metadata": doc.metadata}
        for doc in results
    ]


# ---------------------------------------------------------------------------
# Детерминированное извлечение данных о дедлайнах из assignments.csv
# ---------------------------------------------------------------------------

_DEADLINE_TRIGGERS = [
    "дедлайн", "срок", "сдать", "задани",  # "задани" — основа для задание/задания/заданий
    "когда сдавать", "когда нужно", "до когда",
]
# Само слово «модуль» намеренно не является триггером: «Что изучается в модуле 3?»
# — вопрос о содержании, а не о дедлайне. Номер модуля используется только после
# того, как вопрос распознан как вопрос о дедлайнах/заданиях.


def detect_assignment_query(question: str) -> bool:
    """Return True if the question is likely about deadlines or assignments."""
    q = question.lower()
    return any(kw in q for kw in _DEADLINE_TRIGGERS)


def extract_module_number(question: str) -> int | None:
    """
    Extract an explicit module number from the question.

    Handles any number of digits and the common Russian inflections:
    'модуль 4' → 4, 'модуля 10' → 10. Returns None if no number follows
    the word «модуль».
    """
    match = re.search(r"модул\w*\s+(\d+)", question.lower())
    return int(match.group(1)) if match else None


def _module_number(module_value: str) -> int | None:
    """Number from a structured module value: 'Модуль 10' → 10."""
    match = re.search(r"\d+", module_value)
    return int(match.group()) if match else None


def get_assignment_context(question: str) -> list[dict]:
    """
    Deterministically build context chunks from assignments.csv.

    - If the question names a module that exists in the data, return exactly
      that module's row (matched on the module number, so 1 never matches 10).
    - Otherwise return a compact summary of all assignments. This includes a
      module number that is not in the data: no deadline is invented for it,
      and the summary shows which modules actually exist.

    Always returns a list in the same format as retrieve(); it is empty only
    when there is no assignment data at all.
    """
    rows = _read_assignments()
    if not rows:
        return []

    module_num = extract_module_number(question)

    if module_num is not None:
        for row in rows:
            if _module_number(row["module"]) == module_num:
                text = (
                    "Источник: assignments.csv (точные данные о дедлайне)\n"
                    f"Модуль: {row['module']}\n"
                    f"Задание: {row['assignment']}\n"
                    f"Дедлайн: {row['deadline']}\n"
                    f"Статус: {row['status']}\n"
                    f"Описание: {row['description']}"
                )
                return [{"text": text, "metadata": {
                    "source": "assignments.csv",
                    "resource_type": "lms_stub",
                    "topic": "deadlines",
                    "module": row["module"],
                }}]

    lines = ["Источник: assignments.csv — список всех заданий и дедлайнов\n"]
    for row in rows:
        lines.append(
            f"• {row['module']} | {row['assignment']} "
            f"| Дедлайн: {row['deadline']} | {row['status']}"
        )
    text = "\n".join(lines)
    return [{"text": text, "metadata": {
        "source": "assignments.csv",
        "resource_type": "lms_stub",
        "topic": "deadlines",
        "module": "all",
    }}]
