"""
Query logger: appends each student interaction to logs/query_log.csv.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "query_log.csv"

LOG_COLUMNS = [
    "timestamp",
    "question",
    "learner_level",
    "retrieved_sources",
    "answer_preview",
    "topic_guess",
]

TOPIC_KEYWORDS = {
    "deadlines": [
        "deadline", "due", "late", "extension", "submit", "submission", "when is",
        "дедлайн", "срок", "опоздан", "перенос", "когда сдавать",
        "когда нужно", "когда дедлайн",
    ],
    "assignments": [
        "assignment", "task", "homework", "project", "grade", "score", "pass", "fail", "certificate",
        "задание", "задача", "домашнее", "проект", "оценка", "балл", "зачёт", "зачет",
        "сертификат", "пересдача", "провалил",
    ],
    "rag": [
        "rag", "retrieval", "vector", "faiss", "embedding", "index", "langchain", "chunk",
        "раг", "ретривал", "векторн", "эмбеддинг", "индекс", "чанк", "поиск по документам",
    ],
    "prompts": [
        "prompt", "chain-of-thought", "cot", "few-shot", "zero-shot", "system prompt", "role prompt",
        "промпт", "цепочка рассуждений", "few-shot", "zero-shot", "системный промпт",
        "ролевой промпт", "промптинг",
    ],
    "resources": [
        "resource", "book", "paper", "read", "watch", "learn more", "recommend",
        "ресурс", "книга", "статья", "почитать", "посмотреть", "рекоменд", "дополнительно",
        "материалы посмотреть", "материал",
    ],
}


def guess_topic(question: str) -> str:
    """Classify the question into a topic using keyword matching."""
    question_lower = question.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in question_lower for kw in keywords):
            return topic
    return "general"


def _ensure_log_file() -> None:
    """Create the log file with headers if it does not exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()


def log_query(
    question: str,
    learner_level: str,
    retrieved_sources: list[str],
    answer: str,
) -> None:
    """Append a query record to the log file."""
    _ensure_log_file()

    sources_str = "; ".join(retrieved_sources) if retrieved_sources else "none"
    answer_preview = answer[:200].replace("\n", " ") if answer else ""
    topic = guess_topic(question)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "timestamp": timestamp,
        "question": question,
        "learner_level": learner_level,
        "retrieved_sources": sources_str,
        "answer_preview": answer_preview,
        "topic_guess": topic,
    }

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow(row)
