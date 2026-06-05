"""
EduCurator AI — Streamlit MVP
RAG-ассистент куратора для курса «AI Skills Starter Course».
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import (
    get_or_build_index,
    retrieve,
    detect_assignment_query,
    get_assignment_context,
)
from src.prompt_builder import ask_llm, format_context
from src.logger import log_query
from src import analytics


# ---------------------------------------------------------------------------
# Конфигурация страницы
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EduCurator AI",
    page_icon="🎓",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Визуальный стиль — минимальный CSS поверх темы config.toml
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Форма ввода — белая карточка с мягкой рамкой */
    [data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #DDE1EE;
        border-radius: 12px;
        padding: 1.5rem 1.75rem 1.25rem;
    }

    /* Основная кнопка — спокойный синий */
    button[kind="primaryFormSubmit"],
    button[kind="primary"] {
        background-color: #4F6FDE !important;
        border: none !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        transition: background-color 0.2s ease;
    }
    button[kind="primaryFormSubmit"]:hover,
    button[kind="primary"]:hover {
        background-color: #3A57C4 !important;
    }

    /* Блок ответа ассистента — белая карточка */
    [data-testid="stMarkdownContainer"] > div:has(> p) {
        background: #ffffff;
        border-radius: 10px;
        padding: 0.1rem 0;
    }

    /* Убираем красный у активной вкладки (дополнительная страховка) */
    [data-testid="stTab"][aria-selected="true"] {
        color: #4F6FDE !important;
        border-bottom-color: #4F6FDE !important;
    }

    /* Метрики в аналитике — белый фон */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #DDE1EE;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Проверка API-ключа
# ---------------------------------------------------------------------------
def check_api_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key and key.startswith("sk-"))


# ---------------------------------------------------------------------------
# FAISS-индекс — загружается один раз за сессию
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Строим индекс базы знаний…")
def load_vectorstore():
    vs = get_or_build_index()
    return vs


def rebuild_vectorstore():
    """Сбросить кэш и пересоздать индекс (вызывается при необходимости)."""
    load_vectorstore.clear()
    return load_vectorstore()


# ---------------------------------------------------------------------------
# Навигация (вкладки)
# ---------------------------------------------------------------------------
TABS = ["💬 Задать вопрос", "📚 Курс", "📊 Аналитика"]
tab_ask, tab_course, tab_analytics = st.tabs(TABS)


# ============================================================
# ВКЛАДКА 1 — Задать вопрос
# ============================================================
with tab_ask:
    st.title("🎓 EduCurator AI")
    st.markdown(
        """
        **Ваш AI-ассистент куратора для курса «AI Skills Starter Course».**

        Задайте любой вопрос о материалах курса, дедлайнах, заданиях или ресурсах.
        Я отвечу, опираясь на базу знаний курса, и всегда укажу источники.
        """
    )

    if not check_api_key():
        st.error(
            "⚠️ **Ключ OpenAI API не найден.**\n\n"
            "Создайте файл `.env` в корне проекта:\n"
            "```\nOPENAI_API_KEY=sk-ваш-ключ\n```\n"
            "Затем перезапустите приложение."
        )
        st.stop()

    st.divider()

    # ------------------------------------------------------------------
    # Форма ввода
    # ------------------------------------------------------------------
    with st.form("question_form", clear_on_submit=False):
        question = st.text_area(
            "Ваш вопрос",
            placeholder="Например: Объясни RAG простыми словами. Когда дедлайн по заданию модуля 3?",
            height=100,
        )

        level_display = st.radio(
            "Ваш уровень",
            options=["Начинающий", "Продвинутый"],
            horizontal=True,
            help="Выберите уровень, чтобы ассистент адаптировал сложность ответа.",
        )

        submitted = st.form_submit_button("Отправить вопрос", use_container_width=True, type="primary")

    # ------------------------------------------------------------------
    # Генерация ответа
    # ------------------------------------------------------------------
    if submitted:
        if not question.strip():
            st.warning("Пожалуйста, введите вопрос перед отправкой.")
        else:
            learner_level = "beginner" if level_display == "Начинающий" else "advanced"

            with st.spinner("Ищем в материалах курса и формируем ответ…"):
                try:
                    vectorstore = load_vectorstore()
                    top_k = int(os.getenv("RAG_TOP_K", "4"))

                    # Семантический поиск по FAISS
                    chunks = retrieve(question, vectorstore, top_k=top_k)

                    # Для вопросов о дедлайнах/заданиях — детерминированно
                    # добавляем точные данные из assignments.csv в начало контекста.
                    # Это гарантирует, что LLM видит точную дату и не игнорирует её.
                    if detect_assignment_query(question):
                        priority = get_assignment_context(question)
                        chunks = priority + chunks

                    context, sources = format_context(chunks)
                    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                    answer = ask_llm(question, context, learner_level, model=model)
                    log_query(question, learner_level, sources, answer)

                    st.subheader("Ответ")
                    st.markdown(answer)

                    if sources:
                        st.divider()
                        st.subheader("Источники")
                        for src in sources:
                            st.markdown(f"- `{src}`")
                    else:
                        st.info("По этому запросу конкретные материалы курса не найдены.")

                except Exception as exc:
                    st.error(f"Что-то пошло не так: {exc}")


# ============================================================
# ВКЛАДКА 2 — Курс
# ============================================================

# Краткие описания модулей для отображения на вкладке
MODULE_INFO = [
    {
        "номер": "1",
        "тема": "Как работает AI — основы",
        "уровень": "Начинающий",
        "описание": (
            "Разбираемся, что такое машинное обучение, нейронные сети и большие языковые модели. "
            "Изучаем ключевые понятия: токены, контекстное окно, температура. "
            "**Итог:** понимание того, как LLM генерирует текст и в чём её ограничения."
        ),
    },
    {
        "номер": "2",
        "тема": "Основы prompt engineering",
        "уровень": "Начинающий",
        "описание": (
            "Учимся писать эффективные промпты: zero-shot, few-shot, ролевые инструкции, "
            "управление форматом вывода. Разбираем типичные ошибки и способы их исправления. "
            "**Итог:** портфолио из 5 промптов для разных задач."
        ),
    },
    {
        "номер": "3",
        "тема": "Продвинутые техники промптинга",
        "уровень": "Средний",
        "описание": (
            "Chain-of-thought, self-consistency, ReAct, многошаговые пайплайны. "
            "Практикуем рассуждения шаг за шагом для логических и аналитических задач. "
            "**Итог:** сравнительный анализ CoT vs. прямого промптинга на реальных задачах."
        ),
    },
    {
        "номер": "4",
        "тема": "Работа с AI API",
        "уровень": "Начинающий — Средний",
        "описание": (
            "Подключаемся к OpenAI API на Python: системные сообщения, история диалога, "
            "обработка ошибок, оценка стоимости запросов. "
            "**Итог:** рабочий Python-скрипт с пользовательским системным промптом."
        ),
    },
    {
        "номер": "5",
        "тема": "RAG и AI на основе знаний",
        "уровень": "Средний",
        "описание": (
            "Разбираемся, что такое Retrieval-Augmented Generation, как работают эмбеддинги "
            "и векторный поиск. Строим систему Q&A по документам с LangChain и FAISS. "
            "**Итог:** собственная RAG-система с оценкой качества ответов."
        ),
    },
    {
        "номер": "6",
        "тема": "Этика AI и финальный проект",
        "уровень": "Начинающий — Средний",
        "описание": (
            "Изучаем предвзятость, конфиденциальность и ответственное применение AI. "
            "Проектируем и презентуем собственный AI-инструмент или рабочий процесс. "
            "**Итог:** финальный проект — работающее AI-решение реальной задачи."
        ),
    },
]

with tab_course:
    st.title("📚 Курс")
    st.markdown(
        """
        ### AI Skills Starter Course — стартовый курс по AI-инструментам и prompt engineering

        Практический 6-недельный курс: от основ AI до собственного RAG-приложения.
        Технический бэкграунд не обязателен — начинаем с нуля.

        **Продолжительность:** 6 недель · **Нагрузка:** 5–7 ч/нед · **Формат:** Самостоятельный темп + еженедельные Q&A

        **Куратор:** curator@aiskillscourse.example.com · Q&A каждую пятницу в 19:00 МСК
        """
    )

    # ── Задания и дедлайны ───────────────────────────────────────
    st.divider()
    st.subheader("📋 Задания и дедлайны")

    assignments_path = Path("data/assignments.csv")
    if assignments_path.exists():
        try:
            df_assignments = pd.read_csv(assignments_path)

            # Компактная таблица без колонки описания
            st.dataframe(
                df_assignments.drop(columns=["description"], errors="ignore"),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "module": st.column_config.TextColumn("Модуль", width="small"),
                    "assignment": st.column_config.TextColumn("Задание", width="medium"),
                    "deadline": st.column_config.TextColumn("Дедлайн", width="small"),
                    "status": st.column_config.TextColumn("Статус", width="small"),
                },
            )

            # Полные описания в expanders
            st.markdown("#### Подробности по заданиям")
            for _, row in df_assignments.iterrows():
                label = f"{row['module']} — {row['assignment']}  |  📅 {row['deadline']}"
                with st.expander(label):
                    st.markdown(row.get("description", "Описание отсутствует."))

        except Exception as exc:
            st.error(f"Не удалось загрузить assignments.csv: {exc}")
    else:
        st.warning("Файл assignments.csv не найден в data/.")

    # ── Модули курса ─────────────────────────────────────────────
    st.divider()
    st.subheader("🗂 Модули курса")

    for mod in MODULE_INFO:
        label = f"Модуль {mod['номер']} — {mod['тема']}  ·  {mod['уровень']}"
        with st.expander(label):
            st.markdown(mod["описание"])


# ============================================================
# ВКЛАДКА 3 — Аналитика
# ============================================================
with tab_analytics:
    st.title("📊 Аналитика")
    st.markdown("Статистика использования на основе журнала запросов.")

    df_log = analytics.load_logs()

    if df_log is None:
        st.info(
            "Данных пока нет. Файл журнала создастся автоматически, "
            "когда будет отправлен первый вопрос во вкладке **«Задать вопрос»**."
        )
    else:
        col1, col2, col3 = st.columns(3)
        total = analytics.total_questions(df_log)
        col1.metric("Всего вопросов", total)

        by_level = analytics.questions_by_level(df_log)
        col2.metric("Вопросов от начинающих", int(by_level.get("beginner", 0)))
        col3.metric("Вопросов от продвинутых", int(by_level.get("advanced", 0)))

        st.divider()

        # Перевод значений для отображения на графиках
        LEVEL_RU = {"beginner": "Начинающий", "advanced": "Продвинутый"}
        TOPIC_RU = {
            "deadlines": "Дедлайны",
            "assignments": "Задания",
            "rag": "RAG",
            "prompts": "Промпты",
            "resources": "Ресурсы",
            "general": "Общее",
        }

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("По уровню студента")
            if not by_level.empty:
                by_level_ru = by_level.rename(index=lambda x: LEVEL_RU.get(x, x))
                st.bar_chart(by_level_ru)
            else:
                st.write("Нет данных.")

        with col_right:
            st.subheader("По теме вопроса")
            by_topic = analytics.questions_by_topic(df_log)
            if not by_topic.empty:
                by_topic_ru = by_topic.rename(index=lambda x: TOPIC_RU.get(x, x))
                st.bar_chart(by_topic_ru)
            else:
                st.write("Нет данных.")

        st.divider()
        st.subheader("Последние 10 вопросов")
        recent = analytics.recent_questions(df_log, n=10)
        # Переименовываем колонки для русскоязычного отображения
        COL_RU = {
            "timestamp": "Дата/время",
            "question": "Вопрос",
            "learner_level": "Уровень",
            "topic_guess": "Тема",
        }
        recent_display = recent.rename(columns=COL_RU)
        if "Уровень" in recent_display.columns:
            recent_display["Уровень"] = recent_display["Уровень"].map(
                lambda x: LEVEL_RU.get(x, x)
            )
        if "Тема" in recent_display.columns:
            recent_display["Тема"] = recent_display["Тема"].map(
                lambda x: TOPIC_RU.get(x, x)
            )
        st.dataframe(
            recent_display,
            use_container_width=True,
            hide_index=True,
        )
