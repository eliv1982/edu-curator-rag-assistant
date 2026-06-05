"""
Analytics: load and summarise data from logs/query_log.csv for the dashboard.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

LOG_FILE = Path("logs/query_log.csv")


def load_logs() -> Optional[pd.DataFrame]:
    """
    Load query log CSV.

    Returns a DataFrame or None if the file doesn't exist or is empty.
    """
    if not LOG_FILE.exists():
        return None
    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def total_questions(df: pd.DataFrame) -> int:
    return len(df)


def questions_by_level(df: pd.DataFrame) -> pd.Series:
    return df["learner_level"].value_counts()


def questions_by_topic(df: pd.DataFrame) -> pd.Series:
    return df["topic_guess"].value_counts()


def recent_questions(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = ["timestamp", "question", "learner_level", "topic_guess"]
    available = [c for c in cols if c in df.columns]
    return df[available].tail(n).iloc[::-1].reset_index(drop=True)
