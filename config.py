"""
config.py
=========
Central configuration for the Multi-Source Feedback Intelligence System.
All tuneable parameters, paths, and constants live here.
"""

import os
from typing import List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
REPORTS_DIR: str = os.path.join(BASE_DIR, "reports")
CACHE_DIR: str = os.path.join(BASE_DIR, ".cache")

for _dir in (DATA_DIR, REPORTS_DIR, CACHE_DIR):
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Sentiment model
# ---------------------------------------------------------------------------
SENTIMENT_MODEL_NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_BATCH_SIZE: int = 32
SENTIMENT_MAX_LENGTH: int = 512

# ---------------------------------------------------------------------------
# Fetcher settings
# ---------------------------------------------------------------------------
GOOGLE_PLAY_DEFAULT_COUNT: int = 150
APP_STORE_DEFAULT_COUNT: int = 150
FETCH_TIMEOUT_SECONDS: int = 30
FETCH_MAX_RETRIES: int = 3
FETCH_RETRY_BACKOFF: float = 2.0

# ---------------------------------------------------------------------------
# Trend detection thresholds
# ---------------------------------------------------------------------------
ROLLING_WINDOW_DAYS: int = 7
SENTIMENT_DROP_THRESHOLD: float = 0.20
SENTIMENT_DROP_WINDOW_DAYS: int = 3

# ---------------------------------------------------------------------------
# Issue prioritisation
# ---------------------------------------------------------------------------
TOP_N_ISSUES: int = 15
MIN_KEYWORD_FREQ: int = 3
STOP_WORDS: List[str] = [
    "app", "the", "this", "that", "is", "it", "in", "to", "a", "an",
    "and", "or", "but", "of", "for", "with", "on", "at", "by", "from",
    "not", "no", "are", "was", "be", "as", "i", "we", "you", "have",
    "has", "had", "do", "does", "can", "my", "me", "its", "so", "very",
    "just", "all", "get", "got", "please", "update", "version",
]

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
DASHBOARD_TITLE: str = "Multi-Source Feedback Intelligence System"
DASHBOARD_PAGE_ICON: str = "📊"
DEFAULT_DATE_RANGE_DAYS: int = 30

# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
PDF_COMPANY_NAME: str = os.getenv("COMPANY_NAME", "Your Company")
PDF_LOGO_PATH: str = os.getenv("PDF_LOGO_PATH", "")
PDF_ACCENT_COLOR_HEX: str = "#2980B9"
PDF_PAGE_SIZE: str = "A4"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = os.path.join(BASE_DIR, "feedback_system.log")

# ---------------------------------------------------------------------------
# Canonical review schema columns
# ---------------------------------------------------------------------------
REVIEW_SCHEMA_COLUMNS: List[str] = [
    "review_id",
    "source",
    "review_text",
    "rating",
    "date",
]