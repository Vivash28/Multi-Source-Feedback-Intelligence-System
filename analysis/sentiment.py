"""
analysis/sentiment.py
=====================
Sentiment analysis using the HuggingFace ``distilbert-base-uncased-
finetuned-sst-2-english`` model.

Key design decisions
--------------------
* **Singleton pattern** — the model is loaded once per process and
  reused across calls (``SentimentAnalyser.get_instance()``).
* **Batched inference** — reviews are processed in configurable
  batches to balance memory and throughput.
* Output adds three columns to the input DataFrame:
    - ``sentiment_label``  : ``POSITIVE`` | ``NEGATIVE``
    - ``confidence_score`` : float in [0, 1]
    - ``sentiment_score``  : float in [-1, +1]
"""

from __future__ import annotations

import threading
from typing import Optional, List

import pandas as pd
from transformers import pipeline, Pipeline
from tqdm import tqdm

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalyser:
    """
    Thread-safe singleton wrapper around a HuggingFace sentiment pipeline.

    Use ``SentimentAnalyser.get_instance()`` rather than constructing
    directly to ensure only one model is loaded per process.
    """

    _instance: Optional["SentimentAnalyser"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        logger.info(
            "Loading sentiment model '%s'…", config.SENTIMENT_MODEL_NAME
        )
        self._pipe: Pipeline = pipeline(
            task="sentiment-analysis",
            model=config.SENTIMENT_MODEL_NAME,
            truncation=True,
            max_length=config.SENTIMENT_MAX_LENGTH,
        )
        logger.info("Sentiment model loaded successfully.")

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "SentimentAnalyser":
        """
        Return (or create) the singleton instance.

        Thread-safe via double-checked locking.

        Returns
        -------
        SentimentAnalyser
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyse(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run sentiment analysis on all rows in ``df``.

        The input DataFrame must contain a ``review_text`` column.
        Three new columns are appended in-place:
        ``sentiment_label``, ``confidence_score``, ``sentiment_score``.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least a ``review_text`` column.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with three additional columns.

        Raises
        ------
        ValueError
            If ``review_text`` column is absent or DataFrame is empty.
        """
        if df.empty:
            raise ValueError("Cannot analyse an empty DataFrame.")
        if "review_text" not in df.columns:
            raise ValueError("DataFrame must contain a 'review_text' column.")

        texts: List[str] = (
            df["review_text"]
            .fillna("")
            .astype(str)
            .str.strip()
            .tolist()
        )

        labels: List[str] = []
        confidences: List[float] = []
        scores: List[float] = []

        batch_size = config.SENTIMENT_BATCH_SIZE
        batches = [
            texts[i: i + batch_size]
            for i in range(0, len(texts), batch_size)
        ]

        logger.info(
            "Running sentiment analysis on %d reviews in %d batches…",
            len(texts),
            len(batches),
        )

        for batch in tqdm(batches, desc="Sentiment", unit="batch"):
            try:
                results = self._pipe(batch)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Batch inference failed: %s. Filling with defaults.", exc)
                results = [{"label": "NEGATIVE", "score": 0.5}] * len(batch)

            for res in results:
                label: str = res.get("label", "NEGATIVE").upper()
                confidence: float = round(float(res.get("score", 0.5)), 4)
                # Map to [-1, +1]
                sentiment_score: float = round(
                    confidence if label == "POSITIVE" else -confidence, 4
                )
                labels.append(label)
                confidences.append(confidence)
                scores.append(sentiment_score)

        df = df.copy()
        df["sentiment_label"] = labels
        df["confidence_score"] = confidences
        df["sentiment_score"] = scores

        logger.info("Sentiment analysis complete.")
        return df

    # ------------------------------------------------------------------
    # Convenience class method
    # ------------------------------------------------------------------

    @classmethod
    def analyse_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shortcut: obtain the singleton and run ``analyse(df)``.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        return cls.get_instance().analyse(df)