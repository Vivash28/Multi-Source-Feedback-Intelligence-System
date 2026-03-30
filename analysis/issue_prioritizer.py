"""
analysis/issue_prioritizer.py
==============================
Extracts and prioritises complaint keywords from negative reviews.

Priority Score
--------------
    Priority Score = Frequency × mean(|sentiment_score|)

The higher the score, the more frequently the keyword appears in
reviews AND the more negatively those reviews are rated.
"""

import re
from collections import Counter
from typing import List, Dict, Any

import pandas as pd

import config
from utils.logger import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z]{3,}")   # only words ≥ 3 chars, alpha only


class IssuePrioritizer:
    """
    Identifies and ranks the most critical user complaints.

    Parameters
    ----------
    top_n : int
        Number of top issues to return.
    min_freq : int
        Minimum keyword frequency to be included in results.
    stop_words : List[str]
        Words to exclude from keyword extraction.
    """

    def __init__(
        self,
        top_n: int = config.TOP_N_ISSUES,
        min_freq: int = config.MIN_KEYWORD_FREQ,
        stop_words: List[str] = config.STOP_WORDS,
    ) -> None:
        self.top_n = top_n
        self.min_freq = min_freq
        self.stop_words: set = set(w.lower() for w in stop_words)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def prioritize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract, score, and rank issues from negative reviews.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``review_text``, ``sentiment_label``, and
            ``sentiment_score`` columns (added by SentimentAnalyser).

        Returns
        -------
        pd.DataFrame
            Columns: keyword, frequency, avg_neg_strength, priority_score,
            rank.  Sorted descending by priority_score.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        required = {"review_text", "sentiment_label", "sentiment_score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"IssuePrioritizer: missing columns {missing}"
            )

        negative_df = df[df["sentiment_label"] == "NEGATIVE"].copy()

        if negative_df.empty:
            logger.info("No negative reviews found — no issues to prioritise.")
            return pd.DataFrame(
                columns=[
                    "keyword", "frequency", "avg_neg_strength",
                    "priority_score", "rank",
                ]
            )

        keyword_data = self._extract_keywords(negative_df)
        result = self._score_and_rank(keyword_data)

        logger.info(
            "IssuePrioritizer: identified %d prioritised issues.", len(result)
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_keywords(
        self, neg_df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a dict mapping each keyword to its frequency and the list of
        negative sentiment magnitudes for reviews containing it.

        Parameters
        ----------
        neg_df : pd.DataFrame
            Negative-only review subset.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            ``{ keyword: { 'freq': int, 'neg_strengths': [float, ...] } }``
        """
        keyword_data: Dict[str, Dict[str, Any]] = {}

        for _, row in neg_df.iterrows():
            text: str = str(row["review_text"]).lower()
            tokens = _TOKEN_RE.findall(text)
            tokens = [t for t in tokens if t not in self.stop_words]
            neg_strength = abs(float(row["sentiment_score"]))

            seen_in_review: set = set()
            for token in tokens:
                if token not in keyword_data:
                    keyword_data[token] = {"freq": 0, "neg_strengths": []}
                if token not in seen_in_review:
                    keyword_data[token]["freq"] += 1
                    seen_in_review.add(token)
                keyword_data[token]["neg_strengths"].append(neg_strength)

        return keyword_data

    def _score_and_rank(
        self, keyword_data: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Filter by minimum frequency, compute priority scores, and rank.

        Parameters
        ----------
        keyword_data : Dict[str, Dict[str, Any]]
            Output of ``_extract_keywords``.

        Returns
        -------
        pd.DataFrame
        """
        records = []
        for keyword, data in keyword_data.items():
            freq = data["freq"]
            if freq < self.min_freq:
                continue
            strengths = data["neg_strengths"]
            avg_neg = sum(strengths) / len(strengths) if strengths else 0.0
            priority_score = round(freq * avg_neg, 4)
            records.append(
                {
                    "keyword": keyword,
                    "frequency": freq,
                    "avg_neg_strength": round(avg_neg, 4),
                    "priority_score": priority_score,
                }
            )

        if not records:
            return pd.DataFrame(
                columns=[
                    "keyword", "frequency", "avg_neg_strength",
                    "priority_score", "rank",
                ]
            )

        result = (
            pd.DataFrame(records)
            .sort_values("priority_score", ascending=False)
            .head(self.top_n)
            .reset_index(drop=True)
        )
        result["rank"] = result.index + 1
        return result