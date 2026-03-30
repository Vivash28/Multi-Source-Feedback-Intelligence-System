"""
tests/test_prioritization.py
=============================
Unit tests for analysis/issue_prioritizer.py.
"""

import pytest
import pandas as pd

from analysis.issue_prioritizer import IssuePrioritizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def negative_reviews_df() -> pd.DataFrame:
    """DataFrame with clearly negative reviews containing complaint keywords."""
    texts = [
        "The app keeps crashing every time I open it. The crash is unbearable.",
        "Terrible crash experience, the login fails constantly.",
        "Login is broken and the app crashes on startup.",
        "Cannot login, crash crash crash! Completely unusable.",
        "Slow loading and frequent crash issues with login.",
        "Payment failed again. This payment problem is recurring.",
        "Payment errors are frustrating and the crash happens often.",
    ]
    return pd.DataFrame(
        {
            "review_id": [str(i) for i in range(len(texts))],
            "source": ["google_play"] * len(texts),
            "review_text": texts,
            "rating": [1] * len(texts),
            "date": pd.date_range("2024-01-01", periods=len(texts)),
            "sentiment_label": ["NEGATIVE"] * len(texts),
            "sentiment_score": [-0.9, -0.85, -0.88, -0.95, -0.7, -0.8, -0.75],
        }
    )


@pytest.fixture()
def positive_only_df() -> pd.DataFrame:
    """All-positive DataFrame — should produce empty issue list."""
    return pd.DataFrame(
        {
            "review_id": ["p1", "p2"],
            "source": ["app_store", "app_store"],
            "review_text": ["Great app!", "Love it so much!"],
            "rating": [5, 5],
            "date": pd.date_range("2024-01-01", periods=2),
            "sentiment_label": ["POSITIVE", "POSITIVE"],
            "sentiment_score": [0.95, 0.90],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIssuePrioritizer:

    def test_returns_dataframe(self, negative_reviews_df):
        """prioritize() must return a DataFrame."""
        p = IssuePrioritizer(min_freq=2)
        result = p.prioritize(negative_reviews_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self, negative_reviews_df):
        """Result must have the four canonical columns plus rank."""
        p = IssuePrioritizer(min_freq=2)
        result = p.prioritize(negative_reviews_df)
        for col in ["keyword", "frequency", "avg_neg_strength",
                    "priority_score", "rank"]:
            assert col in result.columns

    def test_top_keyword_is_crash(self, negative_reviews_df):
        """'crash' should be the highest-priority keyword in the fixture."""
        p = IssuePrioritizer(min_freq=2)
        result = p.prioritize(negative_reviews_df)
        assert result.iloc[0]["keyword"] == "crash"

    def test_priority_score_descending(self, negative_reviews_df):
        """Issues must be sorted by priority_score descending."""
        p = IssuePrioritizer(min_freq=2)
        result = p.prioritize(negative_reviews_df)
        scores = result["priority_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_positive_only_returns_empty(self, positive_only_df):
        """Only positive reviews should yield an empty result DataFrame."""
        p = IssuePrioritizer(min_freq=1)
        result = p.prioritize(positive_only_df)
        assert result.empty

    def test_min_freq_filter_respected(self, negative_reviews_df):
        """Keywords below min_freq must not appear in results."""
        p = IssuePrioritizer(min_freq=10)  # very high threshold
        result = p.prioritize(negative_reviews_df)
        # With freq >= 10 threshold and small fixture, result should be empty
        assert result.empty

    def test_top_n_respected(self, negative_reviews_df):
        """Result must contain at most top_n rows."""
        p = IssuePrioritizer(top_n=2, min_freq=1)
        result = p.prioritize(negative_reviews_df)
        assert len(result) <= 2

    def test_missing_columns_raises(self):
        """prioritize() must raise ValueError on missing required columns."""
        p = IssuePrioritizer()
        df = pd.DataFrame({"review_text": ["bad app"]})
        with pytest.raises(ValueError, match="missing columns"):
            p.prioritize(df)

    def test_rank_starts_at_one(self, negative_reviews_df):
        """The first-ranked issue must have rank == 1."""
        p = IssuePrioritizer(min_freq=2)
        result = p.prioritize(negative_reviews_df)
        if not result.empty:
            assert result.iloc[0]["rank"] == 1

    def test_frequency_is_positive_integer(self, negative_reviews_df):
        """All frequency values must be positive integers."""
        p = IssuePrioritizer(min_freq=2)
        result = p.prioritize(negative_reviews_df)
        assert (result["frequency"] > 0).all()
        assert result["frequency"].dtype in [int, "int64", "int32"]