"""
tests/test_trends.py
====================
Unit tests for analysis/trend_analysis.py.
"""

import pytest
import pandas as pd
import numpy as np

from analysis.trend_analysis import TrendAnalyser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def flat_sentiment_df() -> pd.DataFrame:
    """30 days of stable positive sentiment."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    return pd.DataFrame(
        {
            "review_id": [str(i) for i in range(30)],
            "source": ["google_play"] * 30,
            "review_text": ["good app"] * 30,
            "rating": [4] * 30,
            "date": dates,
            "sentiment_score": [0.8] * 30,
            "sentiment_label": ["POSITIVE"] * 30,
        }
    )


@pytest.fixture()
def drop_sentiment_df() -> pd.DataFrame:
    """Sentiment that drops >20% over 3 days (days 15-17)."""
    scores = [0.7] * 14 + [0.7, 0.5, 0.3] + [0.3] * 13
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    return pd.DataFrame(
        {
            "review_id": [str(i) for i in range(30)],
            "source": ["google_play"] * 30,
            "review_text": ["text"] * 30,
            "rating": [3] * 30,
            "date": dates,
            "sentiment_score": scores,
            "sentiment_label": ["POSITIVE" if s > 0 else "NEGATIVE" for s in scores],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrendAnalyser:

    def test_compute_returns_tuple(self, flat_sentiment_df):
        """compute() should return (DataFrame, list)."""
        analyser = TrendAnalyser()
        result = analyser.compute(flat_sentiment_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_trend_df_has_expected_columns(self, flat_sentiment_df):
        """Trend DataFrame must include key columns."""
        analyser = TrendAnalyser()
        trend_df, _ = analyser.compute(flat_sentiment_df)
        for col in ["date", "avg_sentiment", "rolling_avg_sentiment"]:
            assert col in trend_df.columns

    def test_no_alert_on_flat_sentiment(self, flat_sentiment_df):
        """Stable sentiment should produce zero alerts."""
        analyser = TrendAnalyser()
        _, alerts = analyser.compute(flat_sentiment_df)
        assert alerts == []

    def test_alert_detected_on_drop(self, drop_sentiment_df):
        """A >20% 3-day drop must trigger at least one alert."""
        analyser = TrendAnalyser(drop_threshold=0.20, drop_window=3)
        _, alerts = analyser.compute(drop_sentiment_df)
        assert len(alerts) >= 1

    def test_alert_contains_required_keys(self, drop_sentiment_df):
        """Each alert dict must contain date, drop_pct, message."""
        analyser = TrendAnalyser(drop_threshold=0.20, drop_window=3)
        _, alerts = analyser.compute(drop_sentiment_df)
        for alert in alerts:
            assert "date" in alert
            assert "drop_pct" in alert
            assert "message" in alert

    def test_rolling_avg_length_matches_daily(self, flat_sentiment_df):
        """rolling_avg_sentiment length must equal trend_df length."""
        analyser = TrendAnalyser()
        trend_df, _ = analyser.compute(flat_sentiment_df)
        assert len(trend_df["rolling_avg_sentiment"]) == len(trend_df)

    def test_empty_df_raises(self):
        """compute() must raise ValueError for empty input."""
        analyser = TrendAnalyser()
        with pytest.raises(ValueError):
            analyser.compute(pd.DataFrame())

    def test_missing_sentiment_score_column_raises(self):
        """compute() must raise ValueError when sentiment_score is absent."""
        analyser = TrendAnalyser()
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3)})
        with pytest.raises(ValueError):
            analyser.compute(df)

    def test_daily_avg_is_correct(self, flat_sentiment_df):
        """Daily average sentiment should match the constant input value."""
        analyser = TrendAnalyser()
        trend_df, _ = analyser.compute(flat_sentiment_df)
        # All daily averages should be close to 0.8
        assert np.isclose(
            trend_df["avg_sentiment"].dropna().mean(), 0.8, atol=0.05
        )

    def test_rolling_window_respected(self, flat_sentiment_df):
        """Rolling average should use the configured window size."""
        analyser = TrendAnalyser(rolling_window=3)
        trend_df, _ = analyser.compute(flat_sentiment_df)
        # All values after warm-up should be close to 0.8
        assert np.isclose(
            trend_df["rolling_avg_sentiment"].iloc[-1], 0.8, atol=0.05
        )