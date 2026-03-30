"""
tests/test_sentiment.py
=======================
Unit tests for analysis/sentiment.py.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from analysis.sentiment import SentimentAnalyser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "source": ["google_play", "app_store", "csv"],
            "review_text": [
                "This app is absolutely amazing!",
                "Terrible experience, keeps crashing.",
                "It is okay, nothing special.",
            ],
            "rating": [5, 1, 3],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )


@pytest.fixture()
def mock_pipeline_output():
    """Simulate HuggingFace pipeline output."""
    return [
        {"label": "POSITIVE", "score": 0.9987},
        {"label": "NEGATIVE", "score": 0.9876},
        {"label": "NEGATIVE", "score": 0.5123},
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSentimentAnalyser:

    @patch("analysis.sentiment.pipeline")
    def test_analyse_returns_expected_columns(
        self, mock_pl, sample_df, mock_pipeline_output
    ):
        """analyse() should append the three sentiment columns."""
        mock_instance = MagicMock()
        mock_instance.return_value = mock_pipeline_output
        mock_pl.return_value = mock_instance

        # Reset singleton between tests
        SentimentAnalyser._instance = None
        analyser = SentimentAnalyser()
        result = analyser.analyse(sample_df)

        assert "sentiment_label" in result.columns
        assert "confidence_score" in result.columns
        assert "sentiment_score" in result.columns

    @patch("analysis.sentiment.pipeline")
    def test_sentiment_score_range(self, mock_pl, sample_df, mock_pipeline_output):
        """sentiment_score must be in [-1, +1]."""
        mock_instance = MagicMock()
        mock_instance.return_value = mock_pipeline_output
        mock_pl.return_value = mock_instance

        SentimentAnalyser._instance = None
        analyser = SentimentAnalyser()
        result = analyser.analyse(sample_df)

        assert result["sentiment_score"].between(-1, 1).all()

    @patch("analysis.sentiment.pipeline")
    def test_confidence_score_range(self, mock_pl, sample_df, mock_pipeline_output):
        """confidence_score must be in [0, 1]."""
        mock_instance = MagicMock()
        mock_instance.return_value = mock_pipeline_output
        mock_pl.return_value = mock_instance

        SentimentAnalyser._instance = None
        analyser = SentimentAnalyser()
        result = analyser.analyse(sample_df)

        assert result["confidence_score"].between(0, 1).all()

    @patch("analysis.sentiment.pipeline")
    def test_empty_dataframe_raises(self, mock_pl):
        """analyse() must raise ValueError on an empty DataFrame."""
        mock_pl.return_value = MagicMock()
        SentimentAnalyser._instance = None
        analyser = SentimentAnalyser()

        with pytest.raises(ValueError, match="empty"):
            analyser.analyse(pd.DataFrame())

    @patch("analysis.sentiment.pipeline")
    def test_missing_column_raises(self, mock_pl):
        """analyse() must raise ValueError when review_text is absent."""
        mock_pl.return_value = MagicMock()
        SentimentAnalyser._instance = None
        analyser = SentimentAnalyser()

        with pytest.raises(ValueError, match="review_text"):
            analyser.analyse(pd.DataFrame({"other_col": ["x"]}))

    @patch("analysis.sentiment.pipeline")
    def test_positive_label_maps_to_positive_score(self, mock_pl, sample_df):
        """POSITIVE label should produce a positive sentiment_score."""
        mock_instance = MagicMock()
        mock_instance.return_value = [
            {"label": "POSITIVE", "score": 0.95},
            {"label": "POSITIVE", "score": 0.80},
            {"label": "POSITIVE", "score": 0.60},
        ]
        mock_pl.return_value = mock_instance

        SentimentAnalyser._instance = None
        analyser = SentimentAnalyser()
        result = analyser.analyse(sample_df)

        assert (result["sentiment_score"] > 0).all()

    @patch("analysis.sentiment.pipeline")
    def test_negative_label_maps_to_negative_score(self, mock_pl, sample_df):
        """NEGATIVE label should produce a negative sentiment_score."""
        mock_instance = MagicMock()
        mock_instance.return_value = [
            {"label": "NEGATIVE", "score": 0.95},
            {"label": "NEGATIVE", "score": 0.80},
            {"label": "NEGATIVE", "score": 0.60},
        ]
        mock_pl.return_value = mock_instance

        SentimentAnalyser._instance = None
        analyser = SentimentAnalyser()
        result = analyser.analyse(sample_df)

        assert (result["sentiment_score"] < 0).all()

    @patch("analysis.sentiment.pipeline")
    def test_singleton_returns_same_instance(self, mock_pl):
        """get_instance() must return the same object on repeated calls."""
        mock_pl.return_value = MagicMock()
        SentimentAnalyser._instance = None

        inst1 = SentimentAnalyser.get_instance()
        inst2 = SentimentAnalyser.get_instance()
        assert inst1 is inst2

    @patch("analysis.sentiment.pipeline")
    def test_row_count_preserved(self, mock_pl, sample_df, mock_pipeline_output):
        """Output DataFrame must have same row count as input."""
        mock_instance = MagicMock()
        mock_instance.return_value = mock_pipeline_output
        mock_pl.return_value = mock_instance

        SentimentAnalyser._instance = None
        analyser = SentimentAnalyser()
        result = analyser.analyse(sample_df)

        assert len(result) == len(sample_df)