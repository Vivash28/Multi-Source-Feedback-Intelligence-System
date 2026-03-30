"""
analysis/trend_analysis.py
==========================
Time-series trend detection on sentiment data.

Features
--------
* Daily average sentiment score
* 7-day rolling average
* Alert detection: >20% drop over 3 consecutive days
* Returns both a trend DataFrame and a list of alert dicts
"""

from typing import List, Dict, Any, Tuple

import pandas as pd
import numpy as np

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class TrendAnalyser:
    """
    Computes sentiment trends and raises alerts on significant drops.

    Parameters
    ----------
    rolling_window : int
        Window size (in days) for the rolling average.
        Defaults to ``config.ROLLING_WINDOW_DAYS``.
    drop_threshold : float
        Fractional drop that triggers a "Sentiment Alert"
        (e.g. 0.20 for a 20 % drop).
        Defaults to ``config.SENTIMENT_DROP_THRESHOLD``.
    drop_window : int
        Number of consecutive days over which the drop is measured.
        Defaults to ``config.SENTIMENT_DROP_WINDOW_DAYS``.
    """

    def __init__(
        self,
        rolling_window: int = config.ROLLING_WINDOW_DAYS,
        drop_threshold: float = config.SENTIMENT_DROP_THRESHOLD,
        drop_window: int = config.SENTIMENT_DROP_WINDOW_DAYS,
    ) -> None:
        self.rolling_window = rolling_window
        self.drop_threshold = drop_threshold
        self.drop_window = drop_window

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Compute trends and detect sentiment alerts.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``date`` (datetime) and ``sentiment_score``
            (float) columns.

        Returns
        -------
        Tuple[pd.DataFrame, List[Dict[str, Any]]]
            * trend_df : daily & rolling sentiment averages
            * alerts   : list of alert dicts
              (keys: ``date``, ``drop_pct``, ``message``)

        Raises
        ------
        ValueError
            If required columns are missing or DataFrame is empty.
        """
        self._validate(df)

        trend_df = self._daily_averages(df)
        trend_df = self._rolling_average(trend_df)
        alerts = self._detect_alerts(trend_df)

        if alerts:
            logger.warning("%d sentiment alert(s) detected.", len(alerts))
        else:
            logger.info("No sentiment alerts detected.")

        return trend_df, alerts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        """Raise ValueError if the DataFrame is unusable."""
        if df.empty:
            raise ValueError("TrendAnalyser received an empty DataFrame.")
        required = {"date", "sentiment_score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {missing}"
            )

    def _daily_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-day statistics.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Indexed by ``date`` with columns:
            ``avg_sentiment``, ``review_count``, ``avg_rating``.
        """
        work = df.copy()
        work["date"] = pd.to_datetime(work["date"]).dt.normalize()

        trend = (
            work.groupby("date")
            .agg(
                avg_sentiment=("sentiment_score", "mean"),
                review_count=("sentiment_score", "count"),
                avg_rating=("rating", "mean") if "rating" in work.columns
                else ("sentiment_score", "count"),
            )
            .reset_index()
            .sort_values("date")
        )

        # Fill gaps so rolling window works correctly
        date_range = pd.date_range(
            start=trend["date"].min(),
            end=trend["date"].max(),
            freq="D",
        )
        trend = (
            trend.set_index("date")
            .reindex(date_range)
            .rename_axis("date")
            .reset_index()
        )
        trend["avg_sentiment"] = trend["avg_sentiment"].interpolate(
            method="linear", limit_direction="both"
        )
        trend["review_count"] = trend["review_count"].fillna(0).astype(int)

        return trend

    def _rolling_average(self, trend_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a ``rolling_avg_sentiment`` column.

        Parameters
        ----------
        trend_df : pd.DataFrame
            Output of ``_daily_averages``.

        Returns
        -------
        pd.DataFrame
        """
        trend_df = trend_df.copy()
        trend_df["rolling_avg_sentiment"] = (
            trend_df["avg_sentiment"]
            .rolling(window=self.rolling_window, min_periods=1)
            .mean()
            .round(4)
        )
        return trend_df

    def _detect_alerts(
        self, trend_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Scan for consecutive-day sentiment drops exceeding the threshold.

        Parameters
        ----------
        trend_df : pd.DataFrame
            Must contain ``date`` and ``avg_sentiment``.

        Returns
        -------
        List[Dict[str, Any]]
            Each dict: ``{'date': Timestamp, 'drop_pct': float,
            'message': str}``.
        """
        alerts: List[Dict[str, Any]] = []
        sentiments = trend_df["avg_sentiment"].to_numpy()
        dates = trend_df["date"].to_numpy()

        for i in range(self.drop_window, len(sentiments)):
            start_val = sentiments[i - self.drop_window]
            end_val = sentiments[i]

            if start_val == 0 or np.isnan(start_val) or np.isnan(end_val):
                continue

            drop_pct = (start_val - end_val) / abs(start_val)

            if drop_pct >= self.drop_threshold:
                alert_date = pd.Timestamp(dates[i])
                alerts.append(
                    {
                        "date": alert_date,
                        "drop_pct": round(drop_pct * 100, 1),
                        "message": (
                            f"⚠️ Sentiment Alert on {alert_date.date()}: "
                            f"{round(drop_pct * 100, 1)}% drop over "
                            f"{self.drop_window} days "
                            f"(from {round(start_val, 3)} → "
                            f"{round(end_val, 3)})"
                        ),
                    }
                )

        return alerts