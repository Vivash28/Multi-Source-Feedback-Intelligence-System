"""
fetchers/google_play.py
=======================
Fetches app reviews from the Google Play Store using the
``google-play-scraper`` library.  Results are normalised into
the canonical review schema defined in config.REVIEW_SCHEMA_COLUMNS.
"""

import time
import uuid
from typing import List, Dict, Any, Optional

import pandas as pd
from google_play_scraper import reviews, Sort
from google_play_scraper.exceptions import NotFoundError

import config
from utils.logger import get_logger

logger = get_logger(__name__)


class GooglePlayFetcher:
    """
    Fetches and normalises Google Play Store reviews for a given app.

    Parameters
    ----------
    app_id : str
        The package name of the Android app (e.g. ``'com.spotify.music'``).
    lang : str
        Language code for reviews (default: ``'en'``).
    country : str
        Country code for the store (default: ``'us'``).
    count : int
        Number of reviews to retrieve.  Defaults to
        ``config.GOOGLE_PLAY_DEFAULT_COUNT``.
    """

    def __init__(
        self,
        app_id: str,
        lang: str = "en",
        country: str = "us",
        count: int = config.GOOGLE_PLAY_DEFAULT_COUNT,
    ) -> None:
        self.app_id = app_id
        self.lang = lang
        self.country = country
        self.count = count

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self) -> pd.DataFrame:
        """
        Retrieve reviews and return a normalised DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: review_id, source, review_text, rating, date.

        Raises
        ------
        ValueError
            If ``app_id`` is empty or the app is not found in the store.
        RuntimeError
            If the API fails after all retry attempts.
        """
        if not self.app_id:
            raise ValueError("app_id must not be empty.")

        raw_reviews = self._fetch_with_retry()
        if not raw_reviews:
            logger.warning(
                "No reviews returned for app_id=%s. Returning empty DataFrame.",
                self.app_id,
            )
            return self._empty_dataframe()

        df = self._normalise(raw_reviews)
        logger.info(
            "GooglePlayFetcher: fetched %d reviews for app_id=%s",
            len(df),
            self.app_id,
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_with_retry(self) -> List[Dict[str, Any]]:
        """
        Call the google-play-scraper API with exponential back-off retry.

        Returns
        -------
        List[Dict[str, Any]]
            Raw review dicts from the scraper.

        Raises
        ------
        ValueError
            If the app is not found.
        RuntimeError
            If retries are exhausted.
        """
        delay = 1.0
        for attempt in range(1, config.FETCH_MAX_RETRIES + 1):
            try:
                logger.debug(
                    "Fetching Google Play reviews (attempt %d/%d)...",
                    attempt,
                    config.FETCH_MAX_RETRIES,
                )
                result, _ = reviews(
                    self.app_id,
                    lang=self.lang,
                    country=self.country,
                    sort=Sort.NEWEST,
                    count=self.count,
                )
                return result  # type: ignore[return-value]

            except NotFoundError as exc:
                raise ValueError(
                    f"App '{self.app_id}' not found on Google Play Store."
                ) from exc

            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Google Play fetch attempt %d failed: %s", attempt, exc
                )
                if attempt < config.FETCH_MAX_RETRIES:
                    time.sleep(delay)
                    delay *= config.FETCH_RETRY_BACKOFF
                else:
                    raise RuntimeError(
                        f"Failed to fetch Google Play reviews after "
                        f"{config.FETCH_MAX_RETRIES} attempts."
                    ) from exc

        return []  # unreachable, satisfies type checker

    def _normalise(self, raw: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Map raw scraper dicts to the canonical review schema.

        Parameters
        ----------
        raw : List[Dict[str, Any]]
            Raw review objects from google-play-scraper.

        Returns
        -------
        pd.DataFrame
            Normalised DataFrame with columns defined in
            config.REVIEW_SCHEMA_COLUMNS.
        """
        records = []
        for item in raw:
            review_text: str = (item.get("content") or "").strip()
            if not review_text:
                continue  # skip blank reviews

            records.append(
                {
                    "review_id": item.get("reviewId") or str(uuid.uuid4()),
                    "source": "google_play",
                    "review_text": review_text,
                    "rating": int(item.get("score", 0)),
                    "date": pd.to_datetime(item.get("at")).normalize(),
                }
            )

        df = pd.DataFrame(records, columns=config.REVIEW_SCHEMA_COLUMNS)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
        return df

    @staticmethod
    def _empty_dataframe() -> pd.DataFrame:
        """Return an empty DataFrame with the canonical schema."""
        return pd.DataFrame(columns=config.REVIEW_SCHEMA_COLUMNS)