"""
fetchers/app_store.py
=====================
Fetches app reviews from the Apple App Store using the public
RSS JSON feed (no authentication required).

Endpoint template:
  https://itunes.apple.com/{country}/rss/customerreviews/
  page={page}/id={app_id}/sortby=mostrecent/json
"""

import time
import uuid
from typing import List, Dict, Any, Optional

import requests
import pandas as pd

import config
from utils.logger import get_logger

logger = get_logger(__name__)

_APP_STORE_RSS_URL: str = (
    "https://itunes.apple.com/{country}/rss/customerreviews/"
    "page={page}/id={app_id}/sortby=mostrecent/json"
)
_MAX_PAGES: int = 10          # RSS feed exposes at most 10 pages × ~50 reviews


class AppStoreFetcher:
    """
    Fetches and normalises Apple App Store reviews via the iTunes RSS feed.

    Parameters
    ----------
    app_id : str
        Numeric App Store ID (e.g. ``'324684580'`` for Spotify).
    country : str
        Two-letter country code (default: ``'us'``).
    count : int
        Approximate number of reviews to fetch.  Capped at 500 (10 pages).
    """

    def __init__(
        self,
        app_id: str,
        country: str = "us",
        count: int = config.APP_STORE_DEFAULT_COUNT,
    ) -> None:
        if not app_id:
            raise ValueError("app_id must not be empty.")
        self.app_id = app_id
        self.country = country
        self.count = min(count, _MAX_PAGES * 50)

        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "FeedbackIntelligenceSystem/1.0"}
        )

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
            If the app_id is invalid or app is not found.
        RuntimeError
            If the API is unreachable after retries.
        """
        raw_reviews: List[Dict[str, Any]] = []
        pages_needed = max(1, (self.count // 50) + 1)

        for page in range(1, min(pages_needed, _MAX_PAGES) + 1):
            page_reviews = self._fetch_page(page)
            if not page_reviews:
                logger.debug("No reviews on page %d — stopping early.", page)
                break
            raw_reviews.extend(page_reviews)
            if len(raw_reviews) >= self.count:
                break

        if not raw_reviews:
            logger.warning(
                "No App Store reviews found for app_id=%s.", self.app_id
            )
            return self._empty_dataframe()

        df = self._normalise(raw_reviews[: self.count])
        logger.info(
            "AppStoreFetcher: fetched %d reviews for app_id=%s",
            len(df),
            self.app_id,
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_page(self, page: int) -> List[Dict[str, Any]]:
        """
        Fetch a single RSS page with retry logic.

        Parameters
        ----------
        page : int
            RSS page number (1-indexed).

        Returns
        -------
        List[Dict[str, Any]]
            List of raw review entry dicts.
        """
        url = _APP_STORE_RSS_URL.format(
            country=self.country, page=page, app_id=self.app_id
        )
        delay = 1.0

        for attempt in range(1, config.FETCH_MAX_RETRIES + 1):
            try:
                resp = self._session.get(
                    url, timeout=config.FETCH_TIMEOUT_SECONDS
                )
                if resp.status_code == 404:
                    raise ValueError(
                        f"App '{self.app_id}' not found in App Store "
                        f"(country={self.country})."
                    )
                resp.raise_for_status()

                data = resp.json()
                entries = (
                    data.get("feed", {}).get("entry", [])
                )
                # First entry is app metadata, not a review
                if isinstance(entries, list) and len(entries) > 1:
                    return entries[1:]  # skip index 0 (app info)
                return []

            except ValueError:
                raise  # propagate invalid app_id errors immediately

            except (requests.RequestException, Exception) as exc:  # pylint: disable=broad-except
                logger.warning(
                    "App Store page %d attempt %d/%d failed: %s",
                    page, attempt, config.FETCH_MAX_RETRIES, exc,
                )
                if attempt < config.FETCH_MAX_RETRIES:
                    time.sleep(delay)
                    delay *= config.FETCH_RETRY_BACKOFF
                else:
                    raise RuntimeError(
                        f"Failed to fetch App Store page {page} after "
                        f"{config.FETCH_MAX_RETRIES} attempts."
                    ) from exc

        return []

    def _normalise(self, raw: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert raw RSS entry dicts to the canonical schema.

        Parameters
        ----------
        raw : List[Dict[str, Any]]
            Raw RSS entry dicts.

        Returns
        -------
        pd.DataFrame
        """
        records = []
        for entry in raw:
            try:
                review_text = self._extract_label(entry, "content")
                if not review_text:
                    continue

                rating_raw = self._extract_label(
                    entry.get("im:rating", {}), None
                )
                rating = int(rating_raw) if rating_raw else 0

                date_raw = self._extract_label(entry.get("updated", {}), None)
                date = pd.to_datetime(date_raw, utc=True, errors="coerce")
                if pd.isna(date):
                    date = pd.Timestamp.utcnow()

                review_id = self._extract_label(entry.get("id", {}), None)
                if not review_id:
                    review_id = str(uuid.uuid4())

                records.append(
                    {
                        "review_id": review_id,
                        "source": "app_store",
                        "review_text": review_text.strip(),
                        "rating": rating,
                        "date": date.tz_localize(None),
                    }
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("Skipping malformed App Store entry: %s", exc)
                continue

        return pd.DataFrame(records, columns=config.REVIEW_SCHEMA_COLUMNS)

    @staticmethod
    def _extract_label(obj: Any, key: Optional[str]) -> str:
        """
        Extract the ``label`` value from an RSS dict entry.

        Parameters
        ----------
        obj : Any
            A dict potentially containing a ``label`` key, or the value
            directly.
        key : Optional[str]
            If provided, look up this key in ``obj`` first.

        Returns
        -------
        str
        """
        if key:
            obj = obj.get(key, {})
        if isinstance(obj, dict):
            return str(obj.get("label", ""))
        return str(obj) if obj else ""

    @staticmethod
    def _empty_dataframe() -> pd.DataFrame:
        """Return an empty DataFrame with the canonical schema."""
        return pd.DataFrame(columns=config.REVIEW_SCHEMA_COLUMNS)