"""
fetchers/csv_loader.py
======================
Loads survey / export CSV files and normalises them into the canonical
review schema.  Handles common encoding issues, missing columns, and
corrupt data gracefully.

Expected CSV columns (flexible — mapped via ``column_map``):
  - review_id  (optional; auto-generated if absent)
  - review_text (required)
  - rating      (optional; defaults to 0)
  - date        (optional; defaults to today)
  - source      (optional; defaults to 'csv')
"""

import uuid
from typing import Dict, Optional
from pathlib import Path

import pandas as pd

import config
from utils.logger import get_logger

logger = get_logger(__name__)

# Default mapping: canonical name -> list of acceptable CSV column names
DEFAULT_COLUMN_MAP: Dict[str, list] = {
    "review_id":   ["review_id", "id", "ID", "ReviewId"],
    "review_text": ["review_text", "text", "review", "content",
                    "comment", "feedback", "body", "message"],
    "rating":      ["rating", "score", "stars", "Rating", "Score"],
    "date":        ["date", "Date", "timestamp", "Timestamp",
                    "created_at", "submitted_at"],
    "source":      ["source", "Source", "channel"],
}


class CSVLoader:
    """
    Loads and normalises a CSV file containing user reviews or survey data.

    Parameters
    ----------
    filepath : str | Path
        Path to the CSV file.
    column_map : Dict[str, list], optional
        Override the default column-name mapping.  Keys must match the five
        canonical field names.
    source_label : str
        Label to use in the ``source`` column when the CSV does not supply
        one (default: ``'csv'``).
    encoding : str
        File encoding (default: ``'utf-8'``).  Falls back to
        ``'latin-1'`` automatically on decode errors.
    """

    def __init__(
        self,
        filepath: "str | Path",
        column_map: Optional[Dict[str, list]] = None,
        source_label: str = "csv",
        encoding: str = "utf-8",
    ) -> None:
        self.filepath = Path(filepath)
        self.column_map: Dict[str, list] = column_map or DEFAULT_COLUMN_MAP
        self.source_label = source_label
        self.encoding = encoding

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """
        Read, validate, and normalise the CSV file.

        Returns
        -------
        pd.DataFrame
            Columns: review_id, source, review_text, rating, date.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the required ``review_text`` column cannot be found.
        """
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.filepath}"
            )

        raw_df = self._read_csv()
        if raw_df.empty:
            logger.warning("CSV file '%s' is empty.", self.filepath)
            return self._empty_dataframe()

        df = self._map_columns(raw_df)
        df = self._clean(df)

        logger.info(
            "CSVLoader: loaded %d reviews from '%s'.",
            len(df),
            self.filepath.name,
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_csv(self) -> pd.DataFrame:
        """
        Read the CSV file with automatic encoding fallback.

        Returns
        -------
        pd.DataFrame
        """
        encodings_to_try = [self.encoding, "latin-1", "cp1252"]
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(self.filepath, encoding=enc, low_memory=False)
                logger.debug("Read CSV with encoding='%s'.", enc)
                return df
            except UnicodeDecodeError:
                logger.debug(
                    "Encoding '%s' failed for '%s'; trying next.",
                    enc,
                    self.filepath.name,
                )
            except pd.errors.ParserError as exc:
                raise ValueError(
                    f"CSV file '{self.filepath}' appears to be corrupt: {exc}"
                ) from exc

        raise ValueError(
            f"Could not decode '{self.filepath}' with any known encoding."
        )

    def _map_columns(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Map raw DataFrame columns to the canonical schema.

        Parameters
        ----------
        raw : pd.DataFrame
            As-read CSV DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with canonical column names.

        Raises
        ------
        ValueError
            If the ``review_text`` column cannot be found.
        """
        rename: Dict[str, str] = {}
        available_cols = {c.lower(): c for c in raw.columns}

        for canonical, candidates in self.column_map.items():
            for candidate in candidates:
                if candidate.lower() in available_cols:
                    original = available_cols[candidate.lower()]
                    if original != canonical:
                        rename[original] = canonical
                    break

        if rename:
            raw = raw.rename(columns=rename)

        # Check required column
        if "review_text" not in raw.columns:
            raise ValueError(
                f"Could not locate a review text column in '{self.filepath}'. "
                f"Expected one of: {self.column_map['review_text']}. "
                f"Found: {list(raw.columns)}"
            )

        return raw

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean, coerce types, and fill missing values.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with canonical columns (some may be absent).

        Returns
        -------
        pd.DataFrame
            Clean, fully-populated canonical DataFrame.
        """
        # Drop rows with no review text
        df = df.dropna(subset=["review_text"])
        df = df[df["review_text"].astype(str).str.strip() != ""]
        df["review_text"] = df["review_text"].astype(str).str.strip()

        # review_id
        if "review_id" not in df.columns:
            df["review_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        else:
            df["review_id"] = df["review_id"].fillna("").astype(str)
            mask_empty = df["review_id"] == ""
            df.loc[mask_empty, "review_id"] = [
                str(uuid.uuid4()) for _ in range(mask_empty.sum())
            ]

        # source
        if "source" not in df.columns:
            df["source"] = self.source_label
        else:
            df["source"] = df["source"].fillna(self.source_label).astype(str)

        # rating
        if "rating" not in df.columns:
            df["rating"] = 0
        else:
            df["rating"] = (
                pd.to_numeric(df["rating"], errors="coerce")
                .fillna(0)
                .astype(int)
                .clip(0, 5)
            )

        # date
        if "date" not in df.columns:
            df["date"] = pd.Timestamp.today().normalize()
        else:
            df["date"] = pd.to_datetime(
                df["date"], errors="coerce", utc=False
            ).dt.normalize()
            df["date"] = df["date"].fillna(pd.Timestamp.today().normalize())

        return df[config.REVIEW_SCHEMA_COLUMNS].reset_index(drop=True)

    @staticmethod
    def _empty_dataframe() -> pd.DataFrame:
        """Return an empty DataFrame with the canonical schema."""
        return pd.DataFrame(columns=config.REVIEW_SCHEMA_COLUMNS)