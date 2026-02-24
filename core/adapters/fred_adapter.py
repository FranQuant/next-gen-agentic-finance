from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from fredapi import Fred


class FredMacroAdapter:
    """Adapter for fetching macroeconomic time series from the FRED API.

    The adapter has deterministic behavior:
    - API credentials are read once during initialization.
    - Series requests require explicit start and end datetimes.
    - No implicit "today" or moving time window is used.
    """

    def __init__(self) -> None:
        """Initialize a FRED client from ``FRED_API_KEY``.

        Raises:
            ValueError: If ``FRED_API_KEY`` is not set.
        """
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise ValueError("FRED_API_KEY environment variable is required.")

        self._client = Fred(api_key=api_key)

    def get_series(
        self,
        series_id: str,
        start: datetime,
        end: datetime,
    ) -> pd.Series:
        """Fetch a FRED series for an explicit date range.

        Args:
            series_id: FRED series identifier (for example, ``"UNRATE"``).
            start: Inclusive observation start datetime.
            end: Inclusive observation end datetime.

        Returns:
            pandas Series containing observations indexed by datetime.

        Raises:
            ValueError: If ``series_id`` is empty, if ``start``/``end`` are missing,
                if ``end`` is earlier than ``start``, or if no non-null observations
                are returned.
        """
        if not series_id:
            raise ValueError("series_id is required.")
        if start is None or end is None:
            raise ValueError("Explicit start and end datetimes are required.")
        if end < start:
            raise ValueError("end must be greater than or equal to start.")

        series = self._client.get_series(
            series_id,
            observation_start=start,
            observation_end=end,
        )

        if series is None:
            raise ValueError(
                f"No data returned for series_id={series_id}, "
                f"start={start.isoformat()}, end={end.isoformat()}."
            )

        cleaned = series.dropna()
        cleaned.index = pd.to_datetime(cleaned.index)
        cleaned = cleaned.sort_index()

        if cleaned.empty:
            raise ValueError(
                f"No non-null observations for series_id={series_id}, "
                f"start={start.isoformat()}, end={end.isoformat()}."
            )

        cleaned.index = pd.to_datetime(cleaned.index, utc=True)
        return cleaned
