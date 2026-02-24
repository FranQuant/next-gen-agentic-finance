from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

import pandas as pd

from core.adapters.fred_adapter import FredMacroAdapter
from core.adapters.market_data import MarketDataAdapter


class DeterministicFeatureBuilder:
    """Build a deterministic macro-market feature snapshot for a date range.

    This service is intentionally side-effect free:
    - no implicit dates
    - no global state
    - deterministic sorting/alignment by daily date index
    """

    def __init__(
        self,
        market_adapter: MarketDataAdapter,
        macro_adapter: FredMacroAdapter,
    ) -> None:
        """Initialize the feature builder with market and macro adapters.

        Raises:
            ValueError: If any adapter is missing.
        """
        if market_adapter is None:
            raise ValueError("market_adapter is required.")
        if macro_adapter is None:
            raise ValueError("macro_adapter is required.")

        self._market_adapter = market_adapter
        self._macro_adapter = macro_adapter

    def build_features(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
    ) -> Dict[str, float]:
        """Build deterministic features for an instrument and date range.

        Features returned:
        - ``spx_20d_return``
        - ``spx_above_200dma``
        - ``dgs10_level``
        - ``dgs10_20d_change``

        Raises:
            ValueError: If required inputs are missing/invalid, required columns
                are absent, aligned data is insufficient, or feature computation
                cannot produce valid values.
        """
        if not instrument:
            raise ValueError("instrument is required.")
        if start is None or end is None:
            raise ValueError("Explicit start and end datetimes are required.")
        if end < start:
            raise ValueError("end must be greater than or equal to start.")
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware UTC datetimes.")
        if start.tzinfo.utcoffset(start) != timedelta(0) or end.tzinfo.utcoffset(end) != timedelta(0):
            raise ValueError("start and end must be in UTC.")

        candles = self._market_adapter.get_history(
            instrument=instrument,
            start=start,
            end=end,
            granularity="D",
        )
        if candles is None or candles.empty:
            raise ValueError(
                f"No candle history returned for instrument={instrument}, "
                f"start={start.isoformat()}, end={end.isoformat()}."
            )
        if not isinstance(candles, pd.DataFrame):
            raise ValueError("Market history must be returned as a pandas DataFrame.")

        close_col = None
        if "c" in candles.columns:
            close_col = "c"
        elif "close" in candles.columns:
            close_col = "close"
        if close_col is None:
            raise ValueError(
                "Market history must include a close column named 'c' or 'close'."
            )

        close = self._normalize_daily_series(candles[close_col].copy())

        dgs10 = self._macro_adapter.get_series(series_id="DGS10", start=start, end=end)
        if dgs10 is None or dgs10.empty:
            raise ValueError(
                f"No DGS10 data returned for start={start.isoformat()}, "
                f"end={end.isoformat()}."
            )
        if not isinstance(dgs10, pd.Series):
            raise ValueError("DGS10 data must be returned as a pandas Series.")
        dgs10 = self._normalize_daily_series(dgs10.copy())

        aligned = pd.concat([close.rename("close"), dgs10.rename("dgs10")], axis=1)
        aligned = aligned.dropna(how="any")
        aligned = aligned.sort_index()

        if len(aligned) < 200:
            raise ValueError(
                f"Aligned data must contain at least 200 rows; got {len(aligned)}."
            )

        rolling_200dma = aligned["close"].rolling(window=200, min_periods=200).mean()
        if pd.isna(rolling_200dma.iloc[-1]):
            raise ValueError("Unable to compute 200-day moving average at latest row.")

        latest_close = float(aligned["close"].iloc[-1])
        prior_close_20d = float(aligned["close"].iloc[-21])
        if prior_close_20d == 0.0:
            raise ValueError("Cannot compute 20-day return because prior close is zero.")

        latest_dgs10 = float(aligned["dgs10"].iloc[-1])
        dgs10_20d_ago = float(aligned["dgs10"].iloc[-21])

        features: Dict[str, float] = {
            "spx_20d_return": (latest_close / prior_close_20d) - 1.0,
            "spx_above_200dma": 1.0
            if latest_close > float(rolling_200dma.iloc[-1])
            else 0.0,
            "dgs10_level": latest_dgs10,
            "dgs10_20d_change": latest_dgs10 - dgs10_20d_ago,
        }

        if set(features.keys()) != {
            "spx_20d_return",
            "spx_above_200dma",
            "dgs10_level",
            "dgs10_20d_change",
        }:
            raise ValueError("Feature output keys do not match the required contract.")

        if any(pd.isna(value) for value in features.values()):
            raise ValueError("Feature computation produced NaN values.")

        return features

    @staticmethod
    def _normalize_daily_series(series: pd.Series) -> pd.Series:
        """Normalize series index to deterministic daily, tz-naive dates."""
        index = pd.to_datetime(series.index)
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("Series index must be convertible to a DatetimeIndex.")
        if index.tz is None:
            raise ValueError("All series indices must be timezone-aware UTC.")
        if index.tz.utcoffset(index[0]) != timedelta(0):
            raise ValueError("All series indices must be UTC.")
        index = index.tz_convert("UTC")

        series.index = index.normalize()
        series = series.sort_index()
        series = series[~series.index.duplicated(keep="last")]
        return series.dropna()
