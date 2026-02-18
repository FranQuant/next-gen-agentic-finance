from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import tpqoa

from core.adapters.market_data import MarketDataAdapter


class OandaMarketDataAdapter(MarketDataAdapter):
    """Market data adapter backed by Oanda via ``tpqoa``.

    This adapter is responsible only for fetching raw market data from Oanda:
    - historical candles for a time range
    - the latest available close price for an instrument
    """

    def __init__(self) -> None:
        """Initialize the Oanda client using ``OANDA_CFG_PATH``."""
        cfg_path = os.getenv("OANDA_CFG_PATH")
        if not cfg_path:
            raise ValueError("OANDA_CFG_PATH environment variable is required.")
        self._client = tpqoa.tpqoa(cfg_path)

    def get_history(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
        granularity: str,
    ) -> pd.DataFrame:
        """Fetch historical candles for ``instrument`` from Oanda.

        Raises:
            ValueError: If no rows are returned by the provider.
        """
        history = self._client.get_history(
            instrument=instrument,
            start=start,
            end=end,
            granularity=granularity,
            price="M",
        )
        if history is None or history.empty:
            raise ValueError(
                f"No history returned for instrument={instrument}, "
                f"start={start.isoformat()}, end={end.isoformat()}, "
                f"granularity={granularity}."
            )
        return history

    def get_latest_price(self, instrument: str) -> float:
        """Fetch and return the latest close price for ``instrument``.

        Raises:
            ValueError: If price data is unavailable.
        """
        latest = self._client.get_history(
            instrument=instrument,
            granularity="M1",
            count=1,
        )
        if latest is None or latest.empty:
            raise ValueError(f"No latest price data returned for instrument={instrument}.")

        if "c" in latest.columns:
            close_value = latest["c"].iloc[-1]
        elif "close" in latest.columns:
            close_value = latest["close"].iloc[-1]
        else:
            raise ValueError(
                f"Latest price data for instrument={instrument} has no close column."
            )

        return float(close_value)
