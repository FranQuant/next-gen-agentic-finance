from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd


class MarketDataAdapter(Protocol):
    """
    Contract for market data providers used by the research pipeline.

    Implementations are responsible for:
    - Fetching historical OHLCV data.
    - Providing the latest tradable price.
    - Normalizing provider-specific formats into stable pandas DataFrames.
    """

    def get_history(
        self,
        instrument: str,
        start: datetime,
        end: datetime,
        granularity: str,
    ) -> pd.DataFrame:
        """
        Return historical OHLCV data for `instrument`.

        Expected output:
            - pandas DataFrame
            - datetime index
            - columns: ['o', 'h', 'l', 'c', 'volume']
        """

    def get_latest_price(self, instrument: str) -> float:
        """
        Return the most recent available price for `instrument`.

        Must be sourced from real provider data.
        """

