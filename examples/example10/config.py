from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Example10Config:
    db_path: str = "tmp/example10_runs.db"
    market_lookback_days: int = 60
    history_limit: int = 5
    top_topics: int = 4
    default_tickers: tuple[str, ...] = ("SPY", "QQQ", "TLT")
    default_macro_indicators: tuple[str, ...] = (
        "inflation",
        "unemployment",
        "policy_rate",
        "10y_yield",
    )
    tavily_api_key: str | None = None
    fred_api_key: str | None = None


def load_config() -> Example10Config:
    return Example10Config(
        db_path=os.getenv("EXAMPLE10_DB_PATH", "tmp/example10_runs.db"),
        market_lookback_days=int(os.getenv("EXAMPLE10_LOOKBACK_DAYS", "60")),
        history_limit=int(os.getenv("EXAMPLE10_HISTORY_LIMIT", "5")),
        top_topics=int(os.getenv("EXAMPLE10_TOP_TOPICS", "4")),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        fred_api_key=os.getenv("FRED_API_KEY"),
    )
