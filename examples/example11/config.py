from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Example11Config:
    db_path: str = "tmp/example11_runs.db"
    history_limit: int = 5
    top_topics: int = 4
    default_tickers: tuple[str, ...] = ("SPY", "QQQ", "TLT")
    default_macro_indicators: tuple[str, ...] = (
        "inflation",
        "unemployment",
        "policy_rate",
        "10y_yield",
    )
    mcp_server_url: str | None = None
    mcp_timeout_sec: int = 8
    use_mcp_live: bool = False


def load_config() -> Example11Config:
    use_mcp_live = os.getenv("EXAMPLE11_USE_MCP_LIVE", "0").strip().lower() in {"1", "true", "yes"}

    return Example11Config(
        db_path=os.getenv("EXAMPLE11_DB_PATH", "tmp/example11_runs.db"),
        history_limit=int(os.getenv("EXAMPLE11_HISTORY_LIMIT", "5")),
        top_topics=int(os.getenv("EXAMPLE11_TOP_TOPICS", "4")),
        mcp_server_url=os.getenv("EXAMPLE11_MCP_SERVER_URL"),
        mcp_timeout_sec=int(os.getenv("EXAMPLE11_MCP_TIMEOUT_SEC", "8")),
        use_mcp_live=use_mcp_live,
    )
