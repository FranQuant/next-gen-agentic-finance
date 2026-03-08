from __future__ import annotations

import os
import shlex
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
    mcp_web_transport: str = "stdio"
    mcp_web_server_url: str | None = None
    mcp_web_server_command: str | None = None
    mcp_web_server_args: tuple[str, ...] = ()
    mcp_web_tool_name: str = "web.search"


def load_config() -> Example11Config:
    use_mcp_live = os.getenv("EXAMPLE11_USE_MCP_LIVE", "0").strip().lower() in {"1", "true", "yes"}
    mcp_server_url = os.getenv("EXAMPLE11_MCP_SERVER_URL")
    mcp_web_args_raw = os.getenv("EXAMPLE11_MCP_WEB_SERVER_ARGS", "")

    return Example11Config(
        db_path=os.getenv("EXAMPLE11_DB_PATH", "tmp/example11_runs.db"),
        history_limit=int(os.getenv("EXAMPLE11_HISTORY_LIMIT", "5")),
        top_topics=int(os.getenv("EXAMPLE11_TOP_TOPICS", "4")),
        mcp_server_url=mcp_server_url,
        mcp_timeout_sec=int(os.getenv("EXAMPLE11_MCP_TIMEOUT_SEC", "8")),
        use_mcp_live=use_mcp_live,
        mcp_web_transport=os.getenv("EXAMPLE11_MCP_WEB_TRANSPORT", "stdio").strip().lower(),
        mcp_web_server_url=(os.getenv("EXAMPLE11_MCP_WEB_SERVER_URL") or mcp_server_url),
        mcp_web_server_command=os.getenv("EXAMPLE11_MCP_WEB_SERVER_COMMAND"),
        mcp_web_server_args=tuple(shlex.split(mcp_web_args_raw)) if mcp_web_args_raw else (),
        mcp_web_tool_name=(os.getenv("EXAMPLE11_MCP_WEB_TOOL_NAME", "web.search").strip() or "web.search"),
    )
