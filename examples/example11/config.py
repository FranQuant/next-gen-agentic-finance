from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from urllib.parse import quote_plus


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
    mcp_web_tool_name: str = "tavily-search"
    mcp_web_extract_tool_name: str = "tavily-extract"
    mcp_web_enable_extract_enrichment: bool = True


def _parse_bool_env(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def _infer_tavily_mcp_url() -> str | None:
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return None
    return f"https://mcp.tavily.com/mcp/?tavilyApiKey={quote_plus(api_key)}"


def load_config() -> Example11Config:
    use_mcp_live_raw = os.getenv("EXAMPLE11_USE_MCP_LIVE")
    mcp_server_url = os.getenv("EXAMPLE11_MCP_SERVER_URL")
    inferred_tavily_url = _infer_tavily_mcp_url()
    explicit_web_server_url = os.getenv("EXAMPLE11_MCP_WEB_SERVER_URL")
    explicit_web_server_command = os.getenv("EXAMPLE11_MCP_WEB_SERVER_COMMAND")
    resolved_web_server_url = explicit_web_server_url or mcp_server_url or inferred_tavily_url
    default_web_transport = "streamable_http" if resolved_web_server_url else "stdio"
    use_mcp_live = _parse_bool_env(
        use_mcp_live_raw,
        default=bool(resolved_web_server_url or explicit_web_server_command),
    )
    mcp_web_args_raw = os.getenv("EXAMPLE11_MCP_WEB_SERVER_ARGS", "")
    enable_extract = _parse_bool_env(os.getenv("EXAMPLE11_MCP_WEB_ENABLE_EXTRACT_ENRICHMENT"), default=True)

    return Example11Config(
        db_path=os.getenv("EXAMPLE11_DB_PATH", "tmp/example11_runs.db"),
        history_limit=int(os.getenv("EXAMPLE11_HISTORY_LIMIT", "5")),
        top_topics=int(os.getenv("EXAMPLE11_TOP_TOPICS", "4")),
        mcp_server_url=mcp_server_url,
        mcp_timeout_sec=int(os.getenv("EXAMPLE11_MCP_TIMEOUT_SEC", "8")),
        use_mcp_live=use_mcp_live,
        mcp_web_transport=(os.getenv("EXAMPLE11_MCP_WEB_TRANSPORT") or default_web_transport).strip().lower(),
        mcp_web_server_url=resolved_web_server_url,
        mcp_web_server_command=explicit_web_server_command,
        mcp_web_server_args=tuple(shlex.split(mcp_web_args_raw)) if mcp_web_args_raw else (),
        mcp_web_tool_name=(os.getenv("EXAMPLE11_MCP_WEB_TOOL_NAME", "tavily-search").strip() or "tavily-search"),
        mcp_web_extract_tool_name=(
            os.getenv("EXAMPLE11_MCP_WEB_EXTRACT_TOOL_NAME", "tavily-extract").strip() or "tavily-extract"
        ),
        mcp_web_enable_extract_enrichment=enable_extract,
    )
