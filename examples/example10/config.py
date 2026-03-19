from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus


@dataclass(frozen=True)
class Example10Config:
    db_path: str = "tmp/example10_runs.db"
    history_limit: int = 5
    top_topics: int = 4
    default_tickers: tuple[str, ...] = ("SPY", "QQQ", "TLT")
    default_macro_indicators: tuple[str, ...] = (
        "inflation",
        "core_inflation",
        "unemployment",
        "payrolls",
        "policy_rate",
        "10y_yield",
        "curve_slope",
        "credit_spread",
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
    mcp_macro_transport: str = "stdio"
    mcp_macro_server_url: str | None = None
    mcp_macro_server_command: str | None = None
    mcp_macro_server_args: tuple[str, ...] = ()
    mcp_macro_tool_name: str = "macro.get_state"


def _parse_bool_env(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def _get_example_env(name: str, default: str | None = None) -> str | None:
    primary = os.getenv(f"EXAMPLE10_{name}")
    if primary is not None:
        return primary

    deprecated = os.getenv(f"EXAMPLE11_{name}")
    if deprecated is not None:
        return deprecated

    return default


def _infer_tavily_mcp_url() -> str | None:
    api_key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return None
    return f"https://mcp.tavily.com/mcp/?tavilyApiKey={quote_plus(api_key)}"


def _infer_macro_server_command() -> tuple[str | None, tuple[str, ...]]:
    if not (os.getenv("FRED_API_KEY") or "").strip():
        return None, ()

    server_path = Path(__file__).resolve().parent / "adapters" / "mcp_macro_server.py"
    if not server_path.exists():
        return None, ()

    return sys.executable, (str(server_path),)


def load_config() -> Example10Config:
    use_mcp_live_raw = _get_example_env("USE_MCP_LIVE")
    mcp_server_url = _get_example_env("MCP_SERVER_URL")
    inferred_tavily_url = _infer_tavily_mcp_url()
    inferred_macro_command, inferred_macro_args = _infer_macro_server_command()
    explicit_web_server_url = _get_example_env("MCP_WEB_SERVER_URL")
    explicit_web_server_command = _get_example_env("MCP_WEB_SERVER_COMMAND")
    resolved_web_server_url = explicit_web_server_url or mcp_server_url or inferred_tavily_url
    default_web_transport = "streamable_http" if resolved_web_server_url else "stdio"
    explicit_macro_server_url = _get_example_env("MCP_MACRO_SERVER_URL")
    explicit_macro_server_command = _get_example_env("MCP_MACRO_SERVER_COMMAND")
    resolved_macro_server_url = explicit_macro_server_url
    resolved_macro_server_command = explicit_macro_server_command or inferred_macro_command
    default_macro_transport = "stdio" if resolved_macro_server_command else "streamable_http" if resolved_macro_server_url else "stdio"
    use_mcp_live = _parse_bool_env(
        use_mcp_live_raw,
        default=bool(
            resolved_web_server_url
            or explicit_web_server_command
            or resolved_macro_server_url
            or resolved_macro_server_command
        ),
    )
    mcp_web_args_raw = _get_example_env("MCP_WEB_SERVER_ARGS", "") or ""
    mcp_macro_args_raw = _get_example_env("MCP_MACRO_SERVER_ARGS", "") or ""
    enable_extract = _parse_bool_env(_get_example_env("MCP_WEB_ENABLE_EXTRACT_ENRICHMENT"), default=True)

    return Example10Config(
        db_path=_get_example_env("DB_PATH", "tmp/example10_runs.db") or "tmp/example10_runs.db",
        history_limit=int(_get_example_env("HISTORY_LIMIT", "5") or "5"),
        top_topics=int(_get_example_env("TOP_TOPICS", "4") or "4"),
        mcp_server_url=mcp_server_url,
        mcp_timeout_sec=int(_get_example_env("MCP_TIMEOUT_SEC", "8") or "8"),
        use_mcp_live=use_mcp_live,
        mcp_web_transport=(_get_example_env("MCP_WEB_TRANSPORT") or default_web_transport).strip().lower(),
        mcp_web_server_url=resolved_web_server_url,
        mcp_web_server_command=explicit_web_server_command,
        mcp_web_server_args=tuple(shlex.split(mcp_web_args_raw)) if mcp_web_args_raw else (),
        mcp_web_tool_name=((_get_example_env("MCP_WEB_TOOL_NAME", "tavily-search") or "tavily-search").strip() or "tavily-search"),
        mcp_web_extract_tool_name=(
            (_get_example_env("MCP_WEB_EXTRACT_TOOL_NAME", "tavily-extract") or "tavily-extract").strip()
            or "tavily-extract"
        ),
        mcp_web_enable_extract_enrichment=enable_extract,
        mcp_macro_transport=(_get_example_env("MCP_MACRO_TRANSPORT") or default_macro_transport).strip().lower(),
        mcp_macro_server_url=resolved_macro_server_url,
        mcp_macro_server_command=resolved_macro_server_command,
        mcp_macro_server_args=tuple(shlex.split(mcp_macro_args_raw))
        if mcp_macro_args_raw
        else inferred_macro_args if not explicit_macro_server_command else (),
        mcp_macro_tool_name=((_get_example_env("MCP_MACRO_TOOL_NAME", "macro.get_state") or "macro.get_state").strip() or "macro.get_state"),
    )
