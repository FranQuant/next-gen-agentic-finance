from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache

from mcp.server.fastmcp import FastMCP


SERIES_MAP = {
    "inflation": "CPIAUCSL",
    "unemployment": "UNRATE",
    "policy_rate": "FEDFUNDS",
    "10y_yield": "DGS10",
}
DEFAULT_INDICATORS = (
    "inflation",
    "unemployment",
    "policy_rate",
    "10y_yield",
)

server = FastMCP(name="example11-macro")


@lru_cache(maxsize=1)
def _fred_client():
    api_key = (os.getenv("FRED_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("FRED_API_KEY is required for the Example11 macro MCP server.")

    from fredapi import Fred

    return Fred(api_key=api_key)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify_regime(values: dict[str, float]) -> str:
    inflation = values.get("inflation", 2.8)
    unemployment = values.get("unemployment", 4.1)
    policy_rate = values.get("policy_rate", 4.5)

    if unemployment >= 5.5:
        return "recession risk"
    if inflation >= 3.0 and policy_rate >= 4.0:
        return "late-cycle tightening"
    if inflation <= 2.3 and policy_rate <= 3.5 and unemployment <= 4.8:
        return "easing expansion"
    return "mid-cycle mixed"


def _fetch_latest(indicator: str) -> float:
    if indicator == "inflation":
        return _fetch_inflation_yoy()

    series_id = SERIES_MAP.get(indicator)
    if not series_id:
        raise RuntimeError(f"Unsupported indicator: {indicator}")

    fred = _fred_client()
    values = fred.get_series_latest_release(series_id)
    if hasattr(values, "dropna"):
        clean = values.dropna()
        if clean.empty:
            raise RuntimeError(f"No observations returned for {indicator}")
        return float(clean.iloc[-1])
    return float(values)


def _fetch_inflation_yoy() -> float:
    fred = _fred_client()
    cpi = fred.get_series(SERIES_MAP["inflation"]).dropna()
    if len(cpi) < 13:
        raise RuntimeError("Not enough CPI history to compute year-over-year inflation.")

    latest = float(cpi.iloc[-1])
    prior_year = float(cpi.iloc[-13])
    if prior_year <= 0:
        raise RuntimeError("Invalid prior-year CPI value for inflation normalization.")

    return (latest / prior_year - 1.0) * 100.0


@server.tool(
    name="macro.get_state",
    description="Return the latest normalized macro state for inflation, unemployment, policy_rate, and 10y_yield.",
    structured_output=True,
)
def get_macro_state(indicators: list[str] | None = None, normalize: bool = True) -> dict[str, object]:
    requested = [str(indicator).lower() for indicator in (indicators or DEFAULT_INDICATORS)]

    values: dict[str, float] = {}
    notes = ["source: mcp-live"]
    for indicator in requested:
        values[indicator] = round(float(_fetch_latest(indicator)), 3)

    if normalize and "inflation" in values:
        notes.append("inflation: reported as CPI year-over-year percent.")

    return {
        "as_of": _now_iso(),
        "source": "mcp-live",
        "indicators": values,
        "regime": _classify_regime(values),
        "notes": notes,
    }


if __name__ == "__main__":
    server.run("stdio")
