from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache

from mcp.server.fastmcp import FastMCP


INDICATOR_SPECS = {
    "inflation": {
        "series_id": "CPIAUCSL",
        "transform": "yoy",
        "note": "inflation: headline CPI year-over-year percent.",
    },
    "core_inflation": {
        "series_id": "CPILFESL",
        "transform": "yoy",
        "note": "core_inflation: core CPI year-over-year percent.",
    },
    "unemployment": {
        "series_id": "UNRATE",
        "transform": "latest",
        "note": "unemployment: civilian unemployment rate percent.",
    },
    "payrolls": {
        "series_id": "PAYEMS",
        "transform": "yoy",
        "note": "payrolls: nonfarm payrolls year-over-year percent.",
    },
    "policy_rate": {
        "series_id": "FEDFUNDS",
        "transform": "latest",
        "note": "policy_rate: effective fed funds rate percent.",
    },
    "10y_yield": {
        "series_id": "DGS10",
        "transform": "latest",
        "note": "10y_yield: 10-year Treasury yield percent.",
    },
    "curve_slope": {
        "series_id": "T10Y2Y",
        "transform": "latest",
        "note": "curve_slope: 10-year minus 2-year Treasury slope percentage points.",
    },
    "credit_spread": {
        "series_id": "BAMLH0A0HYM2",
        "transform": "latest",
        "note": "credit_spread: US high-yield option-adjusted spread percentage points.",
    },
}
INDICATOR_ALIASES = {
    "headline_inflation": "inflation",
    "cpi": "inflation",
    "cpiaucsl": "inflation",
    "core_cpi": "core_inflation",
    "cpilfesl": "core_inflation",
    "unrate": "unemployment",
    "payems": "payrolls",
    "fedfunds": "policy_rate",
    "dgs10": "10y_yield",
    "t10y2y": "curve_slope",
    "bamlh0a0hym2": "credit_spread",
}
DEFAULT_INDICATORS = tuple(INDICATOR_SPECS.keys())

server = FastMCP(name="example10-macro")


@lru_cache(maxsize=1)
def _fred_client():
    api_key = (os.getenv("FRED_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("FRED_API_KEY is required for the Example10 macro MCP server.")

    from fredapi import Fred

    return Fred(api_key=api_key)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_indicator(indicator: str) -> str:
    normalized = str(indicator).strip().lower()
    if normalized in INDICATOR_SPECS:
        return normalized
    return INDICATOR_ALIASES.get(normalized, normalized)


def _classify_regime(values: dict[str, float]) -> str:
    inflation = values.get("inflation", 2.8)
    core_inflation = values.get("core_inflation", inflation)
    unemployment = values.get("unemployment", 4.1)
    payrolls = values.get("payrolls", 1.5)
    policy_rate = values.get("policy_rate", 4.5)
    curve_slope = values.get("curve_slope", -0.2)
    credit_spread = values.get("credit_spread", 3.6)

    if (unemployment >= 5.0 and payrolls <= 0.5) or (curve_slope <= -0.5 and credit_spread >= 5.0):
        return "recession risk"
    if inflation >= 2.8 and core_inflation >= 3.0 and policy_rate >= 4.0:
        return "late-cycle tightening"
    if (
        inflation <= 2.5
        and core_inflation <= 2.9
        and unemployment <= 4.6
        and payrolls >= 1.0
        and curve_slope > 0.0
        and credit_spread < 4.0
    ):
        return "disinflationary expansion"
    return "mid-cycle mixed"


def _fetch_latest(indicator: str) -> float:
    normalized = _normalize_indicator(indicator)
    spec = INDICATOR_SPECS.get(normalized)
    if not spec:
        raise RuntimeError(f"Unsupported indicator: {indicator}")

    if spec["transform"] == "yoy":
        return _fetch_yoy_percent(str(spec["series_id"]))
    return _fetch_latest_release(str(spec["series_id"]), normalized)


def _fetch_latest_release(series_id: str, indicator: str) -> float:
    fred = _fred_client()
    values = fred.get_series_latest_release(series_id)
    if hasattr(values, "dropna"):
        clean = values.dropna()
        if clean.empty:
            raise RuntimeError(f"No observations returned for {indicator}")
        return float(clean.iloc[-1])
    return float(values)


def _fetch_yoy_percent(series_id: str) -> float:
    fred = _fred_client()
    series = fred.get_series(series_id).dropna()
    if len(series) < 13:
        raise RuntimeError(f"Not enough history to compute year-over-year change for {series_id}.")

    latest = float(series.iloc[-1])
    prior_year = float(series.iloc[-13])
    if prior_year <= 0:
        raise RuntimeError(f"Invalid prior-year value for {series_id} normalization.")

    return (latest / prior_year - 1.0) * 100.0


def _normalization_notes(indicators: list[str]) -> list[str]:
    notes: list[str] = []
    for indicator in indicators:
        spec = INDICATOR_SPECS.get(indicator)
        if spec:
            notes.append(str(spec["note"]))
    return notes


@server.tool(
    name="macro.get_state",
    description=(
        "Return the latest normalized Example10 US macro state across inflation, labor, "
        "policy/rates, curve, and credit indicators."
    ),
    structured_output=True,
)
def get_macro_state(indicators: list[str] | None = None, normalize: bool = True) -> dict[str, object]:
    requested = [_normalize_indicator(indicator) for indicator in (indicators or DEFAULT_INDICATORS)]
    deduped_requested: list[str] = []
    for indicator in requested:
        if indicator in INDICATOR_SPECS and indicator not in deduped_requested:
            deduped_requested.append(indicator)

    values: dict[str, float] = {}
    notes = ["source: mcp-live"]
    failed_indicators: list[str] = []
    for indicator in deduped_requested:
        try:
            values[indicator] = round(float(_fetch_latest(indicator)), 3)
        except Exception as exc:
            failed_indicators.append(indicator)
            notes.append(f"{indicator}: live fetch failed ({type(exc).__name__}).")

    if normalize:
        notes.extend(_normalization_notes(list(values.keys())))

    source = "mcp-live"
    if failed_indicators and values:
        source = "mcp-partial-live"
        notes[0] = f"source: {source}"
        notes.append(f"failed indicators omitted from live response: {', '.join(failed_indicators)}")
    elif failed_indicators and not values:
        source = "mcp-live-error"
        notes[0] = f"source: {source}"
        notes.append("all requested indicators failed live retrieval.")

    return {
        "as_of": _now_iso(),
        "source": source,
        "indicators": values,
        "failed_indicators": failed_indicators,
        "regime": _classify_regime(values),
        "notes": notes,
    }


if __name__ == "__main__":
    server.run("stdio")
