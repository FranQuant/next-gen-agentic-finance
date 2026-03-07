from __future__ import annotations

from datetime import datetime, timezone

try:
    from ..schemas import MacroState
except ImportError:  # pragma: no cover
    from schemas import MacroState


class FREDAdapter:
    SERIES_MAP = {
        "inflation": "CPIAUCSL",
        "unemployment": "UNRATE",
        "policy_rate": "FEDFUNDS",
        "10y_yield": "DGS10",
        "gdp_growth": "A191RL1Q225SBEA",
    }

    FALLBACK_VALUES = {
        "inflation": 2.8,
        "unemployment": 4.1,
        "policy_rate": 4.5,
        "10y_yield": 4.2,
        "gdp_growth": 2.0,
    }

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self._fred = None

        if api_key:
            try:
                from fredapi import Fred

                self._fred = Fred(api_key=api_key)
            except Exception:
                self._fred = None

    def get_macro_state(self, indicators: list[str]) -> MacroState:
        if not indicators:
            indicators = ["inflation", "unemployment", "policy_rate"]

        values: dict[str, float] = {}
        notes: list[str] = []

        for indicator in indicators:
            normalized = indicator.lower()
            fetched = self._fetch_latest(normalized)
            if fetched is None:
                fallback = self.FALLBACK_VALUES.get(normalized, 0.0)
                values[normalized] = fallback
                notes.append(f"{normalized}: fallback baseline used.")
            else:
                values[normalized] = round(float(fetched), 3)

            if normalized == "inflation":
                notes.append("inflation: reported as CPI year-over-year percent.")

        regime = self._classify_regime(values)

        return MacroState(
            as_of=self._now_iso(),
            indicators=values,
            regime=regime,
            notes=notes,
        )

    def _fetch_latest(self, indicator: str) -> float | None:
        if not self._fred:
            return None

        if indicator == "inflation":
            return self._fetch_inflation_yoy()

        series = self.SERIES_MAP.get(indicator, indicator.upper())
        try:
            values = self._fred.get_series_latest_release(series)
            if hasattr(values, "dropna"):
                clean = values.dropna()
                if clean.empty:
                    return None
                return float(clean.iloc[-1])
            return float(values)
        except Exception:
            return None

    def _fetch_inflation_yoy(self) -> float | None:
        if not self._fred:
            return None

        try:
            cpi = self._fred.get_series(self.SERIES_MAP["inflation"]).dropna()
            if len(cpi) < 13:
                return None
            latest = float(cpi.iloc[-1])
            prior_year = float(cpi.iloc[-13])
            if prior_year <= 0:
                return None
            return (latest / prior_year - 1.0) * 100.0
        except Exception:
            return None

    def _classify_regime(self, values: dict[str, float]) -> str:
        # All thresholds are specified in percentage-rate terms.
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

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
