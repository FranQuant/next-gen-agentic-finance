from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
    from ..schemas import MacroState
except ImportError:  # pragma: no cover
    from schemas import MacroState


class _MCPClient:
    def __init__(self, server_url: str | None, timeout_sec: int, use_live: bool) -> None:
        self.server_url = server_url
        self.timeout_sec = timeout_sec
        self.use_live = use_live

    def call_tool(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        # MCP transport boundary: all outbound capability calls should pass through this method.
        if not self.use_live or not self.server_url:
            raise RuntimeError("MCP macro transport not configured.")

        # Placeholder transport for Example11 v1.
        # Replace with a concrete MCP client implementation when server wiring is available.
        raise RuntimeError(f"Live MCP macro call unavailable for tool: {tool_name}")


class MCPMacroAdapter:
    FALLBACK_VALUES = {
        "inflation": 2.8,
        "unemployment": 4.1,
        "policy_rate": 3.6,
        "10y_yield": 4.1,
        "gdp_growth": 2.0,
    }

    def __init__(
        self,
        server_url: str | None = None,
        timeout_sec: int = 8,
        use_live: bool = False,
    ) -> None:
        self.client = _MCPClient(server_url=server_url, timeout_sec=timeout_sec, use_live=use_live)

    def get_macro_state(self, indicators: list[str]) -> MacroState:
        if not indicators:
            indicators = ["inflation", "unemployment", "policy_rate"]

        try:
            response = self.client.call_tool(
                "macro.get_state",
                {
                    "indicators": indicators,
                    "normalize": True,
                },
            )
            state = self._parse_state(response, indicators)
            if state:
                return state
        except Exception:
            pass

        return self._fallback_state(indicators)

    def _parse_state(self, response: dict[str, Any], indicators: list[str]) -> MacroState | None:
        if not isinstance(response, dict):
            return None

        raw_values = response.get("indicators")
        if not isinstance(raw_values, dict):
            return None

        values: dict[str, float] = {}
        for indicator in indicators:
            key = indicator.lower()
            value = raw_values.get(key)
            if isinstance(value, (int, float)):
                values[key] = round(float(value), 3)
            else:
                fallback = self.FALLBACK_VALUES.get(key, 0.0)
                values[key] = fallback

        regime = str(response.get("regime") or self._classify_regime(values))
        notes = response.get("notes") if isinstance(response.get("notes"), list) else []

        return MacroState(
            as_of=self._now_iso(),
            indicators=values,
            regime=regime,
            notes=[str(item) for item in notes],
        )

    def _fallback_state(self, indicators: list[str]) -> MacroState:
        values = {indicator.lower(): self.FALLBACK_VALUES.get(indicator.lower(), 0.0) for indicator in indicators}
        notes = [
            "MCP macro adapter fallback values used.",
            "No live MCP macro server configured for Example11 v1.",
            "Inflation assumed to be CPI YoY percent.",
        ]
        return MacroState(
            as_of=self._now_iso(),
            indicators=values,
            regime=self._classify_regime(values),
            notes=notes,
        )

    def _classify_regime(self, values: dict[str, float]) -> str:
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
