from __future__ import annotations

try:
    from ..schemas import MacroState, ResearchBrief
except ImportError:  # pragma: no cover
    from schemas import MacroState, ResearchBrief


class MacroRegimeAgent:
    def analyze(self, brief: ResearchBrief, macro_state: MacroState) -> dict:
        inflation = float(macro_state.indicators.get("inflation", 2.5))
        unemployment = float(macro_state.indicators.get("unemployment", 4.0))
        policy_rate = float(macro_state.indicators.get("policy_rate", 3.0))
        ten_year_yield = macro_state.indicators.get("10y_yield")

        macro_score = 0.0
        implications: list[str] = []
        risks: list[str] = []

        if inflation > 3.0:
            macro_score -= 0.2
            risks.append("Inflation remains above target; multiples can compress.")
        else:
            implications.append("Inflation is moderate, supporting stable risk premia.")

        if unemployment < 4.5:
            macro_score += 0.1
            implications.append("Labor market is resilient.")
        else:
            macro_score -= 0.2
            risks.append("Labor market softness raises recession probability.")

        if policy_rate > 4.0:
            macro_score -= 0.1
            risks.append("Restrictive policy can pressure cyclical assets.")
        else:
            implications.append("Policy stance is less restrictive.")

        if macro_state.regime.lower().startswith("recession"):
            macro_score -= 0.25
            risks.append("Regime classification is recessionary.")

        if not implications:
            implications.append("Macro backdrop is mixed with no dominant signal.")
        if not risks:
            risks.append("Macro surprises can still disrupt the base case.")

        indicator_snapshot = {
            "inflation_yoy_pct": round(inflation, 2),
            "unemployment_pct": round(unemployment, 2),
            "policy_rate_pct": round(policy_rate, 2),
        }
        if ten_year_yield is not None:
            indicator_snapshot["ten_year_yield_pct"] = round(float(ten_year_yield), 2)

        return {
            "summary": (
                "Macro regime: "
                f"{macro_state.regime} "
                f"(CPI YoY {inflation:.1f}%, "
                f"Unemployment {unemployment:.1f}%, "
                f"Policy {policy_rate:.1f}%)."
            ),
            "regime": macro_state.regime,
            "indicator_snapshot": indicator_snapshot,
            "implications": implications,
            "risks": risks,
            "macro_score": round(macro_score, 3),
            "notes": macro_state.notes,
        }
