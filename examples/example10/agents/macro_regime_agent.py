from __future__ import annotations

try:
    from ..schemas import MacroState, ResearchBrief
except ImportError:  # pragma: no cover
    from schemas import MacroState, ResearchBrief


class MacroRegimeAgent:
    def analyze(self, brief: ResearchBrief, macro_state: MacroState) -> dict:
        indicators = macro_state.indicators
        inflation = float(indicators.get("inflation", 2.8))
        core_inflation = float(indicators.get("core_inflation", inflation))
        unemployment = float(indicators.get("unemployment", 4.1))
        payrolls = float(indicators.get("payrolls", 1.5))
        policy_rate = float(indicators.get("policy_rate", 4.25))
        ten_year_yield = float(indicators.get("10y_yield", 4.1))
        curve_slope = float(indicators.get("curve_slope", -0.2))
        credit_spread = float(indicators.get("credit_spread", 3.7))

        inflation_score = 0.0
        labor_score = 0.0
        policy_score = 0.0
        credit_score = 0.0
        implications: list[str] = []
        risks: list[str] = []

        if inflation >= 3.0 or core_inflation >= 3.2:
            inflation_score -= 0.15
            risks.append("Inflation remains sticky enough to limit near-term easing flexibility.")
        elif inflation <= 2.5 and core_inflation <= 2.9:
            inflation_score += 0.1
            implications.append("Inflation is moderating toward a more market-friendly range.")
        else:
            implications.append("Inflation is easing only gradually, keeping the backdrop mixed.")

        if unemployment <= 4.3 and payrolls >= 1.0:
            labor_score += 0.12
            implications.append("Labor conditions remain resilient, reducing immediate recession pressure.")
        elif unemployment >= 5.0 or payrolls <= 0.5:
            labor_score -= 0.18
            risks.append("Labor conditions are softening enough to raise growth-risk concerns.")
        else:
            risks.append("Labor momentum is cooling from strong levels, leaving the growth signal less decisive.")

        if policy_rate >= 4.5 and curve_slope < 0.0:
            policy_score -= 0.14
            risks.append("Rates remain restrictive and the curve still signals policy tightness.")
        elif policy_rate <= 3.5 and curve_slope > 0.0:
            policy_score += 0.08
            implications.append("Policy and curve structure are less restrictive than in a tightening regime.")
        else:
            implications.append("Policy and rates are not clearly easing financial conditions yet.")

        if credit_spread >= 4.5:
            credit_score -= 0.18
            risks.append("Credit spreads are wide enough to point to tighter risk appetite.")
        elif credit_spread <= 3.5:
            credit_score += 0.08
            implications.append("Credit spreads remain contained, which supports broader risk transmission.")
        else:
            risks.append("Credit is not stressed, but spreads are wide enough to cap risk enthusiasm.")

        macro_score = inflation_score + labor_score + policy_score + credit_score
        regime = self._classify_regime(
            inflation=inflation,
            core_inflation=core_inflation,
            unemployment=unemployment,
            payrolls=payrolls,
            policy_rate=policy_rate,
            curve_slope=curve_slope,
            credit_spread=credit_spread,
        )

        if regime == "recession risk":
            macro_score -= 0.1
            risks.append("Cross-block regime signals remain consistent with recession-risk conditions.")
        elif regime == "late-cycle tightening":
            macro_score -= 0.05
            risks.append("The broader macro mix still resembles a late-cycle tightening backdrop.")
        elif regime == "disinflationary expansion":
            macro_score += 0.05
            implications.append("Cross-block conditions broadly fit a disinflationary expansion setup.")

        if not implications:
            implications.append("Macro backdrop is balanced with no single supportive block dominating.")
        if not risks:
            risks.append("Macro cross-currents remain relevant even without an acute stress signal.")

        indicator_snapshot = {
            "headline_inflation_yoy_pct": round(inflation, 2),
            "core_inflation_yoy_pct": round(core_inflation, 2),
            "unemployment_pct": round(unemployment, 2),
            "payrolls_yoy_pct": round(payrolls, 2),
            "policy_rate_pct": round(policy_rate, 2),
            "ten_year_yield_pct": round(ten_year_yield, 2),
            "curve_slope_pct": round(curve_slope, 2),
            "credit_spread_pct": round(credit_spread, 2),
        }

        return {
            "summary": (
                "Macro regime: "
                f"{regime} "
                f"(headline/core inflation {inflation:.1f}%/{core_inflation:.1f}%, "
                f"unemployment {unemployment:.1f}%, payrolls {payrolls:.1f}% YoY, "
                f"policy {policy_rate:.1f}%, curve {curve_slope:.2f}, HY spread {credit_spread:.1f}%)."
            ),
            "regime": regime,
            "indicator_snapshot": indicator_snapshot,
            "implications": implications[:4],
            "risks": risks[:4],
            "macro_score": round(macro_score, 3),
            "notes": macro_state.notes,
        }

    def _classify_regime(
        self,
        inflation: float,
        core_inflation: float,
        unemployment: float,
        payrolls: float,
        policy_rate: float,
        curve_slope: float,
        credit_spread: float,
    ) -> str:
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
