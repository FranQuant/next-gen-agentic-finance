from __future__ import annotations

try:
    from ..schemas import PortfolioView, ResearchBrief, RunRecord, SentimentSnapshot
except ImportError:  # pragma: no cover
    from schemas import PortfolioView, ResearchBrief, RunRecord, SentimentSnapshot


class PortfolioSynthesisAgent:
    DEFENSIVE_TICKERS = {
        "TLT",
        "IEF",
        "SHY",
        "BIL",
        "AGG",
        "LQD",
        "GLD",
        "UUP",
        "TIP",
    }
    BOND_TICKERS = {"TLT", "IEF", "SHY", "TIP", "AGG", "LQD", "BND", "BIL"}
    TECH_TICKERS = {"QQQ", "XLK", "SMH", "SOXX", "IGV"}
    LONG_DURATION_BONDS = {"TLT", "IEF", "EDV", "TLH"}
    SHORT_DURATION_DEFENSIVES = {"SHY", "BIL", "TIP"}

    def synthesize(
        self,
        query: str,
        brief: ResearchBrief,
        web_view: dict,
        market_view: dict,
        macro_view: dict,
        sentiment: SentimentSnapshot | None = None,
        history: list[RunRecord] | None = None,
    ) -> PortfolioView:
        score = float(web_view.get("sentiment_score", 0.0))
        score += float(market_view.get("trend_score", 0.0))
        score += float(macro_view.get("macro_score", 0.0))
        score += 0.25 * self._intent_bias(query)

        if sentiment and sentiment.bullish_prob is not None and sentiment.bearish_prob is not None:
            score += sentiment.bullish_prob - sentiment.bearish_prob

        if history:
            last_signal = history[0].portfolio_signal.upper()
            if last_signal == "LONG" and score > 0:
                score += 0.03
            if last_signal == "SHORT" and score < 0:
                score -= 0.03

        signal = "NEUTRAL"
        if score >= 0.15:
            signal = "LONG"
        elif score <= -0.15:
            signal = "SHORT"

        conviction = min(0.95, max(0.2, abs(score)))
        allocations = self._build_allocations(
            signal=signal,
            tickers=brief.tickers,
            conviction=conviction,
            query=query,
            brief=brief,
        )

        thesis: list[str] = []
        thesis.extend(web_view.get("key_points", [])[:2])
        thesis.append(market_view.get("summary", "Market signal is mixed."))
        thesis.append(macro_view.get("summary", "Macro signal is mixed."))

        risks: list[str] = []
        risks.extend(web_view.get("risks", [])[:2])
        risks.extend(macro_view.get("risks", [])[:2])
        if sentiment and sentiment.notes:
            risks.extend(sentiment.notes[:1])
        if not risks:
            risks.append("Model confidence is modest because signals are lightweight.")

        return PortfolioView(
            signal=signal,
            conviction=round(conviction, 2),
            horizon=brief.timeframe,
            allocations=allocations,
            thesis=thesis,
            risks=risks,
        )

    def _build_allocations(
        self,
        signal: str,
        tickers: list[str],
        conviction: float,
        query: str,
        brief: ResearchBrief,
    ) -> dict[str, float]:
        universe: list[str] = []
        for ticker in tickers:
            normalized = ticker.upper()
            if normalized not in universe:
                universe.append(normalized)

        if not universe:
            return {"CASH": 1.0}

        intent = self._classify_intent(brief=brief, query=query, universe=universe)
        equities, defensives, bonds, tech = self._split_universe(universe)
        preferred, secondary = self._intent_buckets(intent, equities, defensives, bonds, tech)
        intent_bias = self._intent_bias(query)

        if intent == "bond":
            return self._build_bond_intent_allocations(
                signal=signal,
                conviction=conviction,
                intent_bias=intent_bias,
                bonds=bonds,
                defensives=defensives,
                equities=equities,
            )

        if signal == "LONG":
            long_total = self._clamp(0.58 + (0.32 * conviction) + intent_bias, 0.55, 0.95)
            preferred_share = 1.0 if not secondary else self._clamp(0.70 + 0.15 * conviction, 0.70, 0.88)
            preferred_total = round(long_total * preferred_share, 4)
            secondary_total = round(long_total - preferred_total, 4)

            allocations = self._allocate_bucket(preferred, preferred_total)
            allocations.update(self._allocate_bucket(secondary, secondary_total))
            cash = round(max(0.0, 1.0 - sum(allocations.values())), 4)
            if cash > 0:
                allocations["CASH"] = cash
            return self._rebalance_sum_to_one(allocations)

        if signal == "SHORT":
            short_total = self._clamp(0.35 + (0.45 * conviction) - intent_bias, 0.30, 0.85)
            short_targets = preferred or (equities if equities else universe)
            hedge_targets = secondary or defensives

            allocations = self._allocate_bucket(short_targets, -short_total)
            hedge_total = round(1.0 - short_total, 4)
            if hedge_targets:
                allocations.update(self._allocate_bucket(hedge_targets, hedge_total))
            else:
                allocations["CASH"] = hedge_total
            return self._rebalance_sum_to_one(allocations)

        cash_weight = round(self._clamp(0.70 + (0.20 * (1.0 - conviction)), 0.65, 0.90), 4)
        active_total = round(1.0 - cash_weight, 4)

        allocations = {"CASH": cash_weight}
        preferred_share = self._clamp(0.70 + 0.10 * conviction, 0.65, 0.80)
        preferred_total = round(active_total * preferred_share, 4)
        secondary_total = round(active_total - preferred_total, 4)
        allocations.update(self._allocate_bucket(preferred, preferred_total))
        allocations.update(self._allocate_bucket(secondary, secondary_total))
        return self._rebalance_sum_to_one(allocations)

    def _build_bond_intent_allocations(
        self,
        signal: str,
        conviction: float,
        intent_bias: float,
        bonds: list[str],
        defensives: list[str],
        equities: list[str],
    ) -> dict[str, float]:
        bond_core = bonds or defensives
        if not bond_core:
            # If no explicit bond basket exists, remain conservative in cash.
            return {"CASH": 1.0}

        equity_satellite = [ticker for ticker in equities if ticker not in bond_core]
        short_duration = [ticker for ticker in bond_core if ticker in self.SHORT_DURATION_DEFENSIVES]
        long_duration = [ticker for ticker in bond_core if ticker in self.LONG_DURATION_BONDS]

        if signal == "LONG":
            active_total = self._clamp(0.58 + (0.30 * conviction) + max(0.0, intent_bias), 0.60, 0.95)
            core_share = self._clamp(0.86 + 0.10 * conviction, 0.86, 0.95)
            core_total = round(active_total * core_share, 4)
            satellite_total = round(active_total - core_total, 4)

            allocations = self._allocate_bucket(bond_core, core_total)
            if equity_satellite and satellite_total > 0:
                allocations.update(self._allocate_bucket(equity_satellite, satellite_total))
            cash = round(1.0 - sum(allocations.values()), 4)
            allocations["CASH"] = max(0.0, cash)
            return self._rebalance_sum_to_one(allocations)

        if signal == "SHORT":
            # Keep bond intent clean: avoid mixed long/short unless conviction is very high.
            if conviction < 0.75:
                cash_weight = self._clamp(0.82 + 0.12 * conviction, 0.82, 0.95)
                active_total = round(1.0 - cash_weight, 4)
                defensive_targets = short_duration or bond_core
                allocations = {"CASH": round(cash_weight, 4)}
                allocations.update(self._allocate_bucket(defensive_targets, active_total))
                return self._rebalance_sum_to_one(allocations)

            short_targets = long_duration or bond_core[:1]
            hedge_targets = short_duration or [ticker for ticker in bond_core if ticker not in short_targets]
            short_total = self._clamp(0.22 + 0.20 * conviction, 0.22, 0.40)
            hedge_total = round(1.0 + short_total, 4)

            allocations = self._allocate_bucket(short_targets, -short_total)
            if hedge_targets:
                allocations.update(self._allocate_bucket(hedge_targets, hedge_total))
            else:
                allocations["CASH"] = hedge_total
            return self._rebalance_sum_to_one(allocations)

        cash_weight = self._clamp(0.74 + 0.16 * (1.0 - conviction), 0.74, 0.90)
        active_total = round(1.0 - cash_weight, 4)
        core_total = round(active_total * 0.90, 4)
        satellite_total = round(active_total - core_total, 4)

        allocations = {"CASH": round(cash_weight, 4)}
        allocations.update(self._allocate_bucket(bond_core, core_total))
        if equity_satellite and satellite_total > 0:
            allocations.update(self._allocate_bucket(equity_satellite, satellite_total))
        return self._rebalance_sum_to_one(allocations)

    def _split_universe(self, tickers: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
        equities: list[str] = []
        defensives: list[str] = []
        bonds: list[str] = []
        tech: list[str] = []
        for ticker in tickers:
            if ticker in self.DEFENSIVE_TICKERS:
                defensives.append(ticker)
            else:
                equities.append(ticker)
            if ticker in self.BOND_TICKERS:
                bonds.append(ticker)
            if ticker in self.TECH_TICKERS:
                tech.append(ticker)
        return equities, defensives, bonds, tech

    def _intent_buckets(
        self,
        intent: str,
        equities: list[str],
        defensives: list[str],
        bonds: list[str],
        tech: list[str],
    ) -> tuple[list[str], list[str]]:
        if intent == "bond":
            preferred = bonds or defensives
            secondary = [ticker for ticker in equities if ticker not in preferred]
            return preferred, secondary
        if intent == "defensive":
            preferred = defensives or bonds
            secondary = [ticker for ticker in equities if ticker not in preferred]
            return preferred, secondary
        if intent == "tech":
            preferred = tech or equities
            secondary = [ticker for ticker in defensives if ticker not in preferred]
            return preferred, secondary
        preferred = [ticker for ticker in equities if ticker not in bonds] or equities
        secondary = [ticker for ticker in defensives if ticker not in preferred]
        return preferred, secondary

    def _classify_intent(self, brief: ResearchBrief, query: str, universe: list[str]) -> str:
        intent_hint = " ".join(brief.constraints).lower()
        for intent in ("bond", "defensive", "tech", "equity"):
            if f"intent lens: {intent}" in intent_hint:
                return intent

        normalized = query.lower()
        if any(term in normalized for term in ("bond", "duration", "treasury", "fixed income")):
            return "bond"
        if any(term in normalized for term in ("risk-off", "defensive", "safe haven", "de-risk")):
            return "defensive"
        if any(term in normalized for term in ("tech", "technology", "semiconductor", "nasdaq")):
            return "tech"

        bond_count = sum(ticker in self.BOND_TICKERS for ticker in universe)
        tech_count = sum(ticker in self.TECH_TICKERS for ticker in universe)
        defensive_count = sum(ticker in self.DEFENSIVE_TICKERS for ticker in universe)
        if tech_count >= max(1, len(universe) // 2):
            return "tech"
        if bond_count >= max(1, len(universe) // 2):
            return "bond"
        if defensive_count >= max(1, len(universe) // 2):
            return "defensive"
        return "equity"

    def _intent_bias(self, query: str) -> float:
        normalized = query.lower()
        if "overweight" in normalized and any(term in normalized for term in ("equity", "equities", "stocks")):
            return 0.12
        if "overweight" in normalized and any(term in normalized for term in ("bond", "duration", "treasury")):
            return 0.08

        defensive_terms = (
            "underweight equities",
            "risk-off",
            "defensive stance",
            "de-risk",
            "capital preservation",
        )

        if any(term in normalized for term in defensive_terms):
            return -0.12
        return 0.0

    def _allocate_bucket(self, tickers: list[str], total_weight: float) -> dict[str, float]:
        if not tickers:
            return {}
        weight = round(total_weight / len(tickers), 4)
        return {ticker: weight for ticker in tickers}

    def _rebalance_sum_to_one(self, allocations: dict[str, float]) -> dict[str, float]:
        total = round(sum(allocations.values()), 4)
        if abs(total - 1.0) <= 0.0001:
            return allocations

        adjustment = round(1.0 - total, 4)
        if "CASH" in allocations:
            allocations["CASH"] = round(allocations["CASH"] + adjustment, 4)
            return allocations

        first_key = next(iter(allocations), None)
        if first_key is not None:
            allocations[first_key] = round(allocations[first_key] + adjustment, 4)
        return allocations

    def _clamp(self, value: float, low: float, high: float) -> float:
        return max(low, min(high, value))
