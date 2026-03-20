from __future__ import annotations

try:
    from ..schemas import ResearchBrief, RunRecord, TacticalView
except ImportError:  # pragma: no cover
    from schemas import ResearchBrief, RunRecord, TacticalView


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
        history: list[RunRecord] | None = None,
    ) -> TacticalView:
        universe = self._normalize_universe(brief.tickers)
        intent = self._classify_intent(brief=brief, query=query, universe=universe)
        web_score = float(web_view.get("sentiment_score", 0.0))
        market_score = float(market_view.get("trend_score", 0.0))
        macro_score = float(macro_view.get("macro_score", 0.0))
        intent_score = 0.15 * self._intent_bias(query)

        web_sentiment = str(web_view.get("sentiment", "mixed")).lower()
        if web_sentiment in {"mixed", "neutral"}:
            web_score *= 0.45
        elif web_sentiment == "unavailable":
            web_score = 0.0

        market_trend = str(market_view.get("trend", "")).lower()
        if market_trend == "downtrend":
            market_score = min(market_score, -0.18)
        elif market_trend == "uptrend":
            market_score = max(market_score, 0.12)

        macro_regime = str(macro_view.get("regime", macro_view.get("summary", ""))).lower()
        if "mixed" in macro_regime or "mid-cycle" in macro_regime or "mid cycle" in macro_regime:
            macro_score *= 0.5

        score = web_score + market_score + macro_score + intent_score

        if history:
            last_signal = history[0].stance_signal.upper()
            if last_signal == "LONG" and score > 0.25:
                score += 0.02
            if last_signal == "SHORT" and score < -0.25:
                score -= 0.02

        evidence_mode = str(web_view.get("evidence_mode", "live"))
        signal = "NEUTRAL"
        # Mixed cross-signals should resolve conservatively; directional stances require clearer alignment and stronger net evidence.
        if score >= 0.28:
            signal = "LONG"
        elif score <= -0.28:
            signal = "SHORT"

        component_directions = [
            self._direction_bucket(web_score),
            self._direction_bucket(market_score),
            self._direction_bucket(macro_score),
        ]
        non_zero_directions = [direction for direction in component_directions if direction != 0]
        aligned_components = bool(non_zero_directions) and len(set(non_zero_directions)) == 1

        conviction = 0.2 + (0.4 * min(abs(score), 1.0))
        if aligned_components and len(non_zero_directions) >= 2:
            conviction += 0.08
        elif len(set(non_zero_directions)) > 1:
            conviction -= 0.12
        if signal == "NEUTRAL":
            conviction = min(conviction, 0.4)
        conviction = min(0.85, max(0.2, conviction))
        market_notes = market_view.get("notes", [])
        macro_notes = macro_view.get("notes", [])
        synthetic_market = market_view.get("trend") == "unknown" or self._notes_contain_any(
            market_notes,
            ("fallback", "placeholder"),
        )
        synthetic_macro = self._notes_contain_any(
            macro_notes,
            (
                "source: mcp-fallback",
                "source: mcp-partial",
                "missing indicators filled with fallback",
                "macro completeness: 0/",
            ),
        )
        market_data_weak = synthetic_market or bool(market_notes)
        if market_data_weak:
            conviction = min(conviction, 0.35)
        degraded_evidence = evidence_mode in {"fallback_only", "none"}
        if degraded_evidence:
            signal = "NEUTRAL"
            conviction = 0.2
        elif synthetic_market or synthetic_macro:
            conviction = min(conviction, 0.3)
            strong_live_web_alignment = (
                evidence_mode == "live"
                and float(web_view.get("confidence", 0.0)) >= 0.65
                and (
                    (signal == "LONG" and web_sentiment == "positive")
                    or (signal == "SHORT" and web_sentiment == "negative")
                )
            )
            if not strong_live_web_alignment:
                signal = "NEUTRAL"
                conviction = min(conviction, 0.25)

        if intent == "defensive" and signal == "SHORT":
            signal = "LONG"
            conviction = min(max(conviction, 0.25), 0.55)

        preferred_exposures: list[str] = []
        avoid_exposures: list[str] = []
        if degraded_evidence:
            preferred_exposures = []
            avoid_exposures = []
        elif not universe:
            signal = "VIEW_ONLY"
            conviction = 0.2
            preferred_exposures = []
            avoid_exposures = []
        else:
            preferred_exposures, avoid_exposures = self._build_exposure_guidance(
                signal=signal,
                tickers=universe,
                query=query,
                brief=brief,
            )
            if signal == "SHORT" and not avoid_exposures:
                signal = "NO_ACTION"
                conviction = min(conviction, 0.25)
                preferred_exposures = []
                avoid_exposures = []

        stance_basis: list[str] = []
        stance_basis.extend(web_view.get("key_points", [])[:2])
        stance_basis.append(market_view.get("summary", "Market signal is mixed."))
        stance_basis.append(macro_view.get("summary", "Macro signal is mixed."))
        if degraded_evidence:
            stance_basis.insert(0, "Live web evidence was unavailable; output is research-only and non-actionable.")
        elif signal == "VIEW_ONLY":
            stance_basis.insert(0, "No clean tradable universe was identified; maintain a research-only view.")
        elif signal == "NO_ACTION":
            stance_basis.insert(0, "No directional handoff was robust enough to justify a clear tactical preference.")

        risks: list[str] = []
        risks.extend(web_view.get("risks", [])[:2])
        high_volatility = market_view.get("high_volatility", [])
        if high_volatility:
            joined = ", ".join(high_volatility[:3])
            risks.append(f"High realized volatility detected in: {joined}.")
        risks.extend(macro_view.get("risks", [])[:2])
        if evidence_mode in {"fallback_only", "none"}:
            risks.insert(0, "Fallback-only web evidence detected; actionable portfolio deployment was withheld.")
        if signal == "VIEW_ONLY":
            risks.insert(0, "No tradable universe was identified; output is research-only and non-actionable.")
        elif signal == "NO_ACTION":
            risks.insert(0, "No coherent tactical preference could be expressed from the available evidence.")
        if not risks:
            risks.append("Model confidence is modest because signals are lightweight.")

        return TacticalView(
            signal=signal,
            conviction=round(conviction, 2),
            horizon=brief.timeframe,
            preferred_exposures=preferred_exposures,
            avoid_exposures=avoid_exposures,
            stance_basis=stance_basis,
            risks=risks,
        )

    def _build_exposure_guidance(
        self,
        signal: str,
        tickers: list[str],
        query: str,
        brief: ResearchBrief,
    ) -> tuple[list[str], list[str]]:
        universe = self._normalize_universe(tickers)
        if not universe or signal in {"VIEW_ONLY", "NO_ACTION"}:
            return [], []

        intent = self._classify_intent(brief=brief, query=query, universe=universe)
        equities, defensives, bonds, tech = self._split_universe(universe)
        preferred, secondary = self._intent_buckets(intent, equities, defensives, bonds, tech)
        preferred = preferred or universe

        if signal == "LONG":
            avoid = [ticker for ticker in universe if ticker not in preferred and ticker not in secondary]
            return preferred[:4], avoid[:3]

        if signal == "SHORT":
            if intent == "bond":
                preferred_defensive = [ticker for ticker in (defensives or bonds) if ticker not in self.LONG_DURATION_BONDS]
                avoid = [ticker for ticker in universe if ticker in self.LONG_DURATION_BONDS] or universe[:1]
                return (preferred_defensive or bonds or universe)[:4], avoid[:3]
            if intent == "defensive":
                return (defensives or bonds or universe)[:4], []
            preferred_defensive = (secondary or defensives or bonds or universe)[:4]
            avoid = (preferred or equities or universe)[:3]
            return preferred_defensive, avoid

        if intent in {"bond", "defensive"}:
            return (preferred or defensives or bonds or universe)[:4], []

        balanced = []
        for ticker in (secondary + preferred) if secondary else preferred:
            if ticker not in balanced:
                balanced.append(ticker)
        return balanced[:4], []

    def _normalize_universe(self, tickers: list[str]) -> list[str]:
        universe: list[str] = []
        for ticker in tickers:
            normalized = ticker.upper()
            if normalized not in universe:
                universe.append(normalized)
        return universe

    def _direction_bucket(self, value: float) -> int:
        if value >= 0.08:
            return 1
        if value <= -0.08:
            return -1
        return 0

    def _build_allocations(
        self,
        signal: str,
        tickers: list[str],
        conviction: float,
        query: str,
        brief: ResearchBrief,
    ) -> dict[str, float]:
        universe = self._normalize_universe(tickers)

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
            self._merge_allocations(allocations, self._allocate_bucket(secondary, secondary_total))
            cash = round(max(0.0, 1.0 - sum(allocations.values())), 4)
            if cash > 0:
                allocations["CASH"] = cash
            return self._rebalance_sum_to_one(allocations)

        if signal == "SHORT":
            short_total = self._clamp(0.35 + (0.45 * conviction) - intent_bias, 0.30, 0.85)
            short_targets = preferred or (equities if equities else universe)
            hedge_targets = [ticker for ticker in (secondary or defensives) if ticker not in short_targets]

            allocations = self._allocate_bucket(short_targets, -short_total)
            hedge_total = round(1.0 + short_total, 4)
            if hedge_targets:
                self._merge_allocations(allocations, self._allocate_bucket(hedge_targets, hedge_total))
            else:
                allocations["CASH"] = hedge_total
            return self._rebalance_sum_to_one(allocations)

        # Live NEUTRAL means cautious but still invested; degraded/fallback neutralization remains the truly defensive cash-heavy case.
        cash_weight = round(self._clamp(0.605 - (0.30 * conviction), 0.35, 0.55), 4)
        active_total = round(1.0 - cash_weight, 4)

        allocations = {"CASH": cash_weight}
        preferred_share = self._clamp(0.58 + 0.18 * conviction, 0.58, 0.75)
        preferred_total = round(active_total * preferred_share, 4)
        secondary_total = round(active_total - preferred_total, 4)
        self._merge_allocations(allocations, self._allocate_bucket(preferred, preferred_total))
        self._merge_allocations(allocations, self._allocate_bucket(secondary, secondary_total))
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
                self._merge_allocations(allocations, self._allocate_bucket(equity_satellite, satellite_total))
            cash = round(1.0 - sum(allocations.values()), 4)
            allocations["CASH"] = max(0.0, cash)
            return self._rebalance_sum_to_one(allocations)

        if signal == "SHORT":
            if conviction < 0.75:
                cash_weight = self._clamp(0.80 + 0.10 * (1.0 - conviction), 0.80, 0.92)
                short_targets = long_duration or bond_core[:1]
                defensive_targets = [ticker for ticker in (short_duration or bond_core) if ticker not in short_targets]
                short_total = self._clamp(0.08 + 0.10 * conviction, 0.08, 0.18)
                hedge_total = round(1.0 - cash_weight + short_total, 4)
                allocations = {"CASH": round(cash_weight, 4)}
                self._merge_allocations(allocations, self._allocate_bucket(short_targets, -short_total))
                if defensive_targets:
                    self._merge_allocations(allocations, self._allocate_bucket(defensive_targets, hedge_total))
                else:
                    allocations["CASH"] = round(allocations["CASH"] + hedge_total, 4)
                return self._rebalance_sum_to_one(allocations)

            short_targets = long_duration or bond_core[:1]
            hedge_targets = [
                ticker for ticker in (short_duration or [ticker for ticker in bond_core if ticker not in short_targets])
                if ticker not in short_targets
            ]
            short_total = self._clamp(0.22 + 0.20 * conviction, 0.22, 0.40)
            hedge_total = round(1.0 + short_total, 4)

            allocations = self._allocate_bucket(short_targets, -short_total)
            if hedge_targets:
                self._merge_allocations(allocations, self._allocate_bucket(hedge_targets, hedge_total))
            else:
                allocations["CASH"] = hedge_total
            return self._rebalance_sum_to_one(allocations)

        cash_weight = self._clamp(0.74 + 0.16 * (1.0 - conviction), 0.74, 0.90)
        active_total = round(1.0 - cash_weight, 4)
        core_total = round(active_total * 0.90, 4)
        satellite_total = round(active_total - core_total, 4)

        allocations = {"CASH": round(cash_weight, 4)}
        self._merge_allocations(allocations, self._allocate_bucket(bond_core, core_total))
        if equity_satellite and satellite_total > 0:
            self._merge_allocations(allocations, self._allocate_bucket(equity_satellite, satellite_total))
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

    def _merge_allocations(self, base: dict[str, float], addition: dict[str, float]) -> None:
        for ticker, weight in addition.items():
            base[ticker] = round(base.get(ticker, 0.0) + weight, 4)

    def _notes_contain_any(self, notes: list[str], terms: tuple[str, ...]) -> bool:
        for note in notes:
            normalized = str(note).lower()
            if any(term in normalized for term in terms):
                return True
        return False

    def _is_cash_only(self, allocations: dict[str, float]) -> bool:
        if not allocations:
            return True

        non_cash = [ticker for ticker, weight in allocations.items() if ticker != "CASH" and abs(weight) > 0.0001]
        cash_weight = float(allocations.get("CASH", 0.0))
        return not non_cash and cash_weight >= 0.9999

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
