from __future__ import annotations

import re
from typing import Iterable

try:
    from ..schemas import ResearchBrief
except ImportError:  # pragma: no cover
    from schemas import ResearchBrief


_STOPWORDS = {
    "A",
    "AN",
    "AND",
    "OR",
    "THE",
    "FOR",
    "WITH",
    "TO",
    "OF",
    "IN",
    "ON",
    "AI",
    "ETF",
    "GDP",
    "CPI",
    "BOND",
    "BONDS",
    "LONG",
    "MACRO",
    "MONTH",
    "NEWS",
    "RATE",
    "RATES",
    "RISK",
    "RISKS",
    "SHORT",
    "STOCK",
    "VIEW",
    "YIELD",
    "FED",
    "FOMC",
    "US",
    "USA",
    "WEB",
    "GIVEN",
}


class ResearchBriefAgent:
    INTENT_UNIVERSE = {
        "bond": ["TLT", "IEF", "SHY", "TIP"],
        "equity": ["SPY", "QQQ", "IWM", "TLT"],
        "defensive": ["TLT", "SHY", "GLD", "UUP"],
        "tech": ["QQQ", "XLK", "SMH", "TLT"],
    }
    INTENT_TOPICS = {
        "bond": "bond market outlook",
        "equity": "equity market outlook",
        "defensive": "risk-off positioning",
        "tech": "technology sector outlook",
    }

    def __init__(
        self,
        default_tickers: Iterable[str],
        default_macro_indicators: Iterable[str],
        max_topics: int = 4,
    ) -> None:
        self.default_tickers = [ticker.upper() for ticker in default_tickers]
        self.default_macro_indicators = list(default_macro_indicators)
        self.max_topics = max_topics

    def generate(self, query: str) -> ResearchBrief:
        cleaned_query = query.strip()
        explicit_tickers = self._extract_explicit_tickers(cleaned_query)
        intent = self._classify_intent(cleaned_query, explicit_tickers)
        tickers = self._build_tickers(explicit_tickers, intent)
        topics = self._build_topics(cleaned_query, intent)
        objective = f"Build a multi-source research view for: {cleaned_query}"
        constraints = [
            "Use only available evidence and adapter outputs.",
            "Keep assumptions explicit and minimal.",
            f"Intent lens: {intent}.",
            "Capability transport: MCP-native adapters.",
        ]
        if explicit_tickers:
            constraints.append("Universe mode: explicit tickers only.")
        elif tickers:
            constraints.append("Universe mode: intent-default tradable basket.")
        else:
            constraints.append("Universe mode: view-first; no clean tradable universe identified.")

        return ResearchBrief(
            query=cleaned_query,
            objective=objective,
            topics=topics,
            tickers=tickers,
            macro_indicators=list(self.default_macro_indicators),
            constraints=constraints,
        )

    def _extract_explicit_tickers(self, query: str) -> list[str]:
        tickers: list[str] = []
        candidates = re.findall(r"\$([A-Za-z]{1,5})\b", query)
        candidates.extend(re.findall(r"\b([A-Z]{2,5})\b", query))

        for candidate in candidates:
            normalized = candidate.upper()
            if normalized in _STOPWORDS:
                continue
            if normalized not in tickers:
                tickers.append(normalized)
        return tickers

    def _build_tickers(self, explicit_tickers: list[str], intent: str) -> list[str]:
        if explicit_tickers:
            return explicit_tickers

        return list(self.INTENT_UNIVERSE.get(intent, []))

    def _build_topics(self, query: str, intent: str) -> list[str]:
        cleaned = query.strip().rstrip("?.!")
        topics = [cleaned] if cleaned else []

        if intent in self.INTENT_TOPICS:
            topics.append(self.INTENT_TOPICS[intent])

        normalized = cleaned.lower()
        if "inflation" in normalized:
            topics.append("inflation outlook")
        if "fed" in normalized or "federal reserve" in normalized or "policy rate" in normalized:
            topics.append("Federal Reserve policy outlook")
        if "uncertainty" in normalized:
            topics.append("macro uncertainty")
        if "yield" in normalized or "treasury" in normalized or "treasuries" in normalized:
            topics.append("Treasury yield outlook")
        if "growth" in normalized or "recession" in normalized:
            topics.append("growth and recession risk")

        topics.append("macro regime")

        deduped: list[str] = []
        for topic in topics:
            if topic and topic not in deduped:
                deduped.append(topic)

        return deduped[: self.max_topics]

    def _classify_intent(self, query: str, extracted_tickers: list[str]) -> str:
        normalized = query.lower()
        scores = {
            "bond": 0,
            "equity": 0,
            "defensive": 0,
            "tech": 0,
        }

        bond_terms = (
            "bond",
            "duration",
            "treasury",
            "fixed income",
            "yield",
            "fed uncertainty",
            "rate uncertainty",
        )
        equity_terms = (
            "equity",
            "equities",
            "stock",
            "stocks",
            "risk-on",
            "overweight to equities",
            "overweight equities",
        )
        defensive_terms = (
            "risk-off",
            "defensive",
            "de-risk",
            "safe haven",
            "capital preservation",
            "drawdown hedge",
        )
        tech_terms = (
            "tech",
            "technology",
            "semiconductor",
            "software",
            "ai",
            "cloud",
            "nasdaq",
        )

        scores["bond"] += sum(self._contains_term(normalized, term) for term in bond_terms)
        scores["equity"] += sum(self._contains_term(normalized, term) for term in equity_terms)
        scores["defensive"] += sum(self._contains_term(normalized, term) for term in defensive_terms)
        scores["tech"] += sum(self._contains_term(normalized, term) for term in tech_terms)

        for ticker in extracted_tickers:
            upper = ticker.upper()
            if upper in {"TLT", "IEF", "SHY", "TIP", "BND", "AGG"}:
                scores["bond"] += 1
            elif upper in {"GLD", "UUP", "BIL"}:
                scores["defensive"] += 1
            elif upper in {"QQQ", "XLK", "SMH", "SOXX", "IGV"}:
                scores["tech"] += 1
                scores["equity"] += 1
            else:
                scores["equity"] += 1

        if all(value == 0 for value in scores.values()):
            return "macro"

        return max(scores, key=scores.get)

    def _contains_term(self, text: str, term: str) -> bool:
        return bool(re.search(rf"\b{re.escape(term.lower())}\b", text))
