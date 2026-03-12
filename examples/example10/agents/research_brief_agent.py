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
}


class ResearchBriefAgent:
    INTENT_UNIVERSE = {
        "bond": ["TLT", "IEF", "SHY", "TIP", "SPY"],
        "equity": ["SPY", "QQQ", "IWM", "TLT"],
        "defensive": ["TLT", "SHY", "GLD", "UUP"],
        "tech": ["QQQ", "XLK", "SMH", "TLT"],
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
        extracted_tickers = self._extract_tickers(query)
        intent = self._classify_intent(query, extracted_tickers)
        tickers = extracted_tickers or self.INTENT_UNIVERSE.get(intent, list(self.default_tickers))
        topics = self._extract_topics(query, tickers)
        objective = f"Build a multi-source research view for: {query.strip()}"
        constraints = [
            "Use only available evidence and adapter outputs.",
            "Keep assumptions explicit and minimal.",
            f"Intent lens: {intent}.",
        ]

        return ResearchBrief(
            query=query.strip(),
            objective=objective,
            topics=topics,
            tickers=tickers,
            macro_indicators=list(self.default_macro_indicators),
            constraints=constraints,
        )

    def _extract_tickers(self, query: str) -> list[str]:
        candidates = re.findall(r"\b[A-Za-z]{1,5}\b", query)
        tickers: list[str] = []
        for candidate in candidates:
            normalized = candidate.upper()
            if normalized in _STOPWORDS:
                continue
            if normalized not in tickers:
                tickers.append(normalized)
        return tickers

    def _extract_topics(self, query: str, tickers: list[str]) -> list[str]:
        cleaned = query.strip().rstrip("?.!")
        topics = [cleaned] if cleaned else []

        for ticker in tickers:
            topics.append(f"{ticker} outlook")

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

        scores["bond"] += sum(term in normalized for term in bond_terms)
        scores["equity"] += sum(term in normalized for term in equity_terms)
        scores["defensive"] += sum(term in normalized for term in defensive_terms)
        scores["tech"] += sum(term in normalized for term in tech_terms)

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
            return "equity"

        return max(scores, key=scores.get)
