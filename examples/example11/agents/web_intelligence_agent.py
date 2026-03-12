from __future__ import annotations

try:
    from ..schemas import Evidence, ResearchBrief
except ImportError:  # pragma: no cover
    from schemas import Evidence, ResearchBrief


_POSITIVE_TOKENS = (
    "beat",
    "growth",
    "strong",
    "upgrade",
    "outperform",
    "expansion",
    "bull",
    "up",
)

_NEGATIVE_TOKENS = (
    "miss",
    "downgrade",
    "weak",
    "risk",
    "lawsuit",
    "recession",
    "bear",
    "down",
)


class WebIntelligenceAgent:
    def analyze(self, brief: ResearchBrief, evidence: list[Evidence]) -> dict:
        if not evidence:
            return {
                "summary": "No external evidence was retrieved.",
                "key_points": ["Proceed with low confidence due to sparse web evidence."],
                "opportunities": [],
                "risks": ["Limited external validation."],
                "evidence_count": 0,
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.2,
                "evidence_mode": "none",
                "fallback_count": 0,
                "live_evidence_count": 0,
            }

        live_evidence = [item for item in evidence if not self._is_fallback_evidence(item)]
        fallback_count = len(evidence) - len(live_evidence)

        if not live_evidence:
            return {
                "summary": "Web evidence unavailable; offline placeholder notes only.",
                "key_points": ["Offline fallback placeholders are not sourced research."],
                "opportunities": [],
                "risks": ["Web evidence is placeholder-only; portfolio stance should remain non-actionable."],
                "evidence_count": len(evidence),
                "sentiment": "unavailable",
                "sentiment_score": 0.0,
                "confidence": 0.05,
                "evidence_mode": "fallback_only",
                "fallback_count": fallback_count,
                "live_evidence_count": 0,
            }

        key_points = [self._short_point(item) for item in live_evidence[:3]]
        opportunities = [point for point in key_points if "positive" in point]
        risks = [point for point in key_points if "negative" in point]

        score = self._sentiment_score(live_evidence)
        sentiment = "neutral"
        if score > 0.1:
            sentiment = "positive"
        elif score < -0.1:
            sentiment = "negative"

        if not opportunities and sentiment == "positive":
            opportunities.append("Coverage skews positive across recent headlines.")
        if not risks and sentiment == "negative":
            risks.append("Coverage skews negative across recent headlines.")

        confidence = min(0.9, 0.35 + 0.1 * len(live_evidence))
        evidence_mode = "mixed" if fallback_count else "live"
        if fallback_count:
            risks.append(f"{fallback_count} fallback placeholder item(s) were ignored in web scoring.")
            confidence = min(confidence, 0.35)

        return {
            "summary": (
                f"Web evidence across {len(live_evidence)} live item(s) is {sentiment}."
                if not fallback_count
                else (
                    f"Web evidence across {len(live_evidence)} live item(s) is {sentiment}; "
                    f"{fallback_count} fallback placeholder item(s) were ignored."
                )
            ),
            "key_points": key_points,
            "opportunities": opportunities,
            "risks": risks,
            "evidence_count": len(evidence),
            "sentiment": sentiment,
            "sentiment_score": round(score, 3),
            "confidence": round(confidence, 2),
            "evidence_mode": evidence_mode,
            "fallback_count": fallback_count,
            "live_evidence_count": len(live_evidence),
        }

    def _short_point(self, item: Evidence) -> str:
        tone = self._label_tone(f"{item.title} {item.summary}")
        title = item.title.strip() or "Untitled evidence"
        return f"{title} ({item.source}, {tone})"

    def _label_tone(self, text: str) -> str:
        score = self._token_score(text)
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "mixed"

    def _sentiment_score(self, evidence: list[Evidence]) -> float:
        if not evidence:
            return 0.0
        total = 0.0
        for item in evidence:
            total += self._token_score(f"{item.title} {item.summary}")
        return total / max(len(evidence), 1)

    def _token_score(self, text: str) -> float:
        normalized = text.lower()
        pos = sum(token in normalized for token in _POSITIVE_TOKENS)
        neg = sum(token in normalized for token in _NEGATIVE_TOKENS)
        return float(pos - neg)

    def _is_fallback_evidence(self, item: Evidence) -> bool:
        source = (item.source or "").strip().lower()
        return source.startswith("offline-fallback:") or source.startswith("mcp-fallback:")
