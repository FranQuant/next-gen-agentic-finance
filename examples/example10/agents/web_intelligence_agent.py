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
        actionable_evidence, fallback_evidence = self._split_evidence(evidence)
        evidence_mode = self._evidence_mode(actionable_evidence, fallback_evidence, evidence)

        if not evidence:
            return {
                "summary": "No external evidence was retrieved.",
                "key_points": ["Proceed with low confidence due to sparse web evidence."],
                "opportunities": [],
                "risks": ["Limited external validation."],
                "evidence_count": 0,
                "actionable_evidence_count": 0,
                "fallback_evidence_count": 0,
                "evidence_mode": evidence_mode,
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.1,
                "degraded": True,
            }

        if not actionable_evidence:
            key_points = [self._short_point(item) for item in fallback_evidence[:3]]
            return {
                "summary": (
                    f"Fallback-only web evidence across {len(fallback_evidence)} item(s); "
                    "placeholder context is not treated as sourced research."
                ),
                "key_points": key_points,
                "opportunities": [],
                "risks": [
                    "Fallback-only placeholder web evidence was excluded from directional scoring.",
                ],
                "evidence_count": len(evidence),
                "actionable_evidence_count": 0,
                "fallback_evidence_count": len(fallback_evidence),
                "evidence_mode": evidence_mode,
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.1,
                "degraded": True,
            }

        key_points = [self._short_point(item) for item in actionable_evidence[:3]]
        opportunities = [point for point in key_points if "positive" in point]
        risks = [point for point in key_points if "negative" in point]

        score = self._sentiment_score(actionable_evidence)
        sentiment = "neutral"
        if score > 0.1:
            sentiment = "positive"
        elif score < -0.1:
            sentiment = "negative"

        if not opportunities and sentiment == "positive":
            opportunities.append("Coverage skews positive across recent headlines.")
        if not risks and sentiment == "negative":
            risks.append("Coverage skews negative across recent headlines.")

        confidence = min(0.9, 0.35 + 0.1 * len(actionable_evidence))
        summary = f"Web evidence across {len(actionable_evidence)} live item(s) is {sentiment}."
        degraded = bool(fallback_evidence)
        if fallback_evidence:
            summary = (
                f"Web evidence is {sentiment} across {len(actionable_evidence)} live item(s); "
                f"{len(fallback_evidence)} fallback placeholder item(s) were excluded from scoring."
            )
            risks.append("Fallback placeholder web evidence was ignored in directional scoring.")

        return {
            "summary": summary,
            "key_points": key_points,
            "opportunities": opportunities,
            "risks": risks,
            "evidence_count": len(evidence),
            "actionable_evidence_count": len(actionable_evidence),
            "fallback_evidence_count": len(fallback_evidence),
            "evidence_mode": evidence_mode,
            "sentiment": sentiment,
            "sentiment_score": round(score, 3),
            "confidence": round(confidence, 2),
            "degraded": degraded,
        }

    def _split_evidence(self, evidence: list[Evidence]) -> tuple[list[Evidence], list[Evidence]]:
        actionable: list[Evidence] = []
        fallback: list[Evidence] = []
        for item in evidence:
            if self._is_fallback_item(item):
                fallback.append(item)
            else:
                actionable.append(item)
        return actionable, fallback

    def _evidence_mode(
        self,
        actionable_evidence: list[Evidence],
        fallback_evidence: list[Evidence],
        evidence: list[Evidence],
    ) -> str:
        if not evidence:
            return "none"
        if actionable_evidence and fallback_evidence:
            return "mixed"
        if actionable_evidence:
            return "live"
        return "fallback_only"

    def _is_fallback_item(self, item: Evidence) -> bool:
        source = (item.source or "").strip().lower()
        return source.startswith("fallback:") or source.startswith("stub:")

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
