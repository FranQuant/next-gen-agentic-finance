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
            }

        key_points = [self._short_point(item) for item in evidence[:3]]
        opportunities = [point for point in key_points if "positive" in point]
        risks = [point for point in key_points if "negative" in point]

        score = self._sentiment_score(evidence)
        sentiment = "neutral"
        if score > 0.1:
            sentiment = "positive"
        elif score < -0.1:
            sentiment = "negative"

        if not opportunities and sentiment == "positive":
            opportunities.append("Coverage skews positive across recent headlines.")
        if not risks and sentiment == "negative":
            risks.append("Coverage skews negative across recent headlines.")

        confidence = min(0.9, 0.35 + 0.1 * len(evidence))

        return {
            "summary": f"Web evidence across {len(evidence)} items is {sentiment}.",
            "key_points": key_points,
            "opportunities": opportunities,
            "risks": risks,
            "evidence_count": len(evidence),
            "sentiment": sentiment,
            "sentiment_score": round(score, 3),
            "confidence": round(confidence, 2),
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
