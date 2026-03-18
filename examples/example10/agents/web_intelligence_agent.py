from __future__ import annotations

import re
from urllib.parse import urlparse

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

_OUTLOOK_STYLE_TOKENS = (
    "outlook",
    "market outlook",
    "strategy",
    "considerations",
    "guide",
    "commentary",
    "regime",
    "allocation",
    "weekly",
    "update",
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

        diversified_live_evidence = self._diversify_live_evidence(live_evidence)
        if not diversified_live_evidence:
            diversified_live_evidence = live_evidence[:3]

        key_points = [self._short_point(item) for item in diversified_live_evidence[:3]]
        opportunities = [point for point in key_points if "positive" in point]
        risks = [point for point in key_points if "negative" in point]

        score = self._sentiment_score(diversified_live_evidence)
        outlook_dominated = self._outlook_style_share(diversified_live_evidence) >= 0.5
        # Institutional outlook/research content is intentionally biased toward mixed; positive/negative requires clearer one-sided polarity.
        sentiment = "mixed"
        positive_threshold = 0.5
        negative_threshold = -0.5
        if outlook_dominated:
            positive_threshold = 0.9
            negative_threshold = -0.9

        if score >= positive_threshold:
            sentiment = "positive"
        elif score <= negative_threshold:
            sentiment = "negative"

        if not opportunities and sentiment == "positive":
            opportunities.append("Coverage skews positive across recent headlines.")
        if not risks and sentiment == "negative":
            risks.append("Coverage skews negative across recent headlines.")

        distinct_sources = len({self._source_key(item) for item in diversified_live_evidence})
        confidence = min(
            0.9,
            0.2
            + 0.12 * min(len(diversified_live_evidence), 4)
            + 0.08 * min(distinct_sources, 4),
        )
        evidence_mode = "mixed" if fallback_count else "live"
        if fallback_count:
            risks.append(f"{fallback_count} fallback placeholder item(s) were ignored in web scoring.")
            confidence = min(confidence, 0.35)

        return {
            "summary": (
                f"Web evidence across {len(diversified_live_evidence)} diversified live item(s) "
                f"from {distinct_sources} source(s) is {sentiment}."
                if not fallback_count
                else (
                    f"Web evidence across {len(diversified_live_evidence)} diversified live item(s) "
                    f"from {distinct_sources} source(s) is {sentiment}; "
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

    def _diversify_live_evidence(self, evidence: list[Evidence]) -> list[Evidence]:
        ranked = sorted(
            enumerate(evidence),
            key=lambda pair: (-float(pair[1].relevance), pair[0]),
        )
        selected: list[Evidence] = []
        source_counts: dict[str, int] = {}
        family_keys: set[str] = set()

        for _, item in ranked:
            source_key = self._source_key(item)
            if source_counts.get(source_key, 0) >= 2:
                continue

            family_key = self._title_family_key(item.title)
            if any(self._titles_are_similar(item, existing) for existing in selected):
                continue
            if family_key and family_key in family_keys:
                continue

            selected.append(item)
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
            if family_key:
                family_keys.add(family_key)

        return selected

    def _sentiment_score(self, evidence: list[Evidence]) -> float:
        if not evidence:
            return 0.0
        total = 0.0
        for item in evidence:
            total += self._token_score(f"{item.title} {item.summary}")
        return total / max(len(evidence), 1)

    def _outlook_style_share(self, evidence: list[Evidence]) -> float:
        if not evidence:
            return 0.0
        outlook_count = sum(self._is_outlook_style_evidence(item) for item in evidence)
        return outlook_count / max(len(evidence), 1)

    def _token_score(self, text: str) -> float:
        normalized = text.lower()
        pos = sum(token in normalized for token in _POSITIVE_TOKENS)
        neg = sum(token in normalized for token in _NEGATIVE_TOKENS)
        return float(pos - neg)

    def _is_outlook_style_evidence(self, item: Evidence) -> bool:
        text = f"{item.title} {item.summary}".lower()
        return any(token in text for token in _OUTLOOK_STYLE_TOKENS)

    def _normalized_title(self, text: str) -> str:
        normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
        return re.sub(r"\s+", " ", normalized).strip()

    def _title_family_key(self, title: str) -> str:
        generic_tokens = {
            "the",
            "a",
            "an",
            "and",
            "for",
            "to",
            "of",
            "in",
            "on",
            "with",
            "our",
            "your",
            "market",
            "markets",
            "outlook",
            "guide",
            "weekly",
            "daily",
            "update",
            "commentary",
        }
        tokens = [
            token
            for token in self._normalized_title(title).split()
            if token not in generic_tokens
        ]
        if not tokens:
            return ""
        return " ".join(tokens[:4])

    def _titles_are_similar(self, left: Evidence, right: Evidence) -> bool:
        left_title = self._normalized_title(left.title)
        right_title = self._normalized_title(right.title)
        if not left_title or not right_title:
            return False
        if left_title == right_title:
            return True

        left_tokens = left_title.split()
        right_tokens = right_title.split()
        if left_tokens[:5] == right_tokens[:5] and min(len(left_tokens), len(right_tokens)) >= 5:
            return True

        left_set = set(left_tokens)
        right_set = set(right_tokens)
        overlap = left_set & right_set
        union = left_set | right_set
        if len(union) < 4:
            return False
        return (len(overlap) / len(union)) >= 0.75

    def _source_key(self, item: Evidence) -> str:
        source = (item.source or "").strip().lower()
        if source.startswith(("http://", "https://")):
            host = urlparse(source).netloc.lower()
        else:
            host = urlparse(item.url or "").netloc.lower()
        normalized = host or source or "unknown"
        return normalized.removeprefix("www.")

    def _is_fallback_evidence(self, item: Evidence) -> bool:
        source = (item.source or "").strip().lower()
        return source.startswith("offline-fallback:") or source.startswith("mcp-fallback:")
