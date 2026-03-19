from __future__ import annotations

import re
from datetime import datetime, timezone
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

_BACKGROUND_OUTLOOK_TOKENS = (
    "investment outlook",
    "market outlook",
    "equity market outlook",
    "strategy outlook",
    "annual outlook",
    "quarterly outlook",
    "factor views",
    "year ahead",
    "year-ahead",
)

_TACTICAL_RECENT_TOKENS = (
    "latest",
    "current",
    "today",
    "near-term",
    "near term",
    "weekly",
    "market commentary",
    "weekly commentary",
    "market update",
    "tactical note",
    "market note",
)

_TACTICAL_QUERY_TOKENS = (
    "tactical",
    "current",
    "now",
    "latest",
    "today",
    "near-term",
    "near term",
    "overweight",
    "underweight",
    "current macro",
    "current market",
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

        tactical_brief = self._is_tactical_brief(brief)
        diversified_live_evidence = self._diversify_live_evidence(live_evidence, tactical=tactical_brief)
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

        recent_count = self._recent_item_count(diversified_live_evidence, max_age_days=45)
        freshness_suffix = ""
        if tactical_brief:
            freshness_suffix = (
                f"; freshness-aware tactical ranking favored {recent_count} recent item(s)"
                if recent_count
                else "; freshness-aware tactical ranking was applied"
            )

        return {
            "summary": (
                f"Web evidence across {len(diversified_live_evidence)} diversified live item(s) "
                f"from {distinct_sources} source(s) is {sentiment}{freshness_suffix}."
                if not fallback_count
                else (
                    f"Web evidence across {len(diversified_live_evidence)} diversified live item(s) "
                    f"from {distinct_sources} source(s) is {sentiment}{freshness_suffix}; "
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

    def _diversify_live_evidence(self, evidence: list[Evidence], tactical: bool = False) -> list[Evidence]:
        ranked = list(enumerate(evidence))
        if not tactical:
            ranked = sorted(
                ranked,
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

        if tactical:
            selected = self._rebalance_tactical_slice(selected)

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

    def _is_tactical_brief(self, brief: ResearchBrief) -> bool:
        normalized = " ".join(
            [
                brief.query,
                brief.objective,
                " ".join(brief.topics),
                " ".join(brief.constraints),
            ]
        ).lower()
        return any(token in normalized for token in _TACTICAL_QUERY_TOKENS)

    def _recent_item_count(self, evidence: list[Evidence], max_age_days: int) -> int:
        count = 0
        for item in evidence:
            parsed = self._parse_timestamp(item.timestamp)
            if parsed is None:
                continue
            age_days = (datetime.now(timezone.utc) - parsed).total_seconds() / 86400.0
            if age_days <= max_age_days:
                count += 1
        return count

    def _parse_timestamp(self, value: str) -> datetime | None:
        text = (value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _rebalance_tactical_slice(self, evidence: list[Evidence]) -> list[Evidence]:
        if len(evidence) < 2:
            return evidence

        priority: list[Evidence] = []
        background: list[Evidence] = []
        for item in evidence:
            if self._is_background_outlook_item(item) and not self._is_tactical_priority_item(item):
                background.append(item)
            else:
                priority.append(item)

        if not priority:
            return self._interleave_by_source(evidence)

        return self._interleave_by_source(priority) + background

    def _interleave_by_source(self, evidence: list[Evidence]) -> list[Evidence]:
        buckets: dict[str, list[Evidence]] = {}
        source_order: list[str] = []
        for item in evidence:
            source_key = self._source_key(item)
            if source_key not in buckets:
                buckets[source_key] = []
                source_order.append(source_key)
            buckets[source_key].append(item)

        interleaved: list[Evidence] = []
        active_sources = list(source_order)
        while active_sources:
            next_sources: list[str] = []
            for source in active_sources:
                bucket = buckets.get(source, [])
                if not bucket:
                    continue
                interleaved.append(bucket.pop(0))
                if bucket:
                    next_sources.append(source)
            active_sources = next_sources

        return interleaved

    def _is_background_outlook_item(self, item: Evidence) -> bool:
        text = f"{item.title} {item.summary}".lower()
        if any(token in text for token in _BACKGROUND_OUTLOOK_TOKENS):
            return True
        if re.search(r"\b(?:19|20)\d{2}\s+outlooks?\b", text):
            return True
        return "investment directions" in text and "outlook" in text

    def _is_tactical_priority_item(self, item: Evidence) -> bool:
        text = f"{item.title} {item.summary}".lower()
        parsed = self._parse_timestamp(item.timestamp)
        if parsed is not None:
            age_days = (datetime.now(timezone.utc) - parsed).total_seconds() / 86400.0
            if age_days <= 45:
                return True
        return any(token in text for token in _TACTICAL_RECENT_TOKENS)

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
