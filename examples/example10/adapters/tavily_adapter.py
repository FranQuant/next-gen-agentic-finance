from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import urlparse

try:
    from ..schemas import Evidence
except ImportError:  # pragma: no cover
    from schemas import Evidence


class TavilyAdapter:
    INSTITUTIONAL_DOMAINS = [
        "federalreserve.gov",
        "fred.stlouisfed.org",
        "imf.org",
        "worldbank.org",
        "bis.org",
        "oecd.org",
        "sec.gov",
        "treasury.gov",
    ]
    MARKET_NEWS_DOMAINS = [
        "reuters.com",
        "bloomberg.com",
        "wsj.com",
        "ft.com",
        "cnbc.com",
    ]
    PREFERRED_DOMAINS = INSTITUTIONAL_DOMAINS + MARKET_NEWS_DOMAINS
    LOW_QUALITY_DOMAINS = [
        "walletinvestor.com",
        "gov.capital",
        "coinpriceforecast.com",
        "stockinvest.us",
        "longforecast.com",
    ]
    GENERIC_TOPIC_WORDS = {
        "market",
        "markets",
        "research",
        "outlook",
        "analysis",
        "latest",
        "news",
        "institutional",
        "analyze",
        "analysis",
        "assess",
        "evaluate",
        "whether",
        "current",
        "support",
        "supports",
        "view",
        "portfolio",
        "month",
        "months",
        "context",
        "conditions",
        "given",
        "deserve",
        "tactical",
        "overweight",
        "underweight",
        "macro",
        "and",
        "for",
        "with",
        "the",
        "a",
        "an",
        "to",
    }

    def __init__(self, api_key: str | None = None, max_results: int = 2) -> None:
        self.api_key = api_key
        self.max_results = max_results
        self._client = None

        if api_key:
            try:
                from tavily import TavilyClient

                self._client = TavilyClient(api_key=api_key)
            except Exception:
                self._client = None

    def search(self, topics: list[str]) -> list[Evidence]:
        if not topics:
            return []

        if not self._client:
            return self._fallback(topics, "Tavily client unavailable; using placeholder evidence.")

        evidence: list[Evidence] = []
        for topic in topics:
            try:
                evidence.extend(self._search_topic(topic))
            except Exception:
                evidence.extend(self._fallback([topic], "Tavily request failed; placeholder evidence used."))

        if not evidence:
            return self._fallback(topics, "No Tavily results returned.")
        evidence = self._dedupe_evidence(evidence)
        return self._diversify_evidence(evidence, limit=self.max_results * len(topics))

    def _search_topic(self, topic: str) -> list[Evidence]:
        query = f"{topic} market research institutional outlook"
        institutional_results = self._search_raw(query, include_domains=self.INSTITUTIONAL_DOMAINS)
        market_news_results = self._search_raw(query, include_domains=self.MARKET_NEWS_DOMAINS)
        broad_results = self._search_raw(query)

        merged = self._merge_results(institutional_results + market_news_results + broad_results)
        filtered = [
            result
            for result in merged
            if not self._is_low_quality(result.get("url", ""))
        ]
        candidates = filtered or merged

        ranked = sorted(candidates, key=lambda item: self._rank_result(item, topic), reverse=True)
        picked = self._pick_diverse(ranked, limit=self.max_results)
        return [
            Evidence(
                source=f"tavily:{self._extract_domain(str(result.get('url', ''))) or 'web'}",
                title=str(result.get("title", topic)),
                summary=str(result.get("content", ""))[:280],
                url=str(result.get("url", "")),
                timestamp=self._now_iso(),
                relevance=float(result.get("score", 0.6) or 0.6),
            )
            for result in picked
        ]

    def _search_raw(self, query: str, include_domains: list[str] | None = None) -> list[dict]:
        if not self._client:
            return []

        base_args = {
            "query": query,
            "max_results": self.max_results * 3,
            "search_depth": "advanced",
            "exclude_domains": self.LOW_QUALITY_DOMAINS,
            "topic": "news",
        }
        if include_domains:
            base_args["include_domains"] = include_domains

        try:
            payload = self._client.search(**base_args)
        except TypeError:
            payload = self._client.search(query=query, max_results=self.max_results * 2)
        except Exception:
            return []

        if isinstance(payload, dict):
            results = payload.get("results", [])
            return results if isinstance(results, list) else []
        return []

    def _merge_results(self, results: list[dict]) -> list[dict]:
        deduped: list[dict] = []
        seen: set[str] = set()
        for result in results:
            url = str(result.get("url", "")).strip().lower()
            title = str(result.get("title", "")).strip().lower()
            key = url or title
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(result)
        return deduped

    def _rank_result(self, result: dict, topic: str) -> float:
        domain = self._extract_domain(str(result.get("url", "")))
        source_boost = 1.0 if self._is_preferred(domain) else 0.0
        base_score = float(result.get("score", 0.0) or 0.0)
        content = str(result.get("content", ""))
        content_boost = min(len(content) / 600.0, 0.4)
        topic_boost = self._topic_match_score(result, topic)
        return source_boost + base_score + content_boost + topic_boost

    def _is_preferred(self, domain: str) -> bool:
        return any(domain == preferred or domain.endswith(f".{preferred}") for preferred in self.PREFERRED_DOMAINS)

    def _is_low_quality(self, url: str) -> bool:
        domain = self._extract_domain(url)
        return any(domain == low or domain.endswith(f".{low}") for low in self.LOW_QUALITY_DOMAINS)

    def _extract_domain(self, url: str) -> str:
        parsed = urlparse(url)
        return parsed.netloc.lower().replace("www.", "").strip()

    def _topic_match_score(self, result: dict, topic: str) -> float:
        topic_tokens = self._topic_tokens(topic)
        if not topic_tokens:
            return 0.0

        entity_tokens = self._entity_tokens(topic)
        haystack = " ".join(
            [
                str(result.get("title", "")),
                str(result.get("content", "")),
                str(result.get("url", "")),
            ]
        ).lower()

        matched_tokens = [token for token in topic_tokens if token in haystack]
        boost = min(0.75, 0.18 * len(matched_tokens))

        if entity_tokens:
            matched_entities = [token for token in entity_tokens if token in haystack]
            if matched_entities:
                boost += min(0.35, 0.2 * len(matched_entities))
            else:
                boost -= 0.4
        elif not matched_tokens:
            boost -= 0.1

        return boost

    def _topic_tokens(self, topic: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9]+", topic)
        cleaned: list[str] = []
        for token in tokens:
            normalized = token.lower()
            if len(normalized) <= 1:
                continue
            if normalized.isdigit():
                continue
            if normalized in self.GENERIC_TOPIC_WORDS:
                continue
            if normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned

    def _entity_tokens(self, topic: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9]+", topic)
        entities: list[str] = []
        for token in tokens:
            normalized = token.lower()
            if normalized in self.GENERIC_TOPIC_WORDS:
                continue
            if token.isupper() and len(token) >= 2:
                entities.append(normalized)
                continue
            if token[:1].isupper() and any(char.islower() for char in token[1:]) and len(token) > 3:
                entities.append(normalized)

        deduped: list[str] = []
        for token in entities:
            if token not in deduped:
                deduped.append(token)
        return deduped

    def _pick_diverse(self, ranked: list[dict], limit: int) -> list[dict]:
        selected: list[dict] = []
        domain_counts: dict[str, int] = {}

        for result in ranked:
            if len(selected) >= limit:
                break
            domain = self._extract_domain(str(result.get("url", ""))) or "unknown"
            if domain_counts.get(domain, 0) >= 1:
                continue
            selected.append(result)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        if len(selected) < limit:
            for result in ranked:
                if len(selected) >= limit:
                    break
                if result in selected:
                    continue
                selected.append(result)

        return selected

    def _diversify_evidence(self, evidence: list[Evidence], limit: int) -> list[Evidence]:
        ranked = sorted(evidence, key=lambda item: item.relevance, reverse=True)
        selected: list[Evidence] = []
        domain_counts: dict[str, int] = {}
        domain_cap = 2

        for item in ranked:
            if len(selected) >= limit:
                break
            domain = item.source.split(":", 1)[-1]
            if domain_counts.get(domain, 0) >= domain_cap:
                continue
            selected.append(item)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        if len(selected) < min(limit, len(ranked)):
            for item in ranked:
                if len(selected) >= limit:
                    break
                if item in selected:
                    continue
                selected.append(item)

        return selected

    def _dedupe_evidence(self, evidence: list[Evidence]) -> list[Evidence]:
        ranked = sorted(evidence, key=lambda item: item.relevance, reverse=True)
        selected: list[Evidence] = []
        seen_keys: set[str] = set()
        seen_title_tokens: list[set[str]] = []

        for item in ranked:
            primary_key = self._normalized_url(item.url) or self._normalized_title(item.title)
            if primary_key and primary_key in seen_keys:
                continue

            title_tokens = self._title_token_set(item.title)
            if title_tokens and self._is_near_duplicate_title(title_tokens, seen_title_tokens):
                continue

            selected.append(item)
            if primary_key:
                seen_keys.add(primary_key)
            if title_tokens:
                seen_title_tokens.append(title_tokens)

        return selected

    def _normalized_url(self, url: str) -> str:
        if not url:
            return ""

        parsed = urlparse(url.strip())
        domain = parsed.netloc.lower().replace("www.", "").strip()
        path = parsed.path.rstrip("/").strip().lower()
        if not domain and not path:
            return ""
        return f"{domain}{path}"

    def _normalized_title(self, title: str) -> str:
        if not title:
            return ""

        base = title.split(" - ", 1)[0].split(" | ", 1)[0].lower()
        words = re.findall(r"[a-z0-9]+", base)
        filtered = [word for word in words if word not in self.GENERIC_TOPIC_WORDS]
        return " ".join(filtered)

    def _title_token_set(self, title: str) -> set[str]:
        normalized = self._normalized_title(title)
        if not normalized:
            return set()
        return {token for token in normalized.split() if len(token) > 2}

    def _is_near_duplicate_title(
        self,
        title_tokens: set[str],
        seen_title_tokens: list[set[str]],
    ) -> bool:
        for existing in seen_title_tokens:
            if not existing:
                continue
            overlap = len(title_tokens & existing)
            minimum = min(len(title_tokens), len(existing))
            if minimum > 0 and overlap / minimum >= 0.8:
                return True
        return False

    def _fallback(self, topics: list[str], note: str) -> list[Evidence]:
        return [
            Evidence(
                source="fallback:tavily",
                title=f"Fallback placeholder for {topic}",
                summary=(
                    f"Fallback evidence only. {note} "
                    "Topic tracked for directional context; not live sourced research."
                ),
                timestamp=self._now_iso(),
                relevance=0.1,
            )
            for topic in topics
        ]

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
