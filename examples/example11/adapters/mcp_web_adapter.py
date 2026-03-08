from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
    from ..schemas import Evidence
except ImportError:  # pragma: no cover
    from schemas import Evidence


class _MCPClient:
    def __init__(self, server_url: str | None, timeout_sec: int, use_live: bool) -> None:
        self.server_url = server_url
        self.timeout_sec = timeout_sec
        self.use_live = use_live

    def call_tool(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        # MCP transport boundary: all outbound capability calls should pass through this method.
        if not self.use_live or not self.server_url:
            raise RuntimeError("MCP web transport not configured.")

        # Placeholder transport for Example11 v1.
        # Replace with a concrete MCP client implementation when server wiring is available.
        raise RuntimeError(f"Live MCP web call unavailable for tool: {tool_name}")


class MCPWebAdapter:
    PREFERRED_SOURCES = (
        "reuters",
        "bloomberg",
        "wsj",
        "ft",
        "cnbc",
        "federal reserve",
        "imf",
    )

    def __init__(
        self,
        server_url: str | None = None,
        timeout_sec: int = 8,
        use_live: bool = False,
        max_results_per_topic: int = 2,
    ) -> None:
        self.client = _MCPClient(server_url=server_url, timeout_sec=timeout_sec, use_live=use_live)
        self.max_results_per_topic = max_results_per_topic

    def search(self, topics: list[str]) -> list[Evidence]:
        if not topics:
            return []

        evidence: list[Evidence] = []
        for topic in topics:
            topic_evidence = self._search_topic(topic)
            evidence.extend(topic_evidence)

        if not evidence:
            return self._fallback(topics, "MCP web search unavailable; placeholder evidence used.")

        return self._diversify(evidence, limit=len(topics) * self.max_results_per_topic)

    def _search_topic(self, topic: str) -> list[Evidence]:
        try:
            response = self.client.call_tool(
                "web.search",
                {
                    "query": f"{topic} market research institutional outlook",
                    "max_results": self.max_results_per_topic * 3,
                },
            )
            raw_results = response.get("results", []) if isinstance(response, dict) else []
            parsed = self._parse_results(raw_results, topic)
            if parsed:
                return parsed[: self.max_results_per_topic]
        except Exception:
            pass

        return self._fallback([topic], "MCP web search unavailable; placeholder evidence used.")

    def _parse_results(self, results: list[dict[str, Any]], topic: str) -> list[Evidence]:
        parsed: list[Evidence] = []
        for item in results:
            title = str(item.get("title", topic)).strip()
            summary = str(item.get("summary") or item.get("content") or "").strip()
            source = str(item.get("source", "mcp:web")).strip().lower()
            score = float(item.get("score", 0.6) or 0.6)
            parsed.append(
                Evidence(
                    source=f"mcp:{source}",
                    title=title,
                    summary=summary[:280],
                    url=str(item.get("url", "")),
                    timestamp=self._now_iso(),
                    relevance=min(1.0, max(0.1, score)),
                )
            )

        ranked = sorted(parsed, key=self._rank_evidence, reverse=True)
        return ranked

    def _rank_evidence(self, item: Evidence) -> float:
        source_name = item.source.lower()
        source_boost = 0.0
        if any(name in source_name for name in self.PREFERRED_SOURCES):
            source_boost = 0.4
        content_boost = min(len(item.summary) / 500.0, 0.2)
        return item.relevance + source_boost + content_boost

    def _diversify(self, evidence: list[Evidence], limit: int) -> list[Evidence]:
        ranked = sorted(evidence, key=self._rank_evidence, reverse=True)
        selected: list[Evidence] = []
        counts: dict[str, int] = {}
        cap_per_source = 2

        for item in ranked:
            if len(selected) >= limit:
                break
            key = item.source
            if counts.get(key, 0) >= cap_per_source:
                continue
            selected.append(item)
            counts[key] = counts.get(key, 0) + 1

        if len(selected) < min(limit, len(ranked)):
            for item in ranked:
                if len(selected) >= limit:
                    break
                if item in selected:
                    continue
                selected.append(item)

        return selected

    def _fallback(self, topics: list[str], note: str) -> list[Evidence]:
        source_cycle = [
            "mcp:reuters",
            "mcp:bloomberg",
            "mcp:federal reserve",
            "mcp:imf",
            "mcp:wsj",
        ]

        fallback: list[Evidence] = []
        for idx, topic in enumerate(topics):
            source_index = (sum(ord(ch) for ch in topic) + idx) % len(source_cycle)
            fallback.append(
                Evidence(
                    source=source_cycle[source_index],
                    title=f"Placeholder intelligence for {topic}",
                    summary=f"{note} Topic tracked for directional context.",
                    timestamp=self._now_iso(),
                    relevance=0.35,
                )
            )
        return fallback

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
