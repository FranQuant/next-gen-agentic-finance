from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from ..schemas import Evidence
except ImportError:  # pragma: no cover
    from schemas import Evidence


class _MCPClient:
    def __init__(
        self,
        server_url: str | None,
        timeout_sec: int,
        use_live: bool,
        transport: str = "stdio",
        server_command: str | None = None,
        server_args: list[str] | None = None,
    ) -> None:
        self.server_url = server_url
        self.timeout_sec = timeout_sec
        self.use_live = use_live
        self.transport = (transport or "stdio").strip().lower()
        self.server_command = server_command
        self.server_args = server_args or []

    def call_tool(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        # MCP transport boundary: all outbound capability calls should pass through this method.
        if not self.use_live:
            raise RuntimeError("MCP live mode disabled.")

        if self.transport in {"stdio", "command"} and not self.server_command:
            raise RuntimeError("MCP stdio transport requires EXAMPLE11_MCP_WEB_SERVER_COMMAND.")
        if self.transport in {"streamable_http", "http", "sse"} and not self.server_url:
            raise RuntimeError("MCP URL transport requires EXAMPLE11_MCP_WEB_SERVER_URL.")

        return asyncio.run(self._call_tool_async(tool_name=tool_name, payload=payload))

    async def _call_tool_async(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        from mcp import ClientSession, StdioServerParameters, stdio_client

        if self.transport in {"stdio", "command"}:
            server = StdioServerParameters(
                command=self.server_command or "",
                args=self.server_args,
            )
            async with stdio_client(server) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self.timeout_sec),
                ) as session:
                    return await self._invoke(session, tool_name, payload)

        if self.transport == "sse":
            from mcp.client.sse import sse_client

            async with sse_client(
                self.server_url or "",
                timeout=float(self.timeout_sec),
            ) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self.timeout_sec),
                ) as session:
                    return await self._invoke(session, tool_name, payload)

        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(
            self.server_url or "",
            timeout=float(self.timeout_sec),
        ) as (read_stream, write_stream, _session_id):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self.timeout_sec),
            ) as session:
                return await self._invoke(session, tool_name, payload)

    async def _invoke(self, session: Any, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        await session.initialize()

        selected_tool = tool_name
        tool_names: list[str] = []
        try:
            tools_result = await session.list_tools()
            for tool in getattr(tools_result, "tools", []):
                name = getattr(tool, "name", None)
                if isinstance(name, str) and name:
                    tool_names.append(name)
        except Exception:
            tool_names = []

        if tool_names and selected_tool not in tool_names:
            selected_tool = self._pick_best_tool(tool_names, preferred=selected_tool)

        result = await session.call_tool(
            selected_tool,
            arguments=payload,
            read_timeout_seconds=timedelta(seconds=self.timeout_sec),
        )

        if hasattr(result, "model_dump"):
            data = result.model_dump()
        elif hasattr(result, "dict"):
            data = result.dict()  # pragma: no cover
        elif isinstance(result, dict):
            data = result
        else:
            data = {"content": []}

        data["_used_tool"] = selected_tool
        data["_available_tools"] = tool_names
        return data

    def _pick_best_tool(self, tool_names: list[str], preferred: str) -> str:
        preferred_lower = preferred.lower()
        for name in tool_names:
            if name.lower() == preferred_lower:
                return name

        candidates = (
            "web.search",
            "search",
            "tavily",
            "news",
            "query",
        )
        for keyword in candidates:
            for name in tool_names:
                if keyword in name.lower():
                    return name

        return tool_names[0] if tool_names else preferred


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
        transport: str = "stdio",
        server_command: str | None = None,
        server_args: list[str] | None = None,
        tool_name: str = "web.search",
    ) -> None:
        self.client = _MCPClient(
            server_url=server_url,
            timeout_sec=timeout_sec,
            use_live=use_live,
            transport=transport,
            server_command=server_command,
            server_args=server_args,
        )
        self.max_results_per_topic = max_results_per_topic
        self.tool_name = tool_name

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
                self.tool_name,
                {
                    "query": f"{topic} market research institutional outlook",
                    "max_results": self.max_results_per_topic * 3,
                },
            )
            if isinstance(response, dict) and response.get("isError"):
                raise RuntimeError("MCP tool returned an error payload.")

            raw_results = self._extract_raw_results(response, topic)
            parsed = self._parse_results(raw_results, topic, live=True)
            if parsed:
                return parsed[: self.max_results_per_topic]
        except Exception:
            pass

        return self._fallback([topic], "MCP web search unavailable; placeholder evidence used.")

    def _extract_raw_results(self, response: dict[str, Any], topic: str) -> list[dict[str, Any]]:
        if not isinstance(response, dict):
            return []

        direct = response.get("results")
        if isinstance(direct, list):
            return [item for item in direct if isinstance(item, dict)]

        structured = response.get("structuredContent")
        structured_items = self._extract_structured_items(structured)
        if structured_items:
            return structured_items

        content = response.get("content")
        content_items = self._extract_content_items(content, topic=topic)
        if content_items:
            return content_items

        return []

    def _extract_structured_items(self, structured: Any) -> list[dict[str, Any]]:
        if isinstance(structured, list):
            return [item for item in structured if isinstance(item, dict)]

        if isinstance(structured, dict):
            for key in ("results", "items", "documents", "articles", "data"):
                maybe = structured.get(key)
                if isinstance(maybe, list):
                    return [item for item in maybe if isinstance(item, dict)]

            if any(key in structured for key in ("title", "summary", "content", "text", "url")):
                return [structured]

        return []

    def _extract_content_items(self, content: Any, topic: str) -> list[dict[str, Any]]:
        if not isinstance(content, list):
            return []

        items: list[dict[str, Any]] = []
        for block in content:
            if hasattr(block, "model_dump"):
                block_data = block.model_dump()
            elif isinstance(block, dict):
                block_data = block
            else:
                block_data = {}

            if block_data.get("type") != "text":
                continue

            text = str(block_data.get("text", "")).strip()
            if not text:
                continue

            parsed = self._parse_text_payload(text, topic)
            items.extend(parsed)

        return items

    def _parse_text_payload(self, text: str, topic: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(text)
        except Exception:
            return [{"title": topic, "summary": text, "source": "web"}]

        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            nested = self._extract_structured_items(payload)
            if nested:
                return nested
            return [payload]
        return [{"title": topic, "summary": text, "source": "web"}]

    def _parse_results(self, results: list[dict[str, Any]], topic: str, live: bool) -> list[Evidence]:
        parsed: list[Evidence] = []
        for item in results:
            title = str(item.get("title") or item.get("headline") or topic).strip()
            summary = str(
                item.get("summary")
                or item.get("content")
                or item.get("text")
                or item.get("snippet")
                or ""
            ).strip()
            source = str(item.get("source") or item.get("publisher") or "web").strip().lower()
            score = float(item.get("score") or item.get("relevance") or 0.6)

            prefix = "mcp-live" if live else "mcp-fallback"
            parsed.append(
                Evidence(
                    source=f"{prefix}:{source}",
                    title=title,
                    summary=summary[:280],
                    url=str(item.get("url") or item.get("link") or ""),
                    timestamp=self._now_iso(),
                    relevance=min(1.0, max(0.1, score)),
                )
            )

        return sorted(parsed, key=self._rank_evidence, reverse=True)

    def _rank_evidence(self, item: Evidence) -> float:
        source_name = self._source_name(item.source)
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

    def _source_name(self, source: str) -> str:
        parts = source.split(":", 1)
        if len(parts) == 2:
            return parts[1].lower()
        return source.lower()

    def _fallback(self, topics: list[str], note: str) -> list[Evidence]:
        source_cycle = [
            "mcp-fallback:reuters",
            "mcp-fallback:bloomberg",
            "mcp-fallback:federal reserve",
            "mcp-fallback:imf",
            "mcp-fallback:wsj",
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
