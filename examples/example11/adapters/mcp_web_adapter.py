from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

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

        if "extract" in preferred_lower:
            candidates = (
                "tavily-extract",
                "extract",
                "crawl",
                "scrape",
            )
        else:
            candidates = (
                "tavily-search",
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
    RESULT_CONTAINER_KEYS = (
        "results",
        "items",
        "documents",
        "articles",
        "sources",
        "data",
    )
    TITLE_KEYS = (
        "title",
        "headline",
        "name",
        "page_title",
        "article_title",
    )
    URL_KEYS = (
        "url",
        "link",
        "href",
        "uri",
    )
    SEARCH_SNIPPET_KEYS = (
        "summary",
        "snippet",
        "content",
        "text",
        "description",
        "excerpt",
        "raw_content",
        "body",
        "markdown",
    )
    EXTRACT_SNIPPET_KEYS = (
        "raw_content",
        "content",
        "text",
        "summary",
        "excerpt",
        "description",
        "body",
        "markdown",
    )
    DOMAIN_KEYS = (
        "domain",
        "source",
        "publisher",
        "site_name",
        "site",
        "host",
        "hostname",
        "favicon",
    )
    SCORE_KEYS = (
        "score",
        "relevance",
        "similarity",
    )
    RANK_KEYS = (
        "rank",
        "position",
    )
    PREFERRED_SOURCES = (
        "reuters.com",
        "bloomberg.com",
        "wsj.com",
        "ft.com",
        "cnbc.com",
        "federalreserve.gov",
        "fred.stlouisfed.org",
        "imf.org",
        "bis.org",
        "worldbank.org",
        "invesco.com",
        "blackrock.com",
        "ishares.com",
        "nomuranow.com",
        "jpmorgan.com",
        "goldmansachs.com",
        "gs.com",
    )
    LOW_QUALITY_DOMAINS = (
        "bitget.com",
        "tickeron.com",
        "gurufocus.com",
        "rollingout.com",
        "walletinvestor.com",
        "gov.capital",
        "coinpriceforecast.com",
        "stockinvest.us",
        "longforecast.com",
        "marketbeat.com",
    )
    SOURCE_DOMAIN_MAP = {
        "reuters": "reuters.com",
        "bloomberg": "bloomberg.com",
        "wsj": "wsj.com",
        "wall street journal": "wsj.com",
        "financial times": "ft.com",
        "ft": "ft.com",
        "cnbc": "cnbc.com",
        "federal reserve": "federalreserve.gov",
        "fed": "federalreserve.gov",
        "imf": "imf.org",
        "bis": "bis.org",
        "bank for international settlements": "bis.org",
        "world bank": "worldbank.org",
        "invesco": "invesco.com",
        "blackrock": "blackrock.com",
        "ishares": "ishares.com",
        "nomura": "nomuranow.com",
        "j.p. morgan": "jpmorgan.com",
        "jp morgan": "jpmorgan.com",
        "jpmorgan": "jpmorgan.com",
        "goldman sachs": "goldmansachs.com",
        "goldman": "goldmansachs.com",
    }

    def __init__(
        self,
        server_url: str | None = None,
        timeout_sec: int = 8,
        use_live: bool = False,
        max_results_per_topic: int = 2,
        transport: str = "stdio",
        server_command: str | None = None,
        server_args: list[str] | None = None,
        tool_name: str = "tavily-search",
        extract_tool_name: str = "tavily-extract",
        enable_extract_enrichment: bool = True,
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
        self.extract_tool_name = extract_tool_name
        self.enable_extract_enrichment = enable_extract_enrichment
        self.last_search_report = self._empty_search_report()

    def search(self, topics: list[str]) -> list[Evidence]:
        if not topics:
            return []

        self.last_search_report = self._empty_search_report(topics)
        evidence: list[Evidence] = []
        for topic in topics:
            evidence.extend(self._search_topic(topic))

        if not evidence:
            return self._fallback(topics, "MCP web search unavailable; placeholder evidence used.")

        return self._diversify(evidence, limit=len(topics) * self.max_results_per_topic)

    def _build_queries(self, topic: str) -> list[str]:
        focus = self._topic_focus(topic)
        if self._looks_like_asset_topic(topic, focus):
            variants = [
                f"{focus} institutional outlook",
                f"{focus} macro outlook",
                f"{focus} investment outlook",
                f"{focus} strategy outlook",
            ]
        else:
            variants = [
                f"{focus} institutional market outlook",
                f"{focus} macro outlook",
                f"{focus} investment strategy outlook",
            ]

        queries: list[str] = []
        for query in variants:
            cleaned = self._clean_text(query)
            if cleaned and cleaned not in queries:
                queries.append(cleaned)
        return queries or [self._clean_text(topic)]

    def _build_search_payloads(self, topic: str) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for query in self._build_queries(topic):
            payloads.append(
                {
                    "query": query,
                    "max_results": self.max_results_per_topic * 4,
                    "search_depth": "advanced",
                    "topic": "general",
                    "include_favicon": True,
                    "include_domains": list(self.PREFERRED_SOURCES),
                    "exclude_domains": list(self.LOW_QUALITY_DOMAINS),
                }
            )
            payloads.append(
                {
                    "query": query,
                    "max_results": self.max_results_per_topic * 3,
                    "topic": "general",
                    "include_favicon": True,
                    "exclude_domains": list(self.LOW_QUALITY_DOMAINS),
                }
            )
            payloads.append({"query": query})
        return payloads

    def _search_topic(self, topic: str) -> list[Evidence]:
        payloads = self._build_search_payloads(topic)

        for payload in payloads:
            query = self._clean_text(payload.get("query") or topic)
            try:
                response = self.client.call_tool(self.tool_name, payload)
                if isinstance(response, dict) and response.get("isError"):
                    self._append_report_entry(
                        "search_calls",
                        {
                            "topic": topic,
                            "query": query,
                            "tool": str(response.get("_used_tool") or self.tool_name),
                            "payload_keys": sorted(payload.keys()),
                            "error": "error_payload",
                        },
                    )
                    continue

                self.last_search_report["live_mcp_used"] = True
                records, trace = self._normalize_live_records(response=response, topic=topic)
                needs_extract = bool(records) and self._needs_extract_enrichment(records)
                self._append_report_entry(
                    "search_calls",
                    {
                        "topic": topic,
                        "query": query,
                        "tool": str(response.get("_used_tool") or self.tool_name),
                        "payload_keys": sorted(payload.keys()),
                        "parsed_outputs": trace.get("parsed_outputs", []),
                        "raw_results": trace.get("raw_results", 0),
                        "normalized_results": len(records),
                        "needs_extract": needs_extract,
                    },
                )
                if not records:
                    continue

                if self.enable_extract_enrichment and needs_extract:
                    records = self._apply_extract_enrichment(records, topic=topic)

                parsed = self._records_to_evidence(records=records, live=True)
                if parsed:
                    return parsed[: self.max_results_per_topic]
            except Exception as exc:
                self._append_report_entry(
                    "search_calls",
                    {
                        "topic": topic,
                        "query": query,
                        "tool": self.tool_name,
                        "payload_keys": sorted(payload.keys()),
                        "error": type(exc).__name__,
                    },
                )
                continue

        return self._fallback([topic], "MCP web search unavailable; placeholder evidence used.")

    def _normalize_live_records(
        self,
        response: dict[str, Any],
        topic: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        response_answer = self._extract_response_answer(response)
        raw_results = self._extract_raw_results(response=response, topic=topic)
        if not raw_results:
            return [], {"parsed_outputs": [], "raw_results": 0}

        records: list[dict[str, Any]] = []
        origins: set[str] = set()
        seen: set[str] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                continue

            origin = self._clean_text(item.get("_origin"))
            if origin:
                origins.add(origin)

            normalized = self._normalize_result_item(
                item=item,
                topic=topic,
                fallback_snippet=response_answer,
                snippet_keys=self.SEARCH_SNIPPET_KEYS,
            )
            if not normalized:
                continue

            key = (str(normalized.get("url") or normalized.get("title") or "")).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            records.append(normalized)

        records = self._prune_low_quality_records(records)
        return (
            sorted(records, key=self._record_rank, reverse=True),
            {
                "parsed_outputs": sorted(origins),
                "raw_results": len(raw_results),
            },
        )

    def _extract_raw_results(self, response: dict[str, Any], topic: str) -> list[dict[str, Any]]:
        if not isinstance(response, dict):
            return []

        structured_items = self._extract_structured_items(response.get("structuredContent"), origin="structuredContent")
        if structured_items:
            return structured_items

        direct = response.get("results")
        if isinstance(direct, list):
            return [self._with_origin(item, "results") for item in direct if isinstance(item, dict)]

        content_items = self._extract_content_items(response.get("content"), topic=topic)
        if content_items:
            return content_items

        return []

    def _extract_structured_items(
        self,
        structured: Any,
        origin: str = "structuredContent",
        depth: int = 0,
    ) -> list[dict[str, Any]]:
        if depth > 5:
            return []

        if isinstance(structured, list):
            items: list[dict[str, Any]] = []
            for idx, item in enumerate(structured):
                if not isinstance(item, dict):
                    continue
                if self._looks_like_result_item(item):
                    items.append(self._with_origin(item, origin))
                    continue
                items.extend(self._extract_structured_items(item, origin=f"{origin}[{idx}]", depth=depth + 1))
            return items

        if isinstance(structured, dict):
            items: list[dict[str, Any]] = []
            for key in self.RESULT_CONTAINER_KEYS:
                maybe = structured.get(key)
                if maybe is None:
                    continue
                items.extend(self._extract_structured_items(maybe, origin=f"{origin}.{key}", depth=depth + 1))
            if items:
                return items

            if self._looks_like_result_item(structured):
                return [self._with_origin(structured, origin)]

            for key, value in structured.items():
                if not isinstance(value, (dict, list)):
                    continue
                items.extend(self._extract_structured_items(value, origin=f"{origin}.{key}", depth=depth + 1))
            if items:
                return items

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

            block_type = str(block_data.get("type", "")).strip().lower()

            if block_type == "text" or block_data.get("text") is not None:
                text = self._flatten_text(block_data.get("text"))
                if not text:
                    continue
                items.extend(self._parse_text_payload(text=text, topic=topic, origin="content.text"))
                continue

            if block_type == "resource":
                resource = block_data.get("resource")
                if isinstance(resource, dict):
                    resource_url = self._pick_first_url(resource)
                    resource_text = self._flatten_text(resource.get("text") or resource.get("contents"))
                    if resource_url or resource_text:
                        items.append(
                            {
                                "title": topic,
                                "url": resource_url,
                                "content": resource_text,
                                "source": self._extract_domain(resource_url) or "web",
                                "_origin": "content.resource",
                            }
                        )

        return items

    def _parse_text_payload(self, text: str, topic: str, origin: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(text)
        except Exception:
            return [{"title": topic, "summary": text, "source": "web", "_origin": f"{origin}.raw"}]

        if isinstance(payload, list):
            return [self._with_origin(item, f"{origin}.json") for item in payload if isinstance(item, dict)]

        if isinstance(payload, dict):
            nested = self._extract_structured_items(payload, origin=f"{origin}.json")
            if nested:
                return nested
            return [self._with_origin(payload, f"{origin}.json")]

        return [{"title": topic, "summary": text, "source": "web", "_origin": f"{origin}.raw"}]

    def _is_thin(self, records: list[dict[str, Any]]) -> bool:
        if len(records) < self.max_results_per_topic:
            return True

        top = records[: self.max_results_per_topic]
        snippet_lengths = [len(str(item.get("snippet", "")).strip()) for item in top]
        avg_len = sum(snippet_lengths) / max(1, len(snippet_lengths))
        return avg_len < 120

    def _needs_extract_enrichment(self, records: list[dict[str, Any]]) -> bool:
        if self._is_thin(records):
            return True

        top = records[: min(3, len(records))]
        weak_items = 0
        for item in top:
            domain = self._clean_text(item.get("domain") or "").lower()
            snippet_len = len(self._clean_text(item.get("snippet") or ""))
            if self._is_low_quality_domain(domain) or snippet_len < 180:
                weak_items += 1
        return weak_items >= 2

    def _apply_extract_enrichment(self, records: list[dict[str, Any]], topic: str) -> list[dict[str, Any]]:
        candidate_urls = self._pick_extract_urls(records, topic=topic)
        if not candidate_urls:
            return records

        enrichment_map = self._fetch_extract_map(candidate_urls, topic=topic)
        if not enrichment_map:
            return records

        enriched: list[dict[str, Any]] = []
        for item in records:
            cloned = dict(item)
            url = str(cloned.get("url", "")).strip()
            extract_record = enrichment_map.get(url)
            if not extract_record:
                enriched.append(cloned)
                continue

            updated = False
            extracted_title = self._clean_text(extract_record.get("title") or "")
            extracted_text = self._clean_text(extract_record.get("snippet") or "")
            extracted_domain = self._clean_text(extract_record.get("domain") or "")
            current_title = self._clean_text(cloned.get("title") or "")
            current_snippet = self._clean_text(cloned.get("snippet") or "")
            current_domain = self._clean_text(cloned.get("domain") or "web").lower()

            if extracted_title and self._is_generic_title(current_title, topic):
                cloned["title"] = extracted_title
                updated = True
            if extracted_text and len(current_snippet) < max(180, len(extracted_text) // 2):
                cloned["snippet"] = extracted_text[:280]
                updated = True
            if extracted_domain and current_domain == "web":
                cloned["domain"] = extracted_domain
                updated = True
            if updated:
                cloned["extract_enriched"] = True
                cloned["relevance"] = min(1.0, float(cloned.get("relevance", 0.6)) + 0.05)
                self.last_search_report["extract_enrichment_used"] = True
            enriched.append(cloned)

        return enriched

    def _fetch_extract_map(self, urls: list[str], topic: str) -> dict[str, dict[str, Any]]:
        payloads: list[dict[str, Any]] = [
            {
                "urls": urls,
                "query": topic,
                "extract_depth": "advanced",
                "format": "text",
                "include_favicon": True,
            },
            {
                "urls": urls,
                "query": topic,
                "format": "text",
                "include_favicon": True,
            },
            {
                "urls": urls,
                "extract_depth": "advanced",
                "format": "text",
                "include_favicon": True,
            },
            {
                "urls": urls[:1],
            },
        ]

        for payload in payloads:
            try:
                response = self.client.call_tool(self.extract_tool_name, payload)
                if isinstance(response, dict) and response.get("isError"):
                    self._append_report_entry(
                        "extract_calls",
                        {
                            "tool": str(response.get("_used_tool") or self.extract_tool_name),
                            "payload_keys": sorted(payload.keys()),
                            "error": "error_payload",
                        },
                    )
                    continue

                mapping, trace = self._parse_extract_response(response)
                self._append_report_entry(
                    "extract_calls",
                    {
                        "tool": str(response.get("_used_tool") or self.extract_tool_name),
                        "payload_keys": sorted(payload.keys()),
                        "parsed_outputs": trace.get("parsed_outputs", []),
                        "raw_results": trace.get("raw_results", 0),
                        "normalized_results": len(mapping),
                    },
                )
                if mapping:
                    return mapping
            except Exception as exc:
                self._append_report_entry(
                    "extract_calls",
                    {
                        "tool": self.extract_tool_name,
                        "payload_keys": sorted(payload.keys()),
                        "error": type(exc).__name__,
                    },
                )
                continue

        return {}

    def _parse_extract_response(
        self,
        response: dict[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        if not isinstance(response, dict):
            return {}, {"parsed_outputs": [], "raw_results": 0}

        raw_items = self._extract_raw_results(response=response, topic="extract")
        mapping: dict[str, dict[str, Any]] = {}
        origins: set[str] = set()
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            origin = self._clean_text(item.get("_origin"))
            if origin:
                origins.add(origin)

            normalized = self._normalize_result_item(
                item=item,
                topic="extract",
                snippet_keys=self.EXTRACT_SNIPPET_KEYS,
            )
            if not normalized:
                continue

            url = str(normalized.get("url") or "").strip()
            if url:
                mapping[url] = normalized

        return mapping, {"parsed_outputs": sorted(origins), "raw_results": len(raw_items)}

    def _records_to_evidence(self, records: list[dict[str, Any]], live: bool) -> list[Evidence]:
        prefix = "mcp-live" if live else "mcp-fallback"
        evidence: list[Evidence] = []

        for item in records:
            domain = self._canonical_domain(str(item.get("domain") or "web").lower()) or "web"
            title = self._clean_text(item.get("title") or "Untitled")
            url = self._clean_text(item.get("url") or "")
            snippet = self._clean_text(item.get("snippet") or "")
            relevance = self._coerce_score(item.get("relevance"))

            evidence.append(
                Evidence(
                    source=f"{prefix}:{domain}",
                    title=title,
                    summary=snippet[:280],
                    url=url,
                    timestamp=self._now_iso(),
                    relevance=relevance,
                )
            )

        return sorted(evidence, key=self._rank_evidence, reverse=True)

    def _normalize_result_item(
        self,
        item: dict[str, Any],
        topic: str,
        fallback_snippet: str = "",
        snippet_keys: tuple[str, ...] | None = None,
    ) -> dict[str, Any] | None:
        snippet_fields = snippet_keys or self.SEARCH_SNIPPET_KEYS
        title = self._pick_first_text(item, self.TITLE_KEYS)
        url = self._pick_first_url(item)
        snippet = self._pick_first_text(item, snippet_fields)
        if not snippet and fallback_snippet:
            snippet = fallback_snippet

        domain = self._infer_domain(item=item, url=url)
        if not title:
            title = topic or self._domain_label(domain) or "Untitled"

        if not url and not snippet:
            return None

        return {
            "title": self._clean_text(title),
            "url": self._clean_text(url),
            "snippet": self._clean_text(snippet),
            "domain": domain or "web",
            "relevance": self._extract_relevance(item),
            "_origin": self._clean_text(item.get("_origin")),
        }

    def _rank_evidence(self, item: Evidence) -> float:
        source_name = self._source_name(item.source)
        source_boost = self._source_quality_boost(source_name)
        content_boost = min(len(item.summary) / 420.0, 0.3)
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

    def _topic_focus(self, topic: str) -> str:
        lowered = topic.lower()
        if "equit" in lowered:
            return "equities"
        if "credit" in lowered:
            return "credit markets"
        if "bond" in lowered or "treasury" in lowered:
            return "bond markets"

        if lowered.endswith(" outlook"):
            return self._clean_text(topic[: -len(" outlook")]) or topic

        ticker_match = re.search(r"\b[A-Z]{2,6}\b", topic)
        if ticker_match:
            return ticker_match.group(0)

        return self._clean_text(topic)

    def _looks_like_asset_topic(self, topic: str, focus: str) -> bool:
        if self._clean_text(topic).lower().endswith("outlook"):
            return True
        return bool(re.fullmatch(r"[A-Z]{2,6}", focus))

    def _infer_domain(self, item: dict[str, Any], url: str) -> str:
        url_domain = self._extract_domain(url)
        if url_domain:
            return url_domain

        for key in self.DOMAIN_KEYS:
            domain = self._domain_from_value(item.get(key))
            if domain:
                return domain

        return "web"

    def _extract_domain(self, url: str) -> str:
        normalized_url = self._normalize_url(url)
        if not normalized_url:
            return ""

        parsed = urlparse(normalized_url if "://" in normalized_url else f"https://{normalized_url}")
        domain = self._canonical_domain((parsed.netloc or "").lower().strip())
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    def _extract_relevance(self, item: dict[str, Any]) -> float:
        for key in self.SCORE_KEYS:
            number = self._coerce_float(item.get(key))
            if number is not None:
                return self._coerce_score(number)

        for key in self.RANK_KEYS:
            number = self._coerce_float(item.get(key))
            if number is not None:
                bounded_rank = max(1.0, number)
                return self._coerce_score(1.0 / bounded_rank)

        return 0.6

    def _pick_first_text(self, item: dict[str, Any], keys: tuple[str, ...]) -> str:
        for key in keys:
            if key not in item:
                continue
            text = self._flatten_text(item.get(key))
            if text:
                return text
        return ""

    def _pick_first_url(self, item: dict[str, Any]) -> str:
        for key in self.URL_KEYS:
            if key not in item:
                continue
            url = self._normalize_url(item.get(key))
            if url:
                return url

        resource = item.get("resource")
        if isinstance(resource, dict):
            for key in self.URL_KEYS:
                if key not in resource:
                    continue
                url = self._normalize_url(resource.get(key))
                if url:
                    return url

        return ""

    def _flatten_text(self, value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            return self._clean_text(value)

        if isinstance(value, (int, float, bool)):
            return self._clean_text(value)

        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                text = self._flatten_text(item)
                if not text:
                    continue
                parts.append(text)
                if len(parts) >= 3:
                    break
            return self._clean_text(" ".join(parts))

        if isinstance(value, dict):
            if value.get("type") == "text":
                return self._flatten_text(value.get("text"))

            for key in self.EXTRACT_SNIPPET_KEYS + self.TITLE_KEYS:
                if key not in value:
                    continue
                text = self._flatten_text(value.get(key))
                if text:
                    return text

        return ""

    def _normalize_url(self, value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, list):
            for item in value:
                url = self._normalize_url(item)
                if url:
                    return url
            return ""

        if isinstance(value, dict):
            for key in self.URL_KEYS:
                if key not in value:
                    continue
                url = self._normalize_url(value.get(key))
                if url:
                    return url
            return ""

        candidate = self._clean_text(value).rstrip(").,]")
        if not candidate:
            return ""

        match = re.search(r"https?://\S+", candidate)
        if match:
            candidate = match.group(0).rstrip(").,]")

        if candidate.startswith(("http://", "https://")):
            return candidate

        if "." in candidate and " " not in candidate:
            return candidate

        return ""

    def _domain_from_value(self, value: Any) -> str:
        candidate = self._clean_text(value).lower()
        if not candidate:
            return ""

        candidate = candidate.removeprefix("mcp-live:").removeprefix("mcp-fallback:")
        parsed_like_url = self._extract_domain(candidate)
        if parsed_like_url:
            return parsed_like_url

        direct_map = self.SOURCE_DOMAIN_MAP.get(candidate)
        if direct_map:
            return direct_map

        compact = candidate.replace(" ", "")
        direct_map = self.SOURCE_DOMAIN_MAP.get(compact)
        if direct_map:
            return direct_map

        stripped = candidate.removeprefix("the ").strip()
        direct_map = self.SOURCE_DOMAIN_MAP.get(stripped)
        if direct_map:
            return direct_map

        for alias, mapped_domain in self.SOURCE_DOMAIN_MAP.items():
            if alias in candidate:
                return mapped_domain

        if "." in candidate and " " not in candidate:
            return self._canonical_domain(candidate)

        return ""

    def _canonical_domain(self, domain: str) -> str:
        normalized = domain.lower().strip()
        if normalized.startswith("www."):
            normalized = normalized[4:]
        for preferred in self.PREFERRED_SOURCES:
            if normalized == preferred or normalized.endswith(f".{preferred}"):
                return preferred
        return normalized

    def _is_low_quality_domain(self, domain: str) -> bool:
        normalized = self._canonical_domain(domain)
        return any(normalized == candidate or normalized.endswith(f".{candidate}") for candidate in self.LOW_QUALITY_DOMAINS)

    def _source_quality_boost(self, domain: str) -> float:
        normalized = self._canonical_domain(domain)
        if not normalized:
            return 0.0
        if self._is_low_quality_domain(normalized):
            return -0.45
        if any(normalized == preferred or normalized.endswith(f".{preferred}") for preferred in self.PREFERRED_SOURCES):
            return 0.65
        return 0.0

    def _prune_low_quality_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = sorted(records, key=self._record_rank, reverse=True)
        stronger = [item for item in ranked if not self._is_low_quality_domain(str(item.get("domain") or ""))]
        if len(stronger) >= self.max_results_per_topic:
            return stronger
        return ranked

    def _domain_label(self, domain: str) -> str:
        normalized = self._canonical_domain(domain)
        return normalized or "Untitled"

    def _looks_like_result_item(self, item: dict[str, Any]) -> bool:
        has_url = any(self._normalize_url(item.get(key)) for key in self.URL_KEYS if key in item)
        has_title = any(self._flatten_text(item.get(key)) for key in self.TITLE_KEYS if key in item)
        has_domain = any(self._domain_from_value(item.get(key)) for key in self.DOMAIN_KEYS if key in item)
        has_snippet = any(self._flatten_text(item.get(key)) for key in self.SEARCH_SNIPPET_KEYS if key in item)
        return has_url or has_title or (has_domain and has_snippet)

    def _with_origin(self, item: dict[str, Any], origin: str) -> dict[str, Any]:
        annotated = dict(item)
        annotated.setdefault("_origin", origin)
        return annotated

    def _append_report_entry(self, bucket: str, payload: dict[str, Any]) -> None:
        entries = self.last_search_report.setdefault(bucket, [])
        if isinstance(entries, list):
            entries.append(payload)

    def _empty_search_report(self, topics: list[str] | None = None) -> dict[str, Any]:
        return {
            "topics": list(topics or []),
            "live_mcp_used": False,
            "fallback_used": False,
            "extract_enrichment_used": False,
            "search_calls": [],
            "extract_calls": [],
        }

    def _extract_response_answer(self, response: dict[str, Any]) -> str:
        if not isinstance(response, dict):
            return ""

        structured = response.get("structuredContent")
        if isinstance(structured, dict):
            answer = self._pick_first_text(structured, ("answer", "response"))
            if answer:
                return answer

        return self._pick_first_text(response, ("answer", "response"))

    def _record_rank(self, item: dict[str, Any]) -> float:
        domain = self._canonical_domain(str(item.get("domain") or "web").lower())
        source_boost = self._source_quality_boost(domain)
        snippet = self._clean_text(item.get("snippet") or "")
        content_boost = min(len(snippet) / 420.0, 0.3)
        relevance = self._coerce_score(item.get("relevance"))
        enriched_boost = 0.05 if item.get("extract_enriched") else 0.0
        return relevance + source_boost + content_boost + enriched_boost

    def _pick_extract_urls(self, records: list[dict[str, Any]], topic: str) -> list[str]:
        ranked = sorted(records, key=self._record_rank, reverse=True)
        urls: list[str] = []
        limit = min(3, max(1, self.max_results_per_topic))

        for item in ranked:
            url = self._clean_text(item.get("url") or "")
            if not url or url in urls:
                continue
            snippet = self._clean_text(item.get("snippet") or "")
            domain = self._clean_text(item.get("domain") or "").lower()
            title = self._clean_text(item.get("title") or "")
            if self._is_low_quality_domain(domain):
                continue
            if len(snippet) >= 220 and not self._is_generic_title(title, topic):
                continue
            urls.append(url)
            if len(urls) >= limit:
                return urls

        for item in ranked:
            url = self._clean_text(item.get("url") or "")
            if not url or url in urls:
                continue
            urls.append(url)
            if len(urls) >= limit:
                break

        return urls

    def _is_generic_title(self, title: str, topic: str) -> bool:
        normalized = self._clean_text(title).lower()
        if not normalized:
            return True
        if normalized in {"untitled", "extract"}:
            return True
        return normalized == self._clean_text(topic).lower()

    def _coerce_score(self, value: Any) -> float:
        try:
            num = float(value)
        except Exception:
            num = 0.6
        return min(1.0, max(0.1, num))

    def _coerce_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        return " ".join(text.split())

    def _fallback(self, topics: list[str], note: str) -> list[Evidence]:
        self.last_search_report["fallback_used"] = True
        source_cycle = [
            "mcp-fallback:reuters.com",
            "mcp-fallback:bloomberg.com",
            "mcp-fallback:federalreserve.gov",
            "mcp-fallback:imf.org",
            "mcp-fallback:wsj.com",
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
