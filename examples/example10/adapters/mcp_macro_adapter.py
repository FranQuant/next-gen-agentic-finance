from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from ..schemas import MacroState
except ImportError:  # pragma: no cover
    from schemas import MacroState


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
        if not self.use_live:
            raise RuntimeError("MCP live mode disabled.")

        if self.transport in {"stdio", "command"} and not self.server_command:
            raise RuntimeError("MCP stdio transport requires EXAMPLE10_MCP_MACRO_SERVER_COMMAND.")
        if self.transport in {"streamable_http", "http", "sse"} and not self.server_url:
            raise RuntimeError("MCP URL transport requires EXAMPLE10_MCP_MACRO_SERVER_URL.")

        return asyncio.run(self._call_tool_async(tool_name=tool_name, payload=payload))

    async def _call_tool_async(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if self.transport in {"stdio", "command"}:
            server = StdioServerParameters(
                command=self.server_command or "",
                args=self.server_args,
                env=dict(os.environ),
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
            "macro.get_state",
            "macro",
            "fred",
            "econom",
            "state",
        )
        for keyword in candidates:
            for name in tool_names:
                if keyword in name.lower():
                    return name

        return tool_names[0] if tool_names else preferred


class MCPMacroAdapter:
    FALLBACK_VALUES = {
        "inflation": 2.8,
        "unemployment": 4.1,
        "policy_rate": 3.6,
        "10y_yield": 4.1,
        "gdp_growth": 2.0,
    }

    VALUE_KEYS = (
        "value",
        "latest",
        "observation",
        "result",
    )

    def __init__(
        self,
        server_url: str | None = None,
        timeout_sec: int = 8,
        use_live: bool = False,
        transport: str = "stdio",
        server_command: str | None = None,
        server_args: list[str] | None = None,
        tool_name: str = "macro.get_state",
    ) -> None:
        self.client = _MCPClient(
            server_url=server_url,
            timeout_sec=timeout_sec,
            use_live=use_live,
            transport=transport,
            server_command=server_command,
            server_args=server_args,
        )
        self.tool_name = tool_name
        self.last_fetch_report = self._empty_fetch_report()

    def get_macro_state(self, indicators: list[str]) -> MacroState:
        normalized_indicators = self._normalize_indicators(indicators)
        self.last_fetch_report = self._empty_fetch_report(normalized_indicators)
        payload = {
            "indicators": normalized_indicators,
            "normalize": True,
        }

        try:
            response = self.client.call_tool(self.tool_name, payload)
            if isinstance(response, dict) and response.get("isError"):
                raise RuntimeError("MCP macro tool returned an error payload.")

            state, trace = self._parse_state(response, normalized_indicators)
            self._append_report_entry(
                {
                    "tool": str(response.get("_used_tool") or self.tool_name) if isinstance(response, dict) else self.tool_name,
                    "payload_keys": sorted(payload.keys()),
                    "available_tools": list(response.get("_available_tools", [])) if isinstance(response, dict) else [],
                    "parsed_output": trace.get("parsed_output", ""),
                    "missing_indicators": trace.get("missing_indicators", []),
                }
            )
            if state:
                self.last_fetch_report["live_macro_used"] = True
                self.last_fetch_report["mode"] = "mcp-live"
                return state
        except Exception as exc:
            self._append_report_entry(
                {
                    "tool": self.tool_name,
                    "payload_keys": sorted(payload.keys()),
                    "error": type(exc).__name__,
                }
            )

        fallback = self._fallback_state(normalized_indicators)
        self.last_fetch_report["fallback_used"] = True
        self.last_fetch_report["mode"] = "mcp-fallback"
        return fallback

    def _parse_state(
        self,
        response: dict[str, Any],
        indicators: list[str],
    ) -> tuple[MacroState | None, dict[str, Any]]:
        payload, parsed_output = self._extract_payload(response)
        if not isinstance(payload, dict):
            return None, {"parsed_output": parsed_output, "missing_indicators": indicators}

        raw_values = payload.get("indicators")
        if not isinstance(raw_values, dict):
            raw_values = {indicator: payload.get(indicator) for indicator in indicators if indicator in payload}

        values: dict[str, float] = {}
        missing: list[str] = []
        for indicator in indicators:
            numeric_value = self._coerce_indicator_value(raw_values.get(indicator))
            if numeric_value is None:
                fallback = self.FALLBACK_VALUES.get(indicator)
                if fallback is None:
                    missing.append(indicator)
                    continue
                values[indicator] = round(float(fallback), 3)
                missing.append(indicator)
                continue
            values[indicator] = round(float(numeric_value), 3)

        if not values:
            return None, {"parsed_output": parsed_output, "missing_indicators": indicators}

        regime = str(payload.get("regime") or self._classify_regime(values))
        notes = self._normalize_notes(payload.get("notes"))
        notes = self._ensure_source_note(notes, str(payload.get("source") or "mcp-live"))
        if missing:
            notes.append(f"missing indicators filled with fallback: {', '.join(missing)}")

        return (
            MacroState(
                as_of=self._clean_text(payload.get("as_of")) or self._now_iso(),
                indicators=values,
                regime=regime,
                notes=notes,
            ),
            {
                "parsed_output": parsed_output,
                "missing_indicators": missing,
            },
        )

    def _extract_payload(self, response: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        if not isinstance(response, dict):
            return None, ""

        structured = response.get("structuredContent")
        if isinstance(structured, dict):
            return structured, "structuredContent"

        for key in ("result", "data", "payload"):
            nested = response.get(key)
            if isinstance(nested, dict) and (
                isinstance(nested.get("indicators"), dict) or any(item in nested for item in self.FALLBACK_VALUES)
            ):
                return nested, key

        if isinstance(response.get("indicators"), dict):
            return response, "response"

        payload, origin = self._parse_content_payload(response.get("content"))
        if payload is not None:
            return payload, origin

        return None, ""

    def _parse_content_payload(self, content: Any) -> tuple[dict[str, Any] | None, str]:
        if not isinstance(content, list):
            return None, ""

        for block in content:
            if hasattr(block, "model_dump"):
                block_data = block.model_dump()
            elif isinstance(block, dict):
                block_data = block
            else:
                block_data = {}

            text = self._flatten_text(block_data.get("text"))
            parsed = self._parse_json_object(text)
            if parsed:
                return parsed, "content.text"

            resource = block_data.get("resource")
            if isinstance(resource, dict):
                resource_text = self._flatten_text(resource.get("text") or resource.get("contents"))
                parsed = self._parse_json_object(resource_text)
                if parsed:
                    return parsed, "content.resource"

        return None, ""

    def _parse_json_object(self, text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        if not cleaned:
            return None

        if cleaned.startswith("```"):
            lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        if not cleaned:
            return None

        try:
            payload = json.loads(cleaned)
        except Exception:
            return None

        if isinstance(payload, dict):
            return payload

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and (
                    isinstance(item.get("indicators"), dict) or any(key in item for key in self.FALLBACK_VALUES)
                ):
                    return item

        return None

    def _coerce_indicator_value(self, value: Any) -> float | None:
        if value is None:
            return None

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

        if isinstance(value, str):
            candidate = value.strip().lower().replace(",", "")
            if not candidate or candidate in {"na", "n/a", "none", "null"}:
                return None
            if candidate.endswith("%"):
                candidate = candidate[:-1].strip()
            if candidate.endswith("bps"):
                base = candidate[:-3].strip()
                try:
                    return float(base) / 100.0
                except ValueError:
                    return None
            try:
                return float(candidate)
            except ValueError:
                return None

        if isinstance(value, dict):
            for key in self.VALUE_KEYS:
                if key in value:
                    numeric_value = self._coerce_indicator_value(value.get(key))
                    if numeric_value is not None:
                        return numeric_value
            data = value.get("data")
            if isinstance(data, list):
                return self._coerce_indicator_value(data)
            return None

        if isinstance(value, list):
            for item in reversed(value):
                numeric_value = self._coerce_indicator_value(item)
                if numeric_value is not None:
                    return numeric_value

        return None

    def _fallback_state(self, indicators: list[str]) -> MacroState:
        values = {indicator: self.FALLBACK_VALUES.get(indicator, 0.0) for indicator in indicators}
        notes = [
            "source: mcp-fallback",
            "MCP macro adapter fallback values used.",
            "Inflation assumed to be CPI YoY percent.",
        ]
        return MacroState(
            as_of=self._now_iso(),
            indicators=values,
            regime=self._classify_regime(values),
            notes=notes,
        )

    def _normalize_indicators(self, indicators: list[str]) -> list[str]:
        requested = indicators or ["inflation", "unemployment", "policy_rate"]
        deduped: list[str] = []
        for indicator in requested:
            normalized = self._clean_text(indicator).lower()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return deduped or ["inflation", "unemployment", "policy_rate"]

    def _normalize_notes(self, notes: Any) -> list[str]:
        if not isinstance(notes, list):
            return []
        return [self._clean_text(item) for item in notes if self._clean_text(item)]

    def _ensure_source_note(self, notes: list[str], source: str) -> list[str]:
        normalized_source = self._clean_text(source) or "mcp-live"
        normalized_notes = list(notes)
        if not any(note.lower().startswith("source:") for note in normalized_notes):
            normalized_notes.insert(0, f"source: {normalized_source}")
        return normalized_notes

    def _flatten_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return self._clean_text(value)
        if isinstance(value, (int, float, bool)):
            return self._clean_text(value)
        if isinstance(value, list):
            parts = [self._flatten_text(item) for item in value]
            return self._clean_text(" ".join(part for part in parts if part))
        if isinstance(value, dict):
            for key in ("text", "content", "value"):
                if key in value:
                    text = self._flatten_text(value.get(key))
                    if text:
                        return text
        return ""

    def _empty_fetch_report(self, indicators: list[str] | None = None) -> dict[str, Any]:
        return {
            "requested_indicators": list(indicators or []),
            "live_macro_used": False,
            "fallback_used": False,
            "mode": "uninitialized",
            "tool_calls": [],
        }

    def _append_report_entry(self, payload: dict[str, Any]) -> None:
        entries = self.last_fetch_report.setdefault("tool_calls", [])
        if isinstance(entries, list):
            entries.append(payload)

    def _classify_regime(self, values: dict[str, float]) -> str:
        inflation = values.get("inflation", 2.8)
        unemployment = values.get("unemployment", 4.1)
        policy_rate = values.get("policy_rate", 4.5)

        if unemployment >= 5.5:
            return "recession risk"
        if inflation >= 3.0 and policy_rate >= 4.0:
            return "late-cycle tightening"
        if inflation <= 2.3 and policy_rate <= 3.5 and unemployment <= 4.8:
            return "easing expansion"
        return "mid-cycle mixed"

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).strip().split())

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
