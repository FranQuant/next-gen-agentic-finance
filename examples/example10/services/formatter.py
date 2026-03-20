from __future__ import annotations

try:
    from ..schemas import ResearchPacket
except ImportError:  # pragma: no cover
    from schemas import ResearchPacket


class ReportFormatter:
    def format(self, packet: ResearchPacket) -> str:
        header_lines = [
            "LAYER 1 TACTICAL VIEW PACKET",
            f"- Query: {packet.brief.query}",
            f"- Objective: {packet.brief.objective}",
            f"- Universe: {', '.join(packet.brief.tickers) if packet.brief.tickers else 'N/A'}",
            f"- Topics: {', '.join(packet.brief.topics[:3]) if packet.brief.topics else 'N/A'}",
            (
                "- Macro Lens: "
                f"{', '.join(packet.brief.macro_indicators) if packet.brief.macro_indicators else 'N/A'}"
            ),
            f"- Packet Kind: {packet.packet_kind}",
            f"- Actionability: {packet.actionability}",
            "- Role: reusable evidence/state handoff for downstream allocator or PM review.",
        ]

        stance_line = self._stance_delta_line(packet)
        if stance_line:
            header_lines.append(stance_line)

        status_lines = self._status_lines(packet)
        source_trace_lines = self._source_trace_lines(packet)
        state_lines = self._state_snapshot_lines(packet)
        stance_lines = self._tactical_stance_lines(packet)

        risk_lines = [f"- {risk}" for risk in packet.tactical_view.risks[:4]]
        if not risk_lines:
            risk_lines = ["- No explicit risks were produced."]

        sections = [
            *header_lines,
            "",
            "RUN STATUS",
            *status_lines,
            "",
            "SOURCE TRACE",
            *source_trace_lines,
            "",
            "TACTICAL STATE",
            *state_lines,
            "",
            "TACTICAL STANCE",
            *stance_lines,
            "",
            "HANDOFF RISKS",
            *risk_lines,
        ]

        return "\n".join(sections)

    def _stance_delta_line(self, packet: ResearchPacket) -> str:
        if not packet.history:
            return ""

        prior_signal = packet.history[0].stance_signal.upper()
        current_signal = packet.tactical_view.signal.upper()
        if prior_signal == current_signal:
            return f"- Stance vs Recent History: unchanged ({current_signal})."
        return f"- Stance vs Recent History: {prior_signal} -> {current_signal}."

    def _status_lines(self, packet: ResearchPacket) -> list[str]:
        metadata = packet.metadata or {}
        web_view = getattr(packet, "web_view", {}) or {}
        web_mode = str(metadata.get("web_mode") or web_view.get("evidence_mode") or "unknown")
        market_mode = str(metadata.get("market_mode") or "unknown")
        macro_mode = str(metadata.get("macro_mode") or "unknown")
        degraded = bool(metadata.get("degraded"))
        neutralized = bool(metadata.get("neutralized_for_fallback"))

        lines = [f"- Trust Mode: {'degraded' if degraded else 'live'}"]

        if web_mode == "fallback_only":
            lines.append("- Web Evidence: offline placeholder notes only; not sourced research.")
        elif web_mode == "mixed":
            lines.append("- Web Evidence: live MCP results plus fallback placeholders; placeholders were ignored.")
        elif web_mode == "live":
            lines.append("- Web Evidence: live MCP results.")
        elif web_mode == "none":
            lines.append("- Web Evidence: no usable web evidence was retrieved.")
        else:
            lines.append(f"- Web Evidence: {web_mode}.")

        if market_mode == "market-placeholder":
            lines.append("- Market Data: deterministic placeholder values were used for all requested tickers.")
        elif market_mode == "market-partial-fallback":
            lines.append("- Market Data: live fetch was incomplete; placeholder values were used for some tickers.")
        elif market_mode == "market-live":
            lines.append("- Market Data: live yfinance snapshot.")
        elif market_mode == "market-empty":
            lines.append("- Market Data: no tickers were requested.")
        elif market_mode != "unknown":
            lines.append(f"- Market Data: {market_mode}.")

        if macro_mode == "mcp-fallback":
            lines.append("- Macro Data: fallback baseline values were used.")
        elif macro_mode == "mcp-partial-fallback":
            lines.append("- Macro Data: live MCP macro state was partial; fallback values were used for missing indicators.")
        elif macro_mode == "mcp-live":
            lines.append("- Macro Data: live MCP macro state.")
        elif macro_mode != "unknown":
            lines.append(f"- Macro Data: {macro_mode}.")

        if neutralized:
            lines.append("- Actionability: signal held neutral until live sourced web evidence is available.")

        return lines

    def _source_trace_lines(self, packet: ResearchPacket) -> list[str]:
        tactical_state = packet.tactical_state or {}
        source_trace = tactical_state.get("source_trace", {}) if isinstance(tactical_state, dict) else {}
        lines = [
            (
                "- Coverage: "
                f"evidence={source_trace.get('evidence_count', len(packet.evidence))}, "
                f"live_web={source_trace.get('live_web_items', packet.web_view.get('live_evidence_count', 0))}, "
                f"macro_indicators={source_trace.get('macro_indicator_count', len(packet.macro_state.indicators))}, "
                f"market_tickers={source_trace.get('market_ticker_count', len(packet.market_snapshot.tickers))}"
            )
        ]

        unique_sources = []
        for item in packet.evidence:
            source = item.source.strip()
            if source and source not in unique_sources:
                unique_sources.append(source)
        if unique_sources:
            lines.append(f"- Source Mix: {', '.join(unique_sources[:4])}")

        trace_items = packet.evidence[:4]
        if trace_items:
            lines.extend(f"- {item.title} [{item.source}]" for item in trace_items)
        else:
            lines.append("- No external evidence available.")

        return lines

    def _state_snapshot_lines(self, packet: ResearchPacket) -> list[str]:
        lines = [
            f"- Web State: {packet.web_view.get('summary', 'N/A')}",
            f"- Market State: {packet.market_view.get('summary', 'N/A')}",
            f"- Macro State: {packet.macro_view.get('summary', 'N/A')}",
        ]

        key_points = packet.web_view.get("key_points", [])
        if isinstance(key_points, list) and key_points:
            lines.append(f"- Web Key Points: {' | '.join(str(point) for point in key_points[:2])}")

        indicator_snapshot = packet.macro_view.get("indicator_snapshot", {})
        if isinstance(indicator_snapshot, dict) and indicator_snapshot:
            ordered_keys = (
                "headline_inflation_yoy_pct",
                "core_inflation_yoy_pct",
                "unemployment_pct",
                "payrolls_yoy_pct",
                "policy_rate_pct",
                "curve_slope_pct",
                "credit_spread_pct",
            )
            parts = []
            for name in ordered_keys:
                value = indicator_snapshot.get(name)
                if isinstance(value, (int, float)):
                    parts.append(f"{name}={value:.2f}")
            if parts:
                lines.append(f"- Macro Snapshot: {', '.join(parts)}")

        return lines

    def _tactical_stance_lines(self, packet: ResearchPacket) -> list[str]:
        lines = [
            f"- Signal: {packet.tactical_view.signal}",
            f"- Conviction: {packet.tactical_view.conviction:.2f}",
            f"- Horizon: {packet.tactical_view.horizon}",
            f"- Actionability: {packet.actionability}",
            "- Handoff Use: tactical stance packet for downstream allocator or PM review, not final portfolio construction.",
        ]
        if packet.tactical_view.preferred_exposures:
            lines.append(f"- Preferred Exposures: {', '.join(packet.tactical_view.preferred_exposures[:4])}")
        if packet.tactical_view.avoid_exposures:
            lines.append(f"- Avoid Exposures: {', '.join(packet.tactical_view.avoid_exposures[:4])}")
        stance_basis = packet.tactical_view.stance_basis[:2]
        if stance_basis:
            lines.append(f"- Stance Basis: {' | '.join(stance_basis)}")
        return lines
