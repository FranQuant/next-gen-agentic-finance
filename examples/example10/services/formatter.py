from __future__ import annotations

try:
    from ..schemas import ResearchPacket
except ImportError:  # pragma: no cover
    from schemas import ResearchPacket


class ReportFormatter:
    def format(self, packet: ResearchPacket) -> str:
        brief_summary_lines = [
            "RESEARCH BRIEF",
            f"- Query: {packet.brief.query}",
            f"- Objective: {packet.brief.objective}",
            f"- Universe: {', '.join(packet.brief.tickers) if packet.brief.tickers else 'N/A'}",
            f"- Topics: {', '.join(packet.brief.topics[:3]) if packet.brief.topics else 'N/A'}",
            (
                "- Macro Lens: "
                f"{', '.join(packet.brief.macro_indicators) if packet.brief.macro_indicators else 'N/A'}"
            ),
        ]

        stance_line = self._stance_delta_line(packet)
        if stance_line:
            brief_summary_lines.append(stance_line)

        status_lines = self._status_lines(packet)

        evidence_lines = [
            f"- {item.title} [{item.source}]"
            for item in packet.evidence[:5]
        ]
        if not evidence_lines:
            evidence_lines = ["- No external evidence available."]

        interpretation_lines = [
            f"- Web: {packet.web_view.get('summary', 'N/A')}",
            f"- Market: {packet.market_view.get('summary', 'N/A')}",
            f"- Macro: {packet.macro_view.get('summary', 'N/A')}",
        ]
        indicator_snapshot = packet.macro_view.get("indicator_snapshot", {})
        if isinstance(indicator_snapshot, dict) and indicator_snapshot:
            indicator_text = ", ".join(
                f"{name}={value:.2f}%"
                for name, value in indicator_snapshot.items()
                if isinstance(value, (int, float))
            )
            if indicator_text:
                interpretation_lines.append(f"- Macro Indicators: {indicator_text}")

        allocation_parts = [
            f"{ticker}:{weight:.2f}" for ticker, weight in packet.portfolio_view.allocations.items()
        ]
        portfolio_lines = [
            f"- Signal: {packet.portfolio_view.signal}",
            f"- Conviction: {packet.portfolio_view.conviction:.2f}",
            f"- Horizon: {packet.portfolio_view.horizon}",
            f"- Allocations: {', '.join(allocation_parts)}",
        ]

        risk_lines = [f"- {risk}" for risk in packet.portfolio_view.risks[:5]]
        if not risk_lines:
            risk_lines = ["- No explicit risks were produced."]

        sections = [
            *brief_summary_lines,
            "",
            "RUN STATUS",
            *status_lines,
            "",
            "EVIDENCE",
            *evidence_lines,
            "",
            "INTERPRETATION",
            *interpretation_lines,
            "",
            "PORTFOLIO IMPLICATION",
            *portfolio_lines,
            "",
            "RISKS",
            *risk_lines,
        ]

        return "\n".join(sections)

    def _stance_delta_line(self, packet: ResearchPacket) -> str:
        if not packet.history:
            return ""

        prior_signal = packet.history[0].portfolio_signal.upper()
        current_signal = packet.portfolio_view.signal.upper()
        if prior_signal == current_signal:
            return f"- Stance vs Recent History: unchanged ({current_signal})."
        return f"- Stance vs Recent History: {prior_signal} -> {current_signal}."

    def _status_lines(self, packet: ResearchPacket) -> list[str]:
        web_mode = str(packet.web_view.get("evidence_mode") or "unknown").lower()
        actionable_count = int(packet.web_view.get("actionable_evidence_count", 0) or 0)
        fallback_count = int(packet.web_view.get("fallback_evidence_count", 0) or 0)
        degraded = bool(packet.web_view.get("degraded")) or web_mode != "live"

        lines = [f"- Trust Mode: {'degraded' if degraded else 'live'}"]

        if web_mode == "fallback_only":
            lines.append("- Web Evidence: fallback-only placeholder evidence; excluded from directional scoring.")
        elif web_mode == "mixed":
            lines.append(
                "- Web Evidence: "
                f"{actionable_count} live item(s) plus {fallback_count} fallback placeholder item(s); "
                "scoring used live evidence only."
            )
        elif web_mode == "live":
            lines.append(f"- Web Evidence: {actionable_count} live item(s).")
        elif web_mode == "none":
            lines.append("- Web Evidence: no external evidence was retrieved.")
        else:
            lines.append(f"- Web Evidence: {web_mode}.")

        return lines
