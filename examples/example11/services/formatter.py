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
            "- Capability Layer: MCP web + MCP macro + local market snapshot",
        ]

        stance_line = self._stance_delta_line(packet)
        if stance_line:
            brief_summary_lines.append(stance_line)

        evidence_lines = [f"- {item.title} [{item.source}]" for item in packet.evidence[:5]]
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

        allocation_parts = [f"{ticker}:{weight:.2f}" for ticker, weight in packet.portfolio_view.allocations.items()]
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
