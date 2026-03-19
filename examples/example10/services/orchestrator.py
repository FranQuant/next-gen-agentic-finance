from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

try:
    from ..adapters import MCPMacroAdapter, MCPWebAdapter, MarketAdapter
    from ..agents import (
        MacroRegimeAgent,
        MarketDataAgent,
        PortfolioSynthesisAgent,
        ResearchBriefAgent,
        WebIntelligenceAgent,
    )
    from ..schemas import ResearchPacket
    from .formatter import ReportFormatter
    from .storage import SQLiteStorage
except ImportError:  # pragma: no cover
    from adapters import MCPMacroAdapter, MCPWebAdapter, MarketAdapter
    from agents import (
        MacroRegimeAgent,
        MarketDataAgent,
        PortfolioSynthesisAgent,
        ResearchBriefAgent,
        WebIntelligenceAgent,
    )
    from schemas import ResearchPacket
    from services.formatter import ReportFormatter
    from services.storage import SQLiteStorage


class ResearchOrchestrator:
    def __init__(
        self,
        research_brief_agent: ResearchBriefAgent,
        web_intelligence_agent: WebIntelligenceAgent,
        market_data_agent: MarketDataAgent,
        macro_regime_agent: MacroRegimeAgent,
        portfolio_synthesis_agent: PortfolioSynthesisAgent,
        market_adapter: MarketAdapter,
        mcp_web_adapter: MCPWebAdapter,
        mcp_macro_adapter: MCPMacroAdapter,
        storage: SQLiteStorage,
        formatter: ReportFormatter,
        history_limit: int = 5,
        market_lookback_days: int = 60,
    ) -> None:
        self.research_brief_agent = research_brief_agent
        self.web_intelligence_agent = web_intelligence_agent
        self.market_data_agent = market_data_agent
        self.macro_regime_agent = macro_regime_agent
        self.portfolio_synthesis_agent = portfolio_synthesis_agent

        self.market_adapter = market_adapter
        self.mcp_web_adapter = mcp_web_adapter
        self.mcp_macro_adapter = mcp_macro_adapter

        self.storage = storage
        self.formatter = formatter
        self.history_limit = history_limit
        self.market_lookback_days = market_lookback_days

    def run(self, query: str) -> ResearchPacket:
        brief = self.research_brief_agent.generate(query)

        evidence = self.mcp_web_adapter.search(brief.topics)
        market_snapshot = self.market_adapter.get_snapshot(
            brief.tickers,
            lookback_days=self.market_lookback_days,
        )
        macro_state = self.mcp_macro_adapter.get_macro_state(brief.macro_indicators)

        web_view = self.web_intelligence_agent.analyze(brief, evidence)
        market_view = self.market_data_agent.analyze(brief, market_snapshot)
        macro_view = self.macro_regime_agent.analyze(brief, macro_state)

        history = self.storage.get_recent_runs(limit=self.history_limit)

        portfolio_view = self.portfolio_synthesis_agent.synthesize(
            query=query,
            brief=brief,
            web_view=web_view,
            market_view=market_view,
            macro_view=macro_view,
            history=history,
        )

        web_report = dict(getattr(self.mcp_web_adapter, "last_search_report", {}) or {})
        market_report = dict(getattr(self.market_adapter, "last_fetch_report", {}) or {})
        macro_report = dict(getattr(self.mcp_macro_adapter, "last_fetch_report", {}) or {})
        web_mode = str(web_view.get("evidence_mode") or web_report.get("mode") or "unknown")
        market_mode = str(market_report.get("mode") or "unknown")
        macro_mode = str(macro_report.get("mode") or "unknown")
        degraded = bool(
            web_report.get("fallback_used") or market_report.get("fallback_used") or macro_report.get("fallback_used")
        )
        neutralized = web_mode in {"fallback_only", "none"}
        if neutralized or portfolio_view.signal in {"VIEW_ONLY", "NO_ACTION"}:
            actionability = "research-only"
        elif portfolio_view.signal == "NEUTRAL" or degraded:
            actionability = "cautious-tactical"
        else:
            actionability = "directional-tactical"

        tactical_state = {
            "engine_layer": "Layer 1: Evidence/State Engine",
            "packet_kind": "tactical-view-packet",
            "actionability": actionability,
            "state_summary": {
                "web": str(web_view.get("summary") or "N/A"),
                "market": str(market_view.get("summary") or "N/A"),
                "macro": str(macro_view.get("summary") or "N/A"),
            },
            "stance_summary": {
                "signal": portfolio_view.signal,
                "conviction": round(float(portfolio_view.conviction), 2),
                "horizon": portfolio_view.horizon,
            },
            "source_trace": {
                "evidence_count": len(evidence),
                "live_web_items": int(web_view.get("live_evidence_count", 0) or 0),
                "fallback_web_items": int(web_view.get("fallback_count", 0) or 0),
                "market_ticker_count": len(market_snapshot.tickers),
                "macro_indicator_count": len(macro_state.indicators),
            },
            "state_modes": {
                "web": web_mode,
                "market": market_mode,
                "macro": macro_mode,
            },
        }

        packet = ResearchPacket(
            run_id=uuid4().hex[:12],
            created_at=datetime.now(timezone.utc).isoformat(),
            query=query,
            brief=brief,
            evidence=evidence,
            web_view=web_view,
            market_snapshot=market_snapshot,
            market_view=market_view,
            macro_state=macro_state,
            macro_view=macro_view,
            history=history,
            portfolio_view=portfolio_view,
            tactical_state=tactical_state,
            layer="layer1-evidence-state",
            packet_kind="tactical-view-packet",
            actionability=actionability,
            report="",
            metadata={
                "engine_layer": "Layer 1: Evidence/State Engine",
                "packet_kind": "tactical-view-packet",
                "actionability": actionability,
                "transport": "mcp-native",
                "capabilities": ["web", "macro", "local-market"],
                "degraded": degraded,
                "web_mode": web_mode,
                "market_mode": market_mode,
                "macro_mode": macro_mode,
                "neutralized_for_fallback": neutralized,
                "web_report": web_report,
                "market_report": market_report,
                "macro_report": macro_report,
            },
        )

        packet.report = self.formatter.format(packet)
        self.storage.save_run(packet)

        return packet
