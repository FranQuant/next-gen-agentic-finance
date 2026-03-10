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
            report="",
            metadata={
                "transport": "mcp-native",
                "capabilities": ["web", "macro", "local-market"],
            },
        )

        packet.report = self.formatter.format(packet)
        self.storage.save_run(packet)

        return packet
