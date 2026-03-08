from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

try:
    from ..adapters import MCPMacroAdapter, MCPWebAdapter
    from ..agents import (
        MacroRegimeAgent,
        PortfolioSynthesisAgent,
        ResearchBriefAgent,
        WebIntelligenceAgent,
    )
    from ..schemas import ResearchPacket
    from .formatter import ReportFormatter
    from .storage import SQLiteStorage
except ImportError:  # pragma: no cover
    from adapters import MCPMacroAdapter, MCPWebAdapter
    from agents import (
        MacroRegimeAgent,
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
        macro_regime_agent: MacroRegimeAgent,
        portfolio_synthesis_agent: PortfolioSynthesisAgent,
        mcp_web_adapter: MCPWebAdapter,
        mcp_macro_adapter: MCPMacroAdapter,
        storage: SQLiteStorage,
        formatter: ReportFormatter,
        history_limit: int = 5,
    ) -> None:
        self.research_brief_agent = research_brief_agent
        self.web_intelligence_agent = web_intelligence_agent
        self.macro_regime_agent = macro_regime_agent
        self.portfolio_synthesis_agent = portfolio_synthesis_agent

        self.mcp_web_adapter = mcp_web_adapter
        self.mcp_macro_adapter = mcp_macro_adapter

        self.storage = storage
        self.formatter = formatter
        self.history_limit = history_limit

    def run(self, query: str) -> ResearchPacket:
        brief = self.research_brief_agent.generate(query)

        evidence = self.mcp_web_adapter.search(brief.topics)
        macro_state = self.mcp_macro_adapter.get_macro_state(brief.macro_indicators)

        web_view = self.web_intelligence_agent.analyze(brief, evidence)
        macro_view = self.macro_regime_agent.analyze(brief, macro_state)

        history = self.storage.get_recent_runs(limit=self.history_limit)

        portfolio_view = self.portfolio_synthesis_agent.synthesize(
            query=query,
            brief=brief,
            web_view=web_view,
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
            macro_state=macro_state,
            macro_view=macro_view,
            history=history,
            portfolio_view=portfolio_view,
            report="",
            metadata={
                "transport": "mcp-native",
                "capabilities": ["web", "macro"],
            },
        )

        packet.report = self.formatter.format(packet)
        self.storage.save_run(packet)

        return packet
