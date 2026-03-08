from __future__ import annotations

from dotenv import load_dotenv

try:
    from .adapters import MCPMacroAdapter, MCPWebAdapter
    from .agents import (
        MacroRegimeAgent,
        PortfolioSynthesisAgent,
        ResearchBriefAgent,
        WebIntelligenceAgent,
    )
    from .config import load_config
    from .services import ReportFormatter, ResearchOrchestrator, SQLiteStorage
except ImportError:  # pragma: no cover
    from adapters import MCPMacroAdapter, MCPWebAdapter
    from agents import (
        MacroRegimeAgent,
        PortfolioSynthesisAgent,
        ResearchBriefAgent,
        WebIntelligenceAgent,
    )
    from config import load_config
    from services import ReportFormatter, ResearchOrchestrator, SQLiteStorage


load_dotenv()


def build_example11_system() -> ResearchOrchestrator:
    config = load_config()

    return ResearchOrchestrator(
        research_brief_agent=ResearchBriefAgent(
            default_tickers=config.default_tickers,
            default_macro_indicators=config.default_macro_indicators,
            max_topics=config.top_topics,
        ),
        web_intelligence_agent=WebIntelligenceAgent(),
        macro_regime_agent=MacroRegimeAgent(),
        portfolio_synthesis_agent=PortfolioSynthesisAgent(),
        mcp_web_adapter=MCPWebAdapter(
            server_url=config.mcp_web_server_url,
            timeout_sec=config.mcp_timeout_sec,
            use_live=config.use_mcp_live,
            transport=config.mcp_web_transport,
            server_command=config.mcp_web_server_command,
            server_args=list(config.mcp_web_server_args),
            tool_name=config.mcp_web_tool_name,
        ),
        mcp_macro_adapter=MCPMacroAdapter(
            server_url=config.mcp_server_url,
            timeout_sec=config.mcp_timeout_sec,
            use_live=config.use_mcp_live,
        ),
        storage=SQLiteStorage(db_path=config.db_path),
        formatter=ReportFormatter(),
        history_limit=config.history_limit,
    )
