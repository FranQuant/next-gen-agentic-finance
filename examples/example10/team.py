from __future__ import annotations

from dotenv import load_dotenv

try:
    from .adapters import FREDAdapter, MarketAdapter, PolymarketAdapter, TavilyAdapter
    from .agents import (
        MacroRegimeAgent,
        MarketDataAgent,
        PortfolioSynthesisAgent,
        ResearchBriefAgent,
        WebIntelligenceAgent,
    )
    from .config import load_config
    from .services import ReportFormatter, ResearchOrchestrator, SQLiteStorage
except ImportError:  # pragma: no cover
    from adapters import FREDAdapter, MarketAdapter, PolymarketAdapter, TavilyAdapter
    from agents import (
        MacroRegimeAgent,
        MarketDataAgent,
        PortfolioSynthesisAgent,
        ResearchBriefAgent,
        WebIntelligenceAgent,
    )
    from config import load_config
    from services import ReportFormatter, ResearchOrchestrator, SQLiteStorage


load_dotenv()


def build_example10_system() -> ResearchOrchestrator:
    config = load_config()

    return ResearchOrchestrator(
        research_brief_agent=ResearchBriefAgent(
            default_tickers=config.default_tickers,
            default_macro_indicators=config.default_macro_indicators,
            max_topics=config.top_topics,
        ),
        web_intelligence_agent=WebIntelligenceAgent(),
        market_data_agent=MarketDataAgent(),
        macro_regime_agent=MacroRegimeAgent(),
        portfolio_synthesis_agent=PortfolioSynthesisAgent(),
        tavily_adapter=TavilyAdapter(api_key=config.tavily_api_key),
        market_adapter=MarketAdapter(),
        fred_adapter=FREDAdapter(api_key=config.fred_api_key),
        polymarket_adapter=PolymarketAdapter(),
        storage=SQLiteStorage(db_path=config.db_path),
        formatter=ReportFormatter(),
        history_limit=config.history_limit,
        market_lookback_days=config.market_lookback_days,
    )
