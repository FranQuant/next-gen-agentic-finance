"""Example 7: staged multi-agent stock research brief with optional reasoning and persistence."""

import os
from pathlib import Path
from textwrap import dedent

from dotenv import load_dotenv

load_dotenv()


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


ENABLE_REASONING = _env_flag("EXAMPLE7_ENABLE_REASONING")
SHOW_FULL_REASONING = _env_flag("EXAMPLE7_SHOW_REASONING")
ENABLE_DB = _env_flag("EXAMPLE7_ENABLE_DB")

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team import Team

if ENABLE_DB:
    from agno.db.sqlite import SqliteDb

if ENABLE_REASONING:
    from agno.tools.reasoning import ReasoningTools

from finance_tools import (
    get_analyst_recommendations,
    get_company_info,
    get_company_news_tavily,
    get_current_stock_price,
)

DEFAULT_MODEL_ID = os.getenv("EXAMPLE7_MODEL_ID", os.getenv("OPENAI_MODEL_ID", "gpt-5.4"))
STOCK_SEARCHER_MODEL_ID = os.getenv("EXAMPLE7_STOCK_SEARCHER_MODEL_ID", DEFAULT_MODEL_ID)
COMPANY_INFO_MODEL_ID = os.getenv("EXAMPLE7_COMPANY_INFO_MODEL_ID", DEFAULT_MODEL_ID)
ORCHESTRATOR_MODEL_ID = os.getenv("EXAMPLE7_ORCHESTRATOR_MODEL_ID", DEFAULT_MODEL_ID)

TEAM_DB_PATH = Path(__file__).resolve().parents[1] / "tmp" / "example7_team.db"
DEMO_PROMPT = (
    "Research NVDA and produce a concise institutional-style brief using available market "
    "data, fundamentals, and recent company news."
)


def build_team() -> Team:
    db = None
    if ENABLE_DB:
        TEAM_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        db = SqliteDb(db_file=str(TEAM_DB_PATH))

    team_tools = []
    if ENABLE_REASONING:
        team_tools.append(ReasoningTools(add_instructions=True))

    # ============================================================
    # Agent 1 — Market Data + Analyst Recommendations
    # ============================================================
    stock_searcher = Agent(
        name="stock-searcher",
        model=OpenAIResponses(id=STOCK_SEARCHER_MODEL_ID),
        role="Retrieves factual market data and analyst-positioning context only.",
        tools=[get_current_stock_price, get_analyst_recommendations],
        stream_events=True,
        instructions=dedent("""\
            You are the market-data and analyst-positioning specialist.

            Workflow:
            1. Call `get_current_stock_price` for the requested symbol.
            2. Call `get_analyst_recommendations` for the same symbol.
            3. Wait for both results before writing anything.

            Return exactly one compact JSON object and nothing else. Do not use markdown fences.

            Required schema:
            {
              "symbol": "...",
              "market_data": {
                "price": ...,
                "open": ...,
                "day_high": ...,
                "day_low": ...,
                "previous_close": ...,
                "volume": ...,
                "as_of": ...,
                "history_period": ...,
                "source": ...
              },
              "analyst_positioning": {
                "record_count": ...,
                "recent_recommendations": [
                  {
                    "period": ...,
                    "strongBuy": ...,
                    "buy": ...,
                    "hold": ...,
                    "sell": ...,
                    "strongSell": ...
                  }
                ],
                "consensus_view": "..."
              },
              "data_gaps": ["..."]
            }

            Rules:
            - Do not add any top-level keys other than `symbol`, `market_data`, `analyst_positioning`, and `data_gaps`.
            - Use only tool-derived fields. If a listed field is unavailable, set it to null or a short unavailable note.
            - Keep `recent_recommendations` to at most 5 rows and preserve the aggregate recommendation-count shape returned by the tool.
            - Do not invent firm-level analyst actions or grade changes if the tool only returns period-level counts.
            - `consensus_view` must stay factual. Only summarize a direction if it is clearly supported by the returned count distribution; otherwise write `unclear from returned records`.
            - Do not derive extra market fields such as session-change math inside the JSON packet.
            - If a tool fails, returns no data, or returns ambiguous analyst evidence, explain that in `data_gaps`.
            - Do not add business commentary, fundamentals, catalysts, valuation opinions, or narrative outside the JSON object.
        """),
        markdown=False,
    )

    # ============================================================
    # Agent 2 — Fundamentals + Tavily News
    # ============================================================
    company_info_agent = Agent(
        name="company-info-searcher",
        model=OpenAIResponses(id=COMPANY_INFO_MODEL_ID),
        role="Retrieves company fundamentals and evaluates recent company-news quality.",
        tools=[get_company_info, get_company_news_tavily],
        stream_events=True,
        instructions=dedent("""\
            You are the fundamentals and company-news specialist.

            Workflow:
            1. Call `get_company_info` first.
            2. If `company_info.longName` is available, pass it into `get_company_news_tavily` as `company_name`.
               Otherwise call the news tool with the symbol only.
            3. Wait for both results before writing anything.

            Return exactly one compact JSON object and nothing else. Do not use markdown fences.

            Required schema:
            {
              "symbol": "...",
              "business_profile": {
                "company_name": ...,
                "sector": ...,
                "industry": ...,
                "business_summary": ...
              },
              "fundamentals": {
                "market_cap": ...,
                "enterprise_value": ...,
                "shares_outstanding": ...,
                "total_revenue": ...,
                "net_income_to_common": ...,
                "trailing_eps": ...,
                "revenue_growth": ...,
                "earnings_growth": ...,
                "gross_margins": ...,
                "operating_margins": ...,
                "profit_margins": ...,
                "total_cash": ...,
                "total_debt": ...,
                "operating_cashflow": ...,
                "free_cashflow": ...,
                "current_ratio": ...,
                "quick_ratio": ...,
                "forward_pe": ...,
                "trailing_pe": ...,
                "price_to_book": ...,
                "target_high_price": ...,
                "target_low_price": ...,
                "target_mean_price": ...,
                "target_median_price": ...,
                "recommendation_mean": ...,
                "recommendation_key": ...,
                "number_of_analyst_opinions": ...,
                "dividend_rate": ...,
                "dividend_yield": ...,
                "payout_ratio": ...
              },
              "news_packet_assessment": {
                "returned_count": ...,
                "news_quality_note": ...,
                "event_diversity_note": ...,
                "query_failures": [...]
              },
              "company_specific_catalysts": [
                {
                  "title": ...,
                  "publisher": ...,
                  "date": ...,
                  "query_category": ...,
                  "relevance_bucket": ...,
                  "snippet": ...
                }
              ],
              "weak_or_low_confidence_items": [
                {
                  "title": ...,
                  "publisher": ...,
                  "date": ...,
                  "query_category": ...,
                  "relevance_bucket": ...,
                  "snippet": ...,
                  "caution": ...
                }
              ],
              "data_gaps": ["..."]
            }

            Rules:
            - `business_profile` should stay brief and factual.
            - Do not add any top-level keys other than `symbol`, `business_profile`, `fundamentals`, `news_packet_assessment`, `company_specific_catalysts`, `weak_or_low_confidence_items`, and `data_gaps`.
            - Use only fields returned by the tools. If a listed field is unavailable, set it to null or a short unavailable note.
            - Put only `high_confidence_company_specific` items into `company_specific_catalysts` when available.
            - Use `weak_or_low_confidence_items` for `broader_context` items, mixed or weak packet items, commentary-style items, or anything that should not be treated as a confirmed catalyst.
            - Keep both news arrays selective. Prefer fewer stronger items over padding.
            - If the news packet is mixed, weak, narrow, or partially failed, say so explicitly in `news_packet_assessment` and `data_gaps`.
            - Do not invent headlines, dates, publishers, event details, missing numbers, or unsupported catalysts.
            - Do not add narrative outside the JSON object.
        """),
        markdown=False,
    )

    # ============================================================
    # Coordinator — Team Orchestrator
    # ============================================================
    team_kwargs = {
        "name": "Stock Research Team",
        "model": OpenAIResponses(id=ORCHESTRATOR_MODEL_ID),
        "members": [stock_searcher, company_info_agent],
        "markdown": True,
        "show_members_responses": True,
        "instructions": dedent("""\
            You are the orchestrator for a compact two-specialist research workflow.

            Workflow:
            1. Ask `stock-searcher` for its compact JSON packet.
            2. Ask `company-info-searcher` for its compact JSON packet.
            3. Wait until both members finish.
            4. Produce the final brief using only those member outputs.

            Output rules:
            - Use the exact section titles below, in this exact order.
            - Write each section title as a markdown heading.
            - Under each section, use bullets only.
            - Use single-level bullets only. No nested bullets, numbered lists, tables, or indented sub-bullets.
            - Do not add paragraphs, tables, JSON, commentary, or closing text before the first section or after the last section.
            - Use only member outputs as the factual basis.
            - Keep market and fundamental facts separate from catalyst/news context and from interpretation.
            - Preserve caution where the member outputs describe data as missing, unclear, weak, mixed, or lower confidence.
            - Do not overclaim commentary-based, broader-context, or low-confidence news items.
            - You may make only simple arithmetic or direct comparisons that are plainly supported by member outputs.
            - Keep the brief tight and selective. Prefer fewer stronger bullets over exhaustive restatement.
            - Put limitations in `Missing Information / Data Gaps` unless the limitation directly changes the reading of a specific section.

            Required sections and guidance:
            ## Market Snapshot
            - Market-data facts only: price, session context, volume, timing, and immediate gaps.

            ## Business Overview
            - Company identity and factual business description only.

            ## Financial Profile
            - Revenue, earnings, growth, margins, cash, debt, liquidity, and cash-flow facts only.

            ## Valuation and Analyst Positioning
            - Valuation metrics, target-price fields, recommendation fields, and recent analyst-record context.
            - If analyst consensus is noisy or incomplete, say so explicitly.

            ## Catalysts and News Context
            - Lead with company-specific catalyst items.
            - Keep broader context or weak items clearly labeled as lower confidence or downweighted context.

            ## Risks
            - Provide restrained interpretation only from the member outputs.
            - Do not introduce external risks or unsupported thesis language.

            ## Missing Information / Data Gaps
            - Consolidate material missing fields, tool failures, unclear definitions, and thin coverage.

            ## Source Quality Note
            - State that market and fundamentals came from yfinance-based tools and news came from the Tavily multi-query packet.
            - Mention the news packet quality and diversity notes.
            - End the response immediately after this section.
        """),
    }

    if team_tools:
        team_kwargs["tools"] = team_tools
    if db is not None:
        team_kwargs["db"] = db

    return Team(**team_kwargs)


def main() -> None:
    team = build_team()
    team.print_response(
        DEMO_PROMPT,
        stream=True,
        show_full_reasoning=ENABLE_REASONING and SHOW_FULL_REASONING,
    )


if __name__ == "__main__":
    main()
