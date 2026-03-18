"""Example 8: persistent multi-agent investment memo workflow with visible reasoning."""

import os
from textwrap import dedent

from dotenv import load_dotenv

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools

from finance_tools import get_company_news_tavily

load_dotenv()

DEFAULT_MODEL_ID = os.getenv("EXAMPLE8_MODEL_ID", os.getenv("OPENAI_MODEL_ID", "gpt-5.4"))
MARKET_DATA_MODEL_ID = os.getenv("EXAMPLE8_MARKET_DATA_MODEL_ID", DEFAULT_MODEL_ID)
NEWS_CATALYST_MODEL_ID = os.getenv("EXAMPLE8_NEWS_CATALYST_MODEL_ID", DEFAULT_MODEL_ID)
STRATEGY_NOTE_MODEL_ID = os.getenv("EXAMPLE8_STRATEGY_NOTE_MODEL_ID", DEFAULT_MODEL_ID)
TEAM_MODEL_ID = os.getenv("EXAMPLE8_TEAM_MODEL_ID", DEFAULT_MODEL_ID)


def build_team() -> Team:
    # -----------------------------
    # Persistent storage
    # -----------------------------
    db = SqliteDb(db_file="tmp/team.db")

    # -----------------------------
    # Agent 1: Market Data Agent
    # -----------------------------
    market_data_agent = Agent(
        name="market-data-agent",
        role="Structured market and financial snapshot specialist",
        model=OpenAIResponses(id=MARKET_DATA_MODEL_ID),
        tools=[YFinanceTools()],
        instructions=dedent("""
            You are an expert market data analyst.

            ALWAYS use tools when retrieving data.
            Never fabricate numbers.
            Use only fields explicitly returned by the tools.
            If a field definition, time period, or calculation basis is unclear from the
            tool output, label it as unclear.
            Do not provide business commentary, catalysts, thesis language, or recommendations.
            Keep data-supported observations separate from interpretation.
            Do not infer a thesis from the returned data.

            Focus only on:
            - price snapshot
            - valuation metrics
            - financial statement fields
            - balance sheet and cash-flow fields
            - analyst consensus fields
            - technical context if explicitly available

            Return a compact structured note with exactly these sections:
            - Snapshot
            - Financial Profile
            - Valuation / Analyst / Technical
            - Data Gaps

            Keep each section short.
            Prefer short bullets over prose paragraphs.
            Do not repeat the same fact across sections.
            If a metric is unavailable from the tool output, state that clearly.
            Return a structured, factual summary grounded only in tool results.
        """),
        markdown=True,
    )

    # -----------------------------
    # Agent 2: News / Catalyst Agent
    # -----------------------------
    news_catalyst_agent = Agent(
        name="news-catalyst-agent",
        role="Company news and catalyst retrieval specialist",
        model=OpenAIResponses(id=NEWS_CATALYST_MODEL_ID),
        tools=[get_company_news_tavily],
        instructions=dedent("""
            You retrieve recent company-specific news and catalyst context only.

            ALWAYS use tools when retrieving news.
            Never fabricate stories or catalysts.
            Use the custom company-news retrieval tool for recent company-specific news.
            Summarize the returned Tavily packet, not just raw story text.
            Use the packet fields explicitly:
            - `relevance_bucket`: prioritize `high_confidence_company_specific`
            - `query_category`: preserve regulatory/legal, product/strategy, management/commentary, and broader company context structure
            - `news_quality_note`: report whether the set is strong, mixed, or weak
            - `event_diversity_note`: report whether coverage is broad or narrow

            Prefer clearly material company news over generic market or sector headlines.
            Focus on catalysts, management commentary, regulatory or legal items,
            product or company announcements, and strategic updates.
            Explicitly flag weak, noisy, tangential, or low-confidence news when it appears.
            If returned news is weak, generic, contaminated, or not sufficiently
            company-specific, say so explicitly.
            Do not elevate weak or noisy items into meaningful catalysts.
            Do not elevate broader context into a company catalyst.
            Avoid overstating catalysts when category coverage is thin.
            Keep the output selective and compact.
            Prefer fewer stronger items over padded summaries.

            Return exactly these sections:
            - News Packet Assessment
            - Regulatory / Legal
            - Product / Strategy
            - Management / Commentary
            - Broader Company Context
            - Weak / Generic / Noisy Items To Downweight
            - Catalyst Takeaways

            Rules:
            - `News Packet Assessment` must explicitly mention `news_quality_note` and `event_diversity_note`.
            - Use `query_category` to place items into the correct section.
            - Use `high_confidence_company_specific` items first.
            - Put `broader_context` items only in `Broader Company Context`.
            - Keep `Regulatory / Legal`, `Product / Strategy`, and `Management / Commentary` to at most 2 bullets each.
            - Keep `Broader Company Context` to at most 1 bullet.
            - Keep `Weak / Generic / Noisy Items To Downweight` to at most 2 bullets.
            - Keep Catalyst Takeaways to at most 3 bullets and include only items clearly supported by retrieved news.
            - If category coverage is narrow or absent, say so briefly in `News Packet Assessment` or `Catalyst Takeaways`.
            - If a section has no usable items, write "None."
            - Include publisher and date when available.
            - Keep bullets short and factual.

            Do not discuss valuation, broader strategy, or investment recommendations.
            Keep the summary concise, factual, and grounded only in returned tool results.
        """),
        markdown=True,
    )

    # -----------------------------
    # Agent 3: Strategy Note Agent
    # -----------------------------
    strategy_agent = Agent(
        name="strategy-note-agent",
        role="Institutional strategy memo writer",
        model=OpenAIResponses(id=STRATEGY_NOTE_MODEL_ID),
        instructions=dedent("""
            You write an institutional-style strategy memo from the provided context only.
            Use only the prior agent outputs as evidence.

            In the memo:
            - state the central investment debate
            - explain the key drivers that matter now
            - preserve distinct catalyst buckets when news supports them
            - outline bull / base / bear scenarios
            - frame risk/reward conceptually
            - suggest only high-level trade expression ideas
            - explain what would invalidate the thesis
            - clearly separate direct observations from inference

            Keep scenarios illustrative unless the inputs support more precision.
            Keep trade expression ideas conceptual, conditional, and non-prescriptive.
            Use restrained professional language over overconfident conclusions.
            Keep the memo materially shorter than a full sell-side note.
            Prefer dense bullets over long paragraphs.
            Avoid repeating the same fact across sections.
            Keep data-supported observations, catalyst/news context, and interpretation distinct,
            but do this through structure and wording rather than repetitive labels.
            Use the news summary's category structure when present.
            If regulatory/legal, product/strategy, or management/commentary buckets are absent,
            say coverage is limited rather than inferring missing catalysts.
            If the news summary says quality is mixed or diversity is narrow, reflect that briefly.
            Keep weak or speculative news clearly labeled and out of the core thesis.

            Do not invent missing facts, peer comparisons, management guidance,
            catalyst specifics, or segment detail not present in the inputs.
            Do not imply that you are running a model, factor engine, or forecast
            system unless explicitly provided.
            If important data is missing, acknowledge it and keep the memo conceptual.
            Do not pretend to have precision where the dataset does not support it.

            Memo formatting rules:
            - The memo body must be bullet-only output.
            - Each section header must be followed immediately by bullets only.
            - Do not use paragraph blocks anywhere in the memo body.
            - Do not add commentary before the first section or after Data Limitations.
            - Do not use sub-bullets, numbered lists, or tables in the memo body.
            - Each bullet should be one or two sentences max.
            - Keep each fact in the single most relevant section; do not repeat it in adjacent sections.
            - If a section has no strong content, use one short bullet stating the limitation rather than spilling content from another section.

            Section guidance and caps:
            - Executive Summary: exactly 2 bullets; make them decision-useful by stating the setup, what matters now, and the main gating issue
            - Business / Market Context: 1 to 2 bullets; provide only incremental background and do not restate the executive summary
            - Financial and Valuation Snapshot: 1 to 4 bullets
            - Catalyst / News Context: 1 to 4 bullets; keep regulatory/legal, product/strategy, and management/commentary distinct when present, and flag weak/speculative items explicitly
            - Scenario Analysis: exactly 3 bullets, one each for Bull, Base, and Bear, labeled inline
            - Risk/Reward Framing: 1 to 2 bullets; make upside and downside drivers specific rather than generic
            - Trade Expression Ideas: 1 to 2 bullets; use institutional, conditional phrasing such as bias, monitoring triggers, or preferred expression, without sizing or prescriptive instructions
            - Key Risks: 1 to 4 bullets
            - Data Limitations: 1 to 4 bullets
        """),
        markdown=True,
    )

    # -----------------------------
    # Team Coordinator
    # -----------------------------
    investment_memo_team = Team(
        name="Investment Memo Workflow Team",
        model=OpenAIResponses(id=TEAM_MODEL_ID),
        members=[market_data_agent, news_catalyst_agent, strategy_agent],
        tools=[ReasoningTools(add_instructions=True)],
        instructions=dedent("""
            You orchestrate a three-agent investment memo workflow.

            Workflow:
            1. First, delegate to the market-data-agent for a compact structured market,
               valuation, financial, analyst, and technical snapshot only.
            2. Second, delegate to the news-catalyst-agent for a compact structured
               recent company-specific news and catalyst note only.
            3. Third, delegate to the strategy-note-agent with only the compact outputs
               from the first two agents.
            4. Finally, synthesize a clean institutional-style investment memo.

            Follow this sequence strictly.
            Do not parallelize steps 1 and 2.

            Use tool outputs and delegated agent outputs as the factual basis.
            Clearly separate structured financial observations, catalyst / news context,
            and interpretation.
            Distinguish data-supported observations from interpretation.
            Explicitly flag weak, noisy, generic, or insufficiently company-specific news.
            Treat regulatory/legal, product/strategy, and management/commentary as
            distinct catalyst buckets when present in the news summary.
            Use the news summary's `news_quality_note` and `event_diversity_note`
            to indicate when catalyst coverage is strong but narrow, mixed, or limited.
            If a catalyst bucket is absent, say so briefly instead of inferring one.
            You may derive simple arithmetic or comparisons directly from returned values.
            Do not invent missing facts, numbers, or unsupported precision.
            Keep scenario analysis illustrative unless the inputs support more precision.
            Keep trade expression ideas conceptual and non-prescriptive.
            Keep delegated payloads compact and avoid forwarding bloated narrative text.
            Keep the final memo materially shorter and reduce repetition across sections.
            Do not restate the executive summary in the business / market context section.
            Keep the distinction between facts, catalyst context, and interpretation
            clear through structure and wording rather than repetitive labels.
            Make risk/reward framing specific about upside and downside drivers.
            Keep trade expression ideas institutional and conditional, not placeholder-like.
            Explicitly flag data limitations, unclear definitions or periods, and weak,
            generic, or insufficiently company-specific news.
            The final memo body must be bullet-only output under the required section headers.
            Each section header must be followed immediately by bullets only.
            Do not allow paragraph blocks, section spillover, or repeated facts across adjacent sections.
            Enforce these caps strictly:
            - Executive summary: exactly 2 bullets
            - Business / market context: max 2 bullets
            - Financial and valuation snapshot: max 4 bullets
            - Catalyst / news context: max 4 bullets
            - Scenario analysis: exactly 3 bullets, one each for Bull, Base, and Bear
            - Risk/reward framing: max 2 bullets
            - Trade expression ideas: max 2 bullets
            - Key risks: max 4 bullets
            - Data limitations: max 4 bullets

            The final memo should include:
            - Executive summary
            - Business / market context
            - Financial and valuation snapshot
            - Catalyst / news context
            - Scenario analysis
            - Risk/reward framing
            - Trade expression ideas
            - Key risks
            - Data limitations

            End the response after the Data Limitations section.
            Do not add follow-up offers, suggestions, or conversational closing lines.
        """),
        db=db,
        markdown=True,
    )

    return investment_memo_team


def main() -> None:
    team = build_team()
    team.print_response(
        "Perform a comprehensive institutional-style analysis for AAPL.",
        stream=True,
        show_full_reasoning=True,
    )


if __name__ == "__main__":
    main()
