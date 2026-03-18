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
            Prefer clearly material company news over generic market or sector headlines.
            Focus on catalysts, management commentary, regulatory or legal items,
            product or company announcements, and strategic updates.
            Explicitly flag weak, noisy, tangential, or low-confidence news when it appears.
            If returned news is weak, generic, contaminated, or not sufficiently
            company-specific, say so explicitly.
            Do not elevate weak or noisy items into meaningful catalysts.

            Return:
            - recent relevant stories if available
            - a short catalyst summary only if clearly supported by the returned news
            - explicit notes on management commentary, regulatory or legal items,
              product announcements, and strategic updates when present
            - an explicit statement when news quality is weak or insufficient

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

            In the memo:
            - state the central investment debate
            - explain the key drivers
            - outline bull / base / bear scenarios
            - frame risk/reward conceptually
            - suggest only high-level trade expression ideas
            - explain what would invalidate the thesis
            - clearly separate direct observations from inference
            - clearly separate data-supported observations from interpretation

            Keep scenarios illustrative unless the inputs support more precision.
            Keep trade expression ideas conceptual, conditional, and non-prescriptive.
            Use restrained professional language over overconfident conclusions.

            Do not invent missing facts, peer comparisons, management guidance,
            catalyst specifics, or segment detail not present in the inputs.
            Do not imply that you are running a model, factor engine, or forecast
            system unless explicitly provided.
            If important data is missing, acknowledge it and keep the memo conceptual.
            Do not pretend to have precision where the dataset does not support it.
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
            1. First, delegate to the market-data-agent for a structured market,
               valuation, financial, analyst, and technical snapshot only.
            2. Second, delegate to the news-catalyst-agent for recent company-specific
               news and catalyst context only.
            3. Third, delegate to the strategy-note-agent with both prior outputs.
            4. Finally, synthesize a clean institutional-style investment memo.

            Use tool outputs and delegated agent outputs as the factual basis.
            Clearly separate structured financial observations, catalyst / news context,
            and interpretation.
            Distinguish data-supported observations from interpretation.
            Explicitly flag weak, noisy, generic, or insufficiently company-specific news.
            You may derive simple arithmetic or comparisons directly from returned values.
            Do not invent missing facts, numbers, or unsupported precision.
            Keep scenario analysis illustrative unless the inputs support more precision.
            Keep trade expression ideas conceptual and non-prescriptive.
            Explicitly flag data limitations, unclear definitions or periods, and weak,
            generic, or insufficiently company-specific news.

            The final memo should include:
            - Executive summary
            - Business / market context
            - Financial and valuation snapshot
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
