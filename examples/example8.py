"""Example 8: persistent multi-agent market-data and strategy workflow with visible reasoning."""

from textwrap import dedent

from dotenv import load_dotenv

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools

load_dotenv()


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
        role="Financial data retrieval specialist",
        model=OpenAIResponses(id="gpt-5.2"),
        tools=[YFinanceTools()],
        instructions=dedent("""
            You are an expert market data analyst.

            ALWAYS use tools when retrieving data.
            Never fabricate numbers.

            For any stock ticker, retrieve as many of the following as the tool provides:
            - current stock price and volume
            - valuation multiples (P/E, P/B, dividend yield, EPS, etc.)
            - recent price trends
            - technical context
            - 52-week high and low
            - financial statement and balance-sheet data if available

            If a metric is unavailable from the tool output, state that clearly.
            Return a structured, concise summary grounded only in tool results.
        """),
        markdown=True,
    )

    # -----------------------------
    # Agent 2: Strategy Agent
    # -----------------------------
    strategy_agent = Agent(
        name="quantitative-strategist",
        role="Institutional strategy note writer",
        model=OpenAIResponses(id="gpt-5.2"),
        instructions=dedent("""
            You create institutional-style strategy notes based on provided market and
            fundamental data.

            Based on the data provided:
            - state a directional view
            - explain the key drivers
            - outline bullish / base / bearish scenarios
            - discuss factor exposures and risk/reward
            - propose possible trade expressions
            - explain what would invalidate the thesis

            Do not invent missing data.
            If important data is missing, acknowledge it and keep the strategy conceptual.
            Do not pretend to have precision where the dataset does not support it.
        """),
        markdown=True,
    )

    # -----------------------------
    # Team Coordinator
    # -----------------------------
    hedge_fund_team = Team(
        name="AI Hedge Fund Analysis Team",
        model=OpenAIResponses(id="gpt-5.2"),
        members=[market_data_agent, strategy_agent],
        tools=[ReasoningTools(add_instructions=True)],
        instructions=dedent("""
            You orchestrate a two-agent investment research workflow.

            Workflow:
            1. First, delegate to the market-data-agent.
            2. Then delegate to the quantitative-strategist with the retrieved context.
            3. Finally, synthesize a clean institutional-style investment memo.

            Use tool outputs and delegated agent outputs as the factual basis.
            You may derive simple arithmetic or comparisons directly from returned values.
            Do not invent missing facts, numbers, or unsupported precision.
            If data is unavailable, state that explicitly.

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

    return hedge_fund_team


def main() -> None:
    team = build_team()
    team.print_response(
        "Perform a comprehensive institutional-style analysis for MSFT.",
        stream=True,
        show_full_reasoning=True,
    )


if __name__ == "__main__":
    main()