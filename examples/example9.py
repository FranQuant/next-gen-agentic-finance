"""Example 9: compact multi-agent research-to-portfolio workflow."""

from textwrap import dedent

from dotenv import load_dotenv

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.team import Team

from finance_tools import (
    get_current_stock_price,
    get_analyst_recommendations,
    get_company_info,
    get_company_news_tavily,
)

load_dotenv()


def build_team() -> Team:
    db = SqliteDb(db_file="tmp/research_team.db")
    model = OpenAIResponses(id="gpt-5.4")

    market_data_agent = Agent(
        name="market-data-agent",
        role="Compact factual snapshot specialist",
        model=model,
        tools=[
            get_current_stock_price,
            get_company_info,
            get_company_news_tavily,
            get_analyst_recommendations,
        ],
        instructions=dedent("""
            You are a market data specialist.

            Use the available tools and return ONLY a compact factual snapshot.

            Required output format:

            - Price / Market Cap: ...
            - 52-Week Range / Recent Performance: ...
            - Forward / Trailing Valuation: ...
            - Growth / Margins: ...
            - Cash / Debt / Buybacks / Dividend: ...
            - Analyst Sentiment: ...
            - Company-Specific News / Guidance / Segment Mix: ...

            Rules:
            - Never fabricate numbers
            - If unavailable, weak, or unclear, label it clearly instead of improvising
            - Keep the whole output under 12 lines
            - Prioritize price, market cap, 52-week range, recent performance, valuation, growth, margins, cash, debt, buybacks, dividend, and analyst sentiment
            - Include only a very short note on recent company-specific news if clearly supported
            - Do not produce thesis language, recommendation language, or broad risk interpretation
        """),
        markdown=True,
    )

    strategy_interpreter = Agent(
        name="strategy-interpreter",
        role="Compact institutional strategy interpreter",
        model=model,
        instructions=dedent("""
            You are a strategy interpreter.

            You will receive a market snapshot from the market-data-agent.
            Use ONLY that snapshot.

            Return ONLY:

            - Thesis: ...
            - Positioning: ...
            - Catalysts: ...
            - Risks: ...
            - Key Metrics: ...

            Rules:
            - Keep the interpretation compact and institutional in tone
            - Do not introduce external data
            - Do not fabricate numbers
            - No fake precision
            - No unnecessary expansion
            - Keep the whole output under 8 lines
            - No extra commentary
        """),
        markdown=True,
    )

    portfolio_manager = Agent(
        name="portfolio-manager",
        role="Portfolio manager",
        model=model,
        instructions=dedent("""
            You are a portfolio manager.

            You will receive the strategy-interpreter output.
            Convert it into a compact portfolio action.

            Return ONLY:

            - Signal: LONG / SHORT / NEUTRAL
            - Conviction: Low / Medium / High
            - Weight: Small / Medium / Large
            - Horizon: ...
            - Action: ...
            - Risk Management: ...
            - What Changes the View: ...

            Rules:
            - Base the action only on the strategy-interpreter output
            - Keep the output qualitative and bounded
            - Do not fabricate precise percentages, stops, option costs, target levels, or pseudo-risk-engine language
            - Keep the whole output under 8 lines
            - No extra commentary
        """),
        markdown=True,
    )

    research_team = Team(
        name="Research-to-Portfolio Workflow Team",
        model=model,
        members=[
            market_data_agent,
            strategy_interpreter,
            portfolio_manager,
        ],
        instructions=dedent("""
            You orchestrate a compact staged research-to-portfolio workflow.

            Workflow:
            1. Ask market-data-agent for a compact factual snapshot.
            2. Pass the FULL snapshot explicitly to strategy-interpreter.
            3. Pass the FULL strategy-interpreter output explicitly to portfolio-manager.
            4. Produce the final output.

            Final output must be EXACTLY:

            DATA
            <market snapshot>

            INTERPRETATION
            <strategy-interpreter output>

            PORTFOLIO ACTION
            <portfolio manager output>

            Rules:
            - Maximum 30 lines total
            - Do not ask follow-up questions
            - Do not create extra sections
            - Do not expand into a long memo
            - Do not request spreadsheets, deadlines, attachments, or further deliverables
            - End immediately after PORTFOLIO ACTION
        """),
        db=db,
        markdown=True,
    )

    return research_team


def main() -> None:
    research_team = build_team()
    research_team.print_response(
        "Analyze MSFT and produce a compact investment memo.",
        stream=True,
        show_full_reasoning=True,
    )


if __name__ == "__main__":
    main()
