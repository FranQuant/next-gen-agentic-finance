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
    model = OpenAIResponses(id="gpt-5.2")

    market_data_agent = Agent(
        name="market-data-agent",
        role="Financial market data retrieval specialist",
        model=model,
        tools=[
            get_current_stock_price,
            get_company_info,
            get_company_news_tavily,
            get_analyst_recommendations,
        ],
        instructions=dedent("""
            You are a market data specialist.

            Use the available tools and return ONLY a compact market snapshot.

            Required output format:

            PRICE
            - Last Price: ...
            - Market Cap: ...
            - Forward P/E: ...
            - Revenue Growth: ...
            - Earnings Growth: ...

            SENTIMENT
            - Analyst Rating: ...
            - Consensus Target Price: ...

            NEWS
            - ...
            - ...
            - ...

            Rules:
            - Never fabricate numbers
            - If unavailable, write "Unavailable"
            - Keep the whole output under 12 lines
            - No extra commentary
            - Use company info and Tavily news if available
        """),
        markdown=True,
    )

    quant_strategist = Agent(
        name="quant-strategist",
        role="Institutional strategy interpreter",
        model=model,
        instructions=dedent("""
            You are a quantitative strategist.

            You will receive a market snapshot from the market-data-agent.
            Use ONLY that snapshot.

            Return ONLY:

            THESIS
            - ...
            - ...
            - ...

            RISKS
            - ...
            - ...
            - ...

            HORIZON
            - ... months

            Rules:
            - Do not introduce external data
            - Do not fabricate numbers
            - Keep the whole output under 10 lines
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

            You will receive the strategist output.
            Convert it into a compact portfolio action.

            Return ONLY:

            SIGNAL
            - LONG / SHORT / NEUTRAL

            CONVICTION
            - Low / Medium / High

            WEIGHT
            - Small / Medium / Large

            HORIZON
            - ... months

            RATIONALE
            - ...
            - ...

            Rules:
            - Base the action only on the strategist output
            - Do not fabricate precise percentages, stops, option costs, or target levels
            - Keep the whole output under 8 lines
            - No extra commentary
        """),
        markdown=True,
    )

    research_team = Team(
        name="AI Hedge Fund Research Team",
        model=model,
        members=[
            market_data_agent,
            quant_strategist,
            portfolio_manager,
        ],
        instructions=dedent("""
            You orchestrate a STRICT and COMPACT workflow.

            Workflow:
            1. Ask market-data-agent for a compact market snapshot.
            2. Pass the FULL snapshot explicitly to quant-strategist.
            3. Pass the FULL strategist output explicitly to portfolio-manager.
            4. Produce the final memo.

            Final output must be EXACTLY:

            DATA
            <market snapshot>

            INTERPRETATION
            <strategist output>

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