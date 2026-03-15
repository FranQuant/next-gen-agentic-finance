# Example 7 — Team Orchestration using OpenAI + Finance Tools + Tavily News

from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team.team import Team

from finance_tools import (
    get_current_stock_price,
    get_analyst_recommendations,
    get_company_info,
    get_company_news_tavily,
)


def build_team() -> Team:
    # ============================================================
    # Agent 1 — Market Data + Analyst Recommendations
    # ============================================================
    stock_searcher = Agent(
        name="stock-searcher",
        model=OpenAIResponses(id="gpt-5.4"),
        role="Retrieves market data and analyst consensus available from finance tools.",
        tools=[get_current_stock_price, get_analyst_recommendations],
        instructions="""
            When the user asks for stock price or analyst data:
            1. Call the appropriate tool(s).
            2. WAIT for the tool result.
            3. Return a structured JSON summary containing only tool-derived fields.
            4. Do not invent fields the tools did not return.
            5. If data is missing, label it clearly.

            Always return structured JSON grounded in tool results.
        """,
        markdown=True,
    )

    # ============================================================
    # Agent 2 — Fundamentals + Tavily News
    # ============================================================
    company_info_agent = Agent(
        name="company-info-searcher",
        model=OpenAIResponses(id="gpt-5.4"),
        role="Retrieves company fundamentals and recent news.",
        tools=[get_company_info, get_company_news_tavily],
        instructions="""
            When the user requests company fundamentals or news:
            1. Call get_company_info first.
            2. Then call get_company_news_tavily.
            3. WAIT for tool results.
            4. Return a structured JSON object summarizing the outputs.
            5. If the news tool returns no usable stories, state that clearly.
            6. Do not invent headlines, dates, publishers, catalysts, or missing numbers.

            Always return structured JSON derived from tool outputs.
        """,
        markdown=True,
    )

    # ============================================================
    # COORDINATOR — Team Orchestrator
    # ============================================================
    team = Team(
        name="Stock Research Team",
        model=OpenAIResponses(id="gpt-5.4"),
        members=[stock_searcher, company_info_agent],
        markdown=True,
        show_members_responses=True,
        instructions="""
            You are the Orchestrator.

            Workflow:
            1. Delegate price + analyst tasks to stock-searcher.
            2. Delegate fundamentals + news tasks to company-info-searcher.
            3. WAIT until both agents return results.
            4. Combine them into an institutional-style research brief with:
               - Market snapshot
               - Business overview
               - Financial profile
               - Valuation
               - Analyst positioning
               - Catalysts
               - Risks
               - Missing information / data gaps

            Use only team-member outputs as the factual basis.
            You may derive simple arithmetic directly from those outputs.
            Do not invent missing facts, statistics, peer data, or news items.
            Explicitly label missing information.
            Do not add follow-up questions or conversational filler.
        """,
    )

    return team


def main() -> None:
    team = build_team()
    team.print_response(
        "Research NVDA. Fetch available market data, fundamentals, and recent news, then create an institutional-style research brief.",
        stream=True,
    )


if __name__ == "__main__":
    main()