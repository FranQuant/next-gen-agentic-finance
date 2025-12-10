# Example 7 — Team Orchestration using GPT-5.1 + Custom Finance Tools

from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.team.team import Team

from finance_tools import (
    get_current_stock_price,
    get_analyst_recommendations,
    get_company_info,
    get_company_news,
)

# ============================================================
# Agent 1 — Market Data + Analyst Recommendations
# ============================================================
stock_searcher = Agent(
    name="stock-searcher",
    model=OpenAIResponses(id="gpt-5.1"),
    role="Fetches real-time stock price and analyst recommendations.",
    tools=[get_current_stock_price, get_analyst_recommendations],
    instructions="""
        When the user asks for stock price or analyst data:
        1. Call the appropriate tool(s).
        2. WAIT for the tool result.
        3. Return a JSON summary containing the tool outputs.

        Do NOT answer manually when a tool exists.
        ALWAYS return structured JSON with real data.
    """,
    markdown=True,
)

# ============================================================
# Agent 2 — Fundamentals + Company News
# ============================================================
company_info_agent = Agent(
    name="company-info-searcher",
    model=OpenAIResponses(id="gpt-5.1"),
    role="Retrieves company fundamentals and recent news.",
    tools=[get_company_info, get_company_news],
    instructions="""
        When the user requests company fundamentals or news:
        1. Call get_company_info first.
        2. Then call get_company_news.
        3. WAIT for tool results.
        4. Return a JSON object summarizing the outputs.

        Never produce placeholder JSON or empty responses.
        Always return structured JSON derived from tools.
    """,
    markdown=True,
)

# ============================================================
# COORDINATOR — Team Orchestrator (GPT-5.1)
# ============================================================
team = Team(
    name="Stock Research Team",
    model=OpenAIResponses(id="gpt-5.1"),
    members=[stock_searcher, company_info_agent],
    markdown=True,

    # Allowed in your version of Agno:
    show_members_responses=True,

    instructions="""
        You are the Orchestrator.

        Workflow:
        1. Delegate price + analyst tasks to stock-searcher.
        2. Delegate fundamentals + news tasks to company-info-searcher.
        3. WAIT until BOTH agents return JSON results.
        4. Combine results into a full hedge-fund-grade investment report:
            - Market snapshot
            - Business overview
            - Financial trends
            - Valuation
            - Peer comparison
            - Catalysts
            - Risks
            - Positioning ideas

        NEVER guess numbers.
        Use ONLY data returned by your team members.
    """,
)

# ============================================================
# RUN
# ============================================================
team.print_response(
    "Research NVDA. Fetch full data and create a hedge-fund-style report.",
    stream=True,
)
