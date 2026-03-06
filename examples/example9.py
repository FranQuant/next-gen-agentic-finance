# example9.py — Deterministic Multi-Agent Quant Research System

from dotenv import load_dotenv
import os

load_dotenv()

from agno.agent import Agent
from agno.team import Team
from agno.tools.reasoning import ReasoningTools
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from textwrap import dedent

# local financial tools
from finance_tools import (
    get_current_stock_price,
    get_analyst_recommendations,
    get_company_info,
    get_company_news
)

# ------------------------------------------------
# Persistent storage
# ------------------------------------------------

db = SqliteDb(db_file="tmp/research_team.db")


# ------------------------------------------------
# Agent 1 — Market Data Agent
# ------------------------------------------------

market_data_agent = Agent(
    name="Market Data Agent",
    role="Financial Market Data Retrieval Specialist",

    model=OpenAIResponses(id="gpt-5.1"),

    tools=[
        get_current_stock_price,
        get_company_info,
        get_company_news,
        get_analyst_recommendations
    ],

    instructions=dedent("""
You are a financial data analyst responsible for producing
a structured market snapshot for a given ticker.

Always retrieve data using the available tools.

The snapshot should include:

• current price
• valuation metrics (P/E, P/B, dividend yield, EPS if available)
• business description
• recent news headlines
• analyst recommendation trends

Return the output as a structured dataset suitable for
quantitative research analysis.

Never fabricate numbers.
If data is unavailable, explicitly say so.
"""),

    markdown=True,
)


# ------------------------------------------------
# Agent 2 — Quant Strategy Agent
# ------------------------------------------------

strategy_agent = Agent(
    name="Quantitative Strategist",
    role="Hedge Fund Strategy Analyst",

    model=OpenAIResponses(id="gpt-5.1"),

    instructions=dedent("""
You are a quantitative strategist at a hedge fund.

Using the structured market snapshot provided by the
Market Data Agent, construct an investment thesis.

Your output should include:

• Investment thesis (1–2 lines)
• 3–5 supporting arguments
• Scenario framework:
  - bull case
  - base case
  - bear case
• Key catalysts (6–24 month horizon)
• Major risks
• Suggested positioning (entry zone, horizon, sizing considerations)

Do not fabricate financial data.
Base reasoning only on the provided snapshot.
"""),

    markdown=True,
)


# ------------------------------------------------
# Team Coordinator
# ------------------------------------------------

research_team = Team(
    name="AI Hedge Fund Research Team",

    model=OpenAIResponses(id="gpt-5.1"),

    members=[
        market_data_agent,
        strategy_agent
    ],

    tools=[
        ReasoningTools(add_instructions=True)
    ],

    instructions=dedent("""
You orchestrate a deterministic hedge-fund research workflow.

Workflow:

1. Ask the Market Data Agent to produce a structured market snapshot.

   The snapshot must include:
   - current price
   - valuation metrics
   - business description
   - analyst sentiment
   - relevant recent news

2. Capture the output from the Market Data Agent.

3. Pass the FULL snapshot explicitly to the Quantitative Strategist.

4. The strategist must produce an investment thesis based strictly
   on the snapshot.

5. Finally synthesize BOTH outputs into a clean
   hedge-fund-style research memo.

Rules:

• Never fabricate missing data
• Preserve tool outputs exactly
• Clearly separate DATA vs INTERPRETATION in the final report
"""),

    db=db,

    markdown=True,
)


# ------------------------------------------------
# Run the system
# ------------------------------------------------

if __name__ == "__main__":

    research_team.print_response(
        "Perform a hedge-fund style research memo for MSFT.",
        stream=True,
        show_full_reasoning=True
    )