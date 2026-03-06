# example9.py — Compact Deterministic Multi-Agent Research Demo

from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.team import Team
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from textwrap import dedent

# local financial tools
from finance_tools import (
    get_current_stock_price,
    get_analyst_recommendations,
    get_company_info,
    get_company_news,
)

# ------------------------------------------------
# Persistent storage
# ------------------------------------------------

db = SqliteDb(db_file="tmp/research_team.db")

MODEL = OpenAIResponses(id="gpt-5-mini")
# swap to gpt-5.1 later if you want higher quality:
# MODEL = OpenAIResponses(id="gpt-5.1")


# ------------------------------------------------
# Agent 1 — Market Data Agent
# ------------------------------------------------

market_data_agent = Agent(
    name="market-data-agent",
    role="Financial Market Data Retrieval Specialist",
    model=MODEL,
    tools=[
        get_current_stock_price,
        get_company_info,
        get_company_news,
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
"""),
    markdown=True,
)


# ------------------------------------------------
# Agent 2 — Quant Strategist
# ------------------------------------------------

quant_strategist = Agent(
    name="quant-strategist",
    role="Hedge Fund Strategy Analyst",
    model=MODEL,
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


# ------------------------------------------------
# Agent 3 — Portfolio Manager
# ------------------------------------------------

portfolio_manager = Agent(
    name="portfolio-manager",
    role="Portfolio Manager",
    model=MODEL,
    instructions=dedent("""
You are a portfolio manager.

You will receive the strategist output.
Convert it into a compact portfolio action.

Return ONLY:

SIGNAL
- LONG / SHORT / NEUTRAL

CONVICTION
- 0.xx

WEIGHT
- x%

HORIZON
- ... months

RATIONALE
- ...
- ...

Rules:
- Keep the whole output under 8 lines
- No extra commentary
"""),
    markdown=True,
)


# ------------------------------------------------
# Team Coordinator
# ------------------------------------------------

research_team = Team(
    name="AI Hedge Fund Research Team",
    model=MODEL,
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


# ------------------------------------------------
# Run the system
# ------------------------------------------------

if __name__ == "__main__":
    research_team.print_response(
        "Analyze MSFT and produce an investment memo.",
        stream=True,
        show_full_reasoning=True,
    )