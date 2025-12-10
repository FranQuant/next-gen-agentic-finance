# example8.py  — Multi-Agent Hedge Fund Team using GPT-5.1

from agno.agent import Agent
from agno.team import Team
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from textwrap import dedent

# -----------------------------
# Persistent storage (optional but recommended)
# -----------------------------
db = SqliteDb(db_file="tmp/team.db")

# -----------------------------
# Agent 1: Market Data Agent
# -----------------------------
market_data_agent = Agent(
    name="Market Data Agent",
    role="Financial Data Retrieval Specialist",
    model=OpenAIResponses(id="gpt-5.1"),
    tools=[YFinanceTools()],
    instructions=dedent("""
        You are an expert market data analyst. 
        ALWAYS use tools when retrieving data. Never fabricate numbers.

        For any stock ticker, retrieve:
        - current stock price and volume
        - P/E, P/B, dividend yield, EPS (if available)
        - daily / weekly / monthly price trends
        - key technical indicators
        - 52-week high and low

        When tools exist, call them. Never answer with fabricated data.
    """),
    markdown=True,
)

# -----------------------------
# Agent 2: Quant Strategy Agent
# -----------------------------
strategy_agent = Agent(
    name="Quantitative Strategist",
    role="Trading Strategy Developer",
    model=OpenAIResponses(id="gpt-5.1"),
    instructions=dedent("""
        You create hedge-fund-grade quantitative trading strategies.

        Based on the market data provided:
        - give a buy/sell/hold recommendation
        - propose entry/exit levels
        - propose stop-loss and take-profit levels
        - suggest position sizing
        - reference signals (momentum, trend, volatility, valuation)

        If data is missing, acknowledge it and produce a conceptual strategy.
    """),
    markdown=True,
)

# -----------------------------
# TEAM COORDINATOR
# -----------------------------
hedge_fund_team = Team(
    name="AI Hedge Fund Analysis Team",
    model=OpenAIResponses(id="gpt-5.1"),
    members=[market_data_agent, strategy_agent],
    tools=[ReasoningTools(add_instructions=True)],
    instructions=dedent("""
        You orchestrate a two-agent hedge fund research workflow.

        1. First, delegate to the Market Data Agent.
        2. Then delegate to the Quantitative Strategist with full context.
        3. Finally, synthesize a clean institutional-grade investment report.

        Do NOT fabricate data. Use tool outputs directly. 
        If tools fail to provide certain metrics, state so explicitly.
    """),
    db=db,
    markdown=True,
)

# -----------------------------
# RUN THE TEAM
# -----------------------------
if __name__ == "__main__":
    hedge_fund_team.print_response(
        "Perform a comprehensive hedge-fund style analysis for MSFT.",
        stream=True,
        show_full_reasoning=True
    )
