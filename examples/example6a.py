# example6a_openai.py — Maximum Pain Agent (OpenAI GPT-5.1)

from dotenv import load_dotenv
load_dotenv()   # <---- Load .env automatically, works with uv & python

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools import tool
from textwrap import dedent
import yfinance as yf


@tool(
    name="maximum_pain_level",
    description="Compute the maximum pain strike for a stock and an expiration date (YYYY-MM-DD)."
)
def maximum_pain_level(symbol: str, expiration: str):

    ticker = yf.Ticker(symbol)

    try:
        opt_chain = ticker.option_chain(expiration)
    except Exception as e:
        return f"Failed to fetch option chain for {symbol} {expiration}: {e}"

    calls = opt_chain.calls
    puts = opt_chain.puts

    if calls.empty or puts.empty:
        return f"No options data available for {symbol} on {expiration}"

    strikes = sorted(set(calls['strike']).union(puts['strike']))
    pain_values = {}

    for K in strikes:
        pain = 0

        for _, row in calls.iterrows():
            OI = row.get("openInterest", 0) or 0
            strike_call = row["strike"]
            pain += OI * max(0, K - strike_call)

        for _, row in puts.iterrows():
            OI = row.get("openInterest", 0) or 0
            strike_put = row["strike"]
            pain += OI * max(0, strike_put - K)

        pain_values[K] = pain

    max_pain_strike = min(pain_values, key=pain_values.get)
    return f"Max pain strike for {symbol} on {expiration}: {max_pain_strike}"


# -------------------------------------------------------
# Agent powered by GPT-5.1
# -------------------------------------------------------
agent = Agent(
    name="Max Pain Analyst",
    model=OpenAIResponses(id="gpt-5.1"),

    tools=[maximum_pain_level],
    tool_choice="auto",

    instructions=dedent("""
        You are an options market analyst specializing in the maximum pain concept.
        When the user asks a question, ALWAYS call the maximum_pain_level tool.
        Do not produce explanatory text before the tool call.
    """),

    markdown=True,
)


if __name__ == "__main__":
    agent.print_response(
        "What is the maximum pain for AAPL options expiring on 2025-12-12?",
        stream=True
    )
