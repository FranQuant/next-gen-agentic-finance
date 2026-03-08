"""Example 6: maximum pain analysis agent for listed options expirations."""

from textwrap import dedent

from dotenv import load_dotenv

import numpy as np
import pandas as pd
import yfinance as yf

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools import tool

load_dotenv()


def compute_max_pain_from_chain(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    """Deterministic maximum-pain calculation from calls and puts open interest."""
    calls = calls[["strike", "openInterest"]].copy()
    puts = puts[["strike", "openInterest"]].copy()

    calls["openInterest"] = calls["openInterest"].fillna(0)
    puts["openInterest"] = puts["openInterest"].fillna(0)

    all_strikes = np.array(sorted(set(calls["strike"]).union(set(puts["strike"]))), dtype=float)

    total_payouts = []
    for settlement_price in all_strikes:
        call_payout = (calls["openInterest"] * np.maximum(settlement_price - calls["strike"], 0)).sum()
        put_payout = (puts["openInterest"] * np.maximum(puts["strike"] - settlement_price, 0)).sum()
        total_payouts.append(call_payout + put_payout)

    min_idx = int(np.argmin(total_payouts))
    return float(all_strikes[min_idx])


@tool(
    name="maximum_pain_level",
    description="Compute the maximum pain strike for a stock and a listed expiration date (YYYY-MM-DD).",
)
def maximum_pain_level(symbol: str, expiration: str) -> str:
    ticker = yf.Ticker(symbol)

    try:
        expirations = list(ticker.options or [])
    except Exception as e:
        return f"Failed to fetch listed expirations for {symbol}: {e}"

    if expiration not in expirations:
        nearest = ", ".join(expirations[:5]) if expirations else "none available"
        return (
            f"Expiration {expiration} is not listed for {symbol}. "
            f"Nearest available expirations: {nearest}."
        )

    try:
        chain = ticker.option_chain(expiration)
        calls = chain.calls
        puts = chain.puts
    except Exception as e:
        return f"Failed to fetch option chain for {symbol} {expiration}: {e}"

    if calls.empty or puts.empty:
        return f"No options data available for {symbol} on {expiration}."

    max_pain_strike = compute_max_pain_from_chain(calls, puts)

    return f"Maximum pain strike for {symbol} on {expiration}: {max_pain_strike}"


def build_agent() -> Agent:
    return Agent(
        model=OpenAIResponses(id="gpt-5.2"),
        tools=[maximum_pain_level],
        tool_choice="auto",
        markdown=True,
        instructions=dedent("""\
            You are an options microstructure assistant specialized in maximum pain analysis.

            When the user asks about maximum pain:
            1. Always call the maximum_pain_level tool.
            2. Read the tool output carefully.
            3. If the tool returns a valid result, respond in this format:

               **Maximum Pain Result:** <tool_output>

               - What maximum pain represents
               - Why it may matter into expiry
               - One short microstructure intuition

            4. If the tool reports an invalid or unavailable expiration, do not invent a result.
               State the issue clearly and preserve the listed available expirations from the tool output.
            5. Do not add filler text.
        """),
    )


def main() -> None:
    agent = build_agent()
    agent.print_response(
        "What is the maximum pain for AAPL options expiring on 2026-03-27?",
        stream=True,
    )


if __name__ == "__main__":
    main()
    