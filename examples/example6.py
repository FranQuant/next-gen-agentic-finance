"""Example 6: maximum pain analysis agent for listed options expirations."""

import os
from textwrap import dedent

from dotenv import load_dotenv

import numpy as np
import pandas as pd
import yfinance as yf

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools import tool

load_dotenv()

MODEL_ID = os.getenv("EXAMPLE6_MODEL_ID", os.getenv("OPENAI_MODEL_ID", "gpt-5.4"))


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
    symbol = symbol.upper().strip()
    ticker = yf.Ticker(symbol)

    try:
        expirations = list(ticker.options or [])
    except Exception as e:
        return f"Failed to fetch listed expirations for {symbol}: {e}"

    if expiration not in expirations:
        available = ", ".join(expirations[:5]) if expirations else "none available"
        return (
            f"Expiration {expiration} is not listed for {symbol}. "
            f"Some available expirations: {available}."
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
        model=OpenAIResponses(id=MODEL_ID),
        tools=[maximum_pain_level],
        tool_choice="auto",
        markdown=True,
        instructions=dedent("""\
            You are an options microstructure assistant specialized in maximum pain analysis.

            For any maximum pain request, always call the maximum_pain_level tool.
            Treat the tool output as the source of truth.
            Do not invent unavailable expirations or maximum pain results.

            If the tool returns a valid result, respond in this format:

            **Maximum Pain Result:** <tool_output>

            Maximum pain is the strike where aggregate option-holder payout at expiry is minimized based on open interest.
            It is a heuristic microstructure indicator, not a forecast.

            If the tool reports an invalid expiration, unavailable data, or a fetch failure, state the issue clearly using the tool output.
            Do not add filler text.
        """),
    )


def main() -> None:
    agent = build_agent()
    agent.print_response(
        "What is the maximum pain for EWZ options expiring on 2026-06-30?",
        stream=True,
    )


if __name__ == "__main__":
    main()

