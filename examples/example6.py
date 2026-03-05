# Example 6 - Maximum Pain Calculation Agent (Improved Version)
from agno.agent import Agent
from agno.models.lmstudio import LMStudio
from agno.tools import tool
from textwrap import dedent
import yfinance as yf
import pandas as pd
import numpy as np


# -------------------------------------------
# Custom Tool: Maximum Pain Calculation
# -------------------------------------------
def compute_max_pain_from_chain(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    """
    Pure deterministic max pain computation.
    No IO. No time dependence.
    """
    calls = calls[['strike', 'openInterest']].rename(
        columns={'openInterest': 'call_oi'}
    )
    puts = puts[['strike', 'openInterest']].rename(
        columns={'openInterest': 'put_oi'}
    )

    all_data = pd.merge(calls, puts, on='strike', how='outer').fillna(0)

    strikes = all_data['strike'].values
    call_oi = all_data['call_oi'].values
    put_oi = all_data['put_oi'].values

    candidate_prices = np.unique(strikes)

    payouts = []

    for S in candidate_prices:
        call_payout = call_oi * np.maximum(S - strikes, 0)
        put_payout = put_oi * np.maximum(strikes - S, 0)
        total_payout = call_payout.sum() + put_payout.sum()
        payouts.append(total_payout)

    min_index = np.argmin(payouts)
    return float(candidate_prices[min_index])


@tool
def maximum_pain_level(symbol: str, expiration: str) -> str:
    """
    Compute the *maximum pain* strike for the given symbol & expiration.

    Maximum Pain Definition:
    The strike at which option buyers (both calls and puts) experience
    the greatest total loss at expiration — often used to understand
    dealer hedging pressure or potential price magnet effects.
    """
    ticker = yf.Ticker(symbol)

    try:
        opt_chain = ticker.option_chain(expiration)
        calls = opt_chain.calls
        puts = opt_chain.puts

        # Basic validation
        if calls.empty or puts.empty:
            return f"No options data available for {symbol} on {expiration}."

        max_pain_strike = compute_max_pain_from_chain(calls, puts)

        return f"{symbol} max pain for {expiration}: {max_pain_strike}"

    except Exception as e:
        return f"Error calculating max pain: {str(e)}"


# -------------------------------------------
# Agent Definition
# -------------------------------------------
agent = Agent(
    model=LMStudio(
        id="meta-llama-3.1-8b-instruct",
        cache_response=True,
        cache_ttl=3600,
    ),
    tools=[maximum_pain_level],
    tool_choice="auto",
    markdown=True,
    instructions=dedent("""
        You are an Options Microstructure Assistant specialized in
        *maximum pain* analytics.

        When the user asks about maximum pain:

        1. ALWAYS call the `maximum_pain_level` tool.
        2. After the tool executes, ALWAYS read its return value.
        3. Your final answer MUST include the tool output in this format:

            **Maximum Pain Result:** <tool_output>

        4. Then explain, in 2–3 bullet points:
            - What maximum pain represents,
            - Why this strike may matter for expiry dynamics,
            - Any microstructure intuition (OI pressure, dealer gamma).

        5. NEVER return generic filler text like:
           “This is the output of the tool.”

        6. ALWAYS embed the raw strike returned by the tool.
    """),
)


# -------------------------------------------
# Run the Agent
# -------------------------------------------
agent.print_response("What is the maximum pain for AAPL options expiring on 2026-03-06?")
