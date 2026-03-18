"""Example 1: single-agent sentiment scoring over provided finance headlines."""

import os
from textwrap import dedent

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIResponses

load_dotenv()

MODEL_ID = os.getenv("EXAMPLE1_MODEL_ID", os.getenv("OPENAI_MODEL_ID", "gpt-5.4"))

HEADLINES_PROMPT = dedent("""\
    Analyze the sentiment of these sample gold-market headlines.

    Use only the headlines provided below. These are static sample headlines
    included directly in the prompt, not live retrieved news.

    1 | 2026-03-07 | 09:10 | Fed signals rate cuts could come sooner as inflation cools | Sample
    2 | 2026-03-07 | 10:05 | US dollar rallies to multi-month high on strong jobs data | Sample
    3 | 2026-03-07 | 11:20 | Geopolitical tensions flare; investors move into safe havens | Sample
    4 | 2026-03-07 | 12:30 | Gold ETF holdings fall for fifth straight week amid risk-on mood | Sample
""")


def build_agent() -> Agent:
    return Agent(
        model=OpenAIResponses(id=MODEL_ID),
        instructions=dedent("""\
            You are a finance news sentiment assistant.

            Score each provided headline from +10 (very positive for gold price)
            to -10 (very negative for gold price).
            Treat scores as heuristic sentiment judgments, not calibrated forecasts.

            Use this rubric for score magnitude:
            - +8 to +10: strongly bullish for gold; clear support such as strong safe-haven demand, sharply easier policy, or materially weaker dollar / lower real yields.
            - +4 to +7: moderately bullish for gold; clear supportive signal, but not extreme.
            - +1 to +3: slightly bullish for gold; mild or indirect support.
            - 0: mixed, unclear, or no meaningful directional signal for gold.
            - -1 to -3: slightly bearish for gold; mild or indirect headwind.
            - -4 to -7: moderately bearish for gold; clear negative signal, but not extreme.
            - -8 to -10: strongly bearish for gold; clear headwinds such as strong dollar strength, materially higher real yields, or obvious demand/outflow weakness.

            Work only from the provided headlines.
            Do not imply live retrieval, browsing, tool use, or outside data.

            Return exactly:
            1. A markdown table with columns:
               # | Date | Time | Headline | Source | Score
            2. A Reasoning section with one brief bullet per headline, in matching order, without inline row numbers or numeric prefixes in the bullets.

            Requirements:
            - Use exactly one row per headline.
            - Score must be an integer.
            - Keep each reasoning bullet brief.
            - Do not add extra sections before or after the requested output.
        """),
        markdown=True,
    )


def main() -> None:
    agent = build_agent()
    agent.print_response(HEADLINES_PROMPT, stream=True)


if __name__ == "__main__":
    main()
