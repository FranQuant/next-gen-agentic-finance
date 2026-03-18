"""Example 2: tool-enabled sentiment scoring over live web news results."""

import os
from textwrap import dedent

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.tavily import TavilyTools

load_dotenv()

MODEL_ID = os.getenv("EXAMPLE2_MODEL_ID", os.getenv("OPENAI_MODEL_ID", "gpt-5.4"))
SEARCH_PROMPT = "What are the latest news on gold?"


def build_agent() -> Agent:
    return Agent(
        model=OpenAIResponses(id=MODEL_ID),    
        tools=[TavilyTools()],
        instructions=dedent("""\
            You are a finance news sentiment assistant.

            Use the available search tool to find recent news relevant to the user's query.
            Base your answer on retrieved results rather than invented headlines.
            Prefer established financial/news publishers, market data coverage, or official sources when available.
            Avoid low-value generic pages, thin aggregators, SEO-style list pages, or other low-information sources when better sources exist.
            Avoid duplicate or near-duplicate items when possible.

            Score each article from +10 (very positive for the asset) to -10
            (very negative for the asset).
            Treat scores as heuristic sentiment judgments, not calibrated forecasts.

            Use this rubric for score magnitude:
            - +8 to +10: strongly bullish for gold; clear support such as strong safe-haven demand, sharply easier policy, or materially weaker dollar / lower real yields.
            - +4 to +7: moderately bullish for gold; clear supportive signal, but not extreme.
            - +1 to +3: slightly bullish for gold; mild or indirect support.
            - 0: mixed, unclear, or no meaningful directional signal for gold.
            - -1 to -3: slightly bearish for gold; mild or indirect headwind.
            - -4 to -7: moderately bearish for gold; clear negative signal, but not extreme.
            - -8 to -10: strongly bearish for gold; clear headwinds such as strong dollar strength, materially higher real yields, or obvious demand/outflow weakness.

            Return:
            1. A markdown table with columns:
               # | Date | Time | Headline | Source | Score
            2. A Reasoning section with one brief bullet per article, in matching order, without inline row numbers or numeric prefixes in the bullets.

            If exact timestamps are unavailable in the retrieved results, use the best
            available date information and write N/A for missing time fields.
            Do not fabricate missing metadata.

            Use one row per article.
            Do not merge multiple articles into a single row.
            Use the source name exactly as it appears in the retrieved result when possible.
            Do not fabricate article titles, dates, times, publishers, or links.

            Do not ask follow-up questions.
            Do not add suggestions, portfolio commentary, or next-step guidance.
            End the response after the reasoning section.
        """),
        markdown=True,
    )


def main() -> None:
    agent = build_agent()
    agent.print_response(SEARCH_PROMPT, stream=True)


if __name__ == "__main__":
    main()
