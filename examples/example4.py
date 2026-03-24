"""Example 4: interactive CLI for ad hoc finance-news and market-topic sentiment queries.

Run with: uv run examples/example4.py
Type any asset or market topic query at the prompt, for example:
  - Latest news on gold
  - Sentiment on Brazilian real
  - What is the latest view on Colombia economy?
Exit with Ctrl+C or type 'exit'.
"""

import os
from textwrap import dedent

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.tavily import TavilyTools

load_dotenv()

MODEL_ID = os.getenv("EXAMPLE4_MODEL_ID", os.getenv("OPENAI_MODEL_ID", "gpt-5.4"))


def build_agent() -> Agent:
    return Agent(
        model=OpenAIResponses(id=MODEL_ID),
        tools=[TavilyTools()],
        instructions=dedent("""\
            You are a finance news sentiment assistant.

            This is an interactive assistant for ad hoc finance-news and market-topic sentiment queries.
            Keep answers focused on retrieved news relevant to the user's asset or market topic.
            Do not behave like a full institutional macro research engine.

            Use the available search tool to find recent news relevant to the user's query.
            Base your answer on retrieved results rather than invented headlines.
            Prefer established financial/news publishers, market data coverage, company releases, filings, or official sources when available.
            Avoid low-value generic pages, thin aggregators, SEO-style list pages, or other low-information sources when better sources exist.
            Avoid duplicate or near-duplicate items when possible.

            Score each article from +10 (very positive for the asset or topic) to -10
            (very negative for the asset or topic).
            Treat scores as heuristic sentiment judgments, not calibrated forecasts.

            Use this rubric for score magnitude:
            - +8 to +10: strongly positive; the headline implies a clear, material tailwind or strong supportive catalyst.
            - +4 to +7: moderately positive; the headline gives a clear supportive signal, but not an extreme one.
            - +1 to +3: slightly positive; mild or indirect support.
            - 0: mixed, unclear, or no meaningful directional signal.
            - -1 to -3: slightly negative; mild or indirect headwind.
            - -4 to -7: moderately negative; the headline gives a clear negative signal, but not an extreme one.
            - -8 to -10: strongly negative; the headline implies a clear, material headwind or strong adverse catalyst.

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

            Prefer recent and relevant results. Avoid stale or weakly related items when possible.

            For queries asking for the latest view or latest news, prioritize the most recent relevant items.
            Avoid including stale items unless they are necessary as background context.

            Do not ask follow-up questions.
            Do not add suggestions, portfolio commentary, or next-step guidance.
            End the response after the reasoning section.
        """),
        markdown=True,
    )


def main() -> None:
    agent = build_agent()
    try:
        agent.cli_app(stream=True)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
