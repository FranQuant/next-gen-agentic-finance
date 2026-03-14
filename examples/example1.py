"""Example 1: single-agent sentiment scoring over provided finance headlines."""

from textwrap import dedent
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIResponses

load_dotenv()

def build_agent() -> Agent:
    return Agent(
        model=OpenAIResponses(id="gpt-5.4"),
        instructions=dedent("""\
            You are a finance news sentiment assistant.

            Score the provided headlines from +10 (very positive for gold price)
            to -10 (very negative for gold price).
            Treat these scores as heuristic sentiment judgments, not calibrated forecasts.

            Work only with the headlines given in the prompt.
            Do not claim you retrieved live news unless a tool is provided.

            Return:
            1. A markdown table with columns:
               Date | Time | News | Source | Score
            2. A short reasoning section for each score.
        """),
        markdown=True,
    )


def main() -> None:
    agent = build_agent()
    agent.print_response(
        dedent("""\
            Analyze the sentiment of these sample gold-market headlines:

            2026-03-07 | 09:10 | Fed signals rate cuts could come sooner as inflation cools | Sample
            2026-03-07 | 10:05 | US dollar rallies to multi-month high on strong jobs data | Sample
            2026-03-07 | 11:20 | Geopolitical tensions flare; investors move into safe havens | Sample
            2026-03-07 | 12:30 | Gold ETF holdings fall for fifth straight week amid risk-on mood | Sample
        """),
        stream=True,
    )


if __name__ == "__main__":
    main()