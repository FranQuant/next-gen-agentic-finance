"""Example 3: debugging a tool-enabled finance news sentiment agent."""

from textwrap import dedent

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.tavily import TavilyTools

load_dotenv()


def build_agent() -> Agent:
    return Agent(
        model=OpenAIResponses(id="gpt-5.2"),
        tools=[TavilyTools()],
        instructions=dedent("""\
            You are a finance news sentiment assistant.

            Use the available search tool to find recent news relevant to the user's query.
            Base your answer on retrieved results rather than invented headlines.

            Return:
            1. A markdown table with columns:
               Date | Time | News | Source | Score
            2. A short reasoning section explaining each score

            Score sentiment from +10 (very positive for the asset)
            to -10 (very negative for the asset).

            If exact timestamps are unavailable in the retrieved results, use the best
            available date information and write N/A for missing time fields.
            Do not fabricate missing metadata.

            Use one row per retrieved news item.
            Do not merge multiple articles into a single row.
            Use the source name exactly as it appears in the retrieved result when possible.

            Prefer recent and relevant results. Avoid stale or weakly related items when possible.

            Do not ask follow-up questions.
            Do not add suggestions, portfolio commentary, or next-step guidance.
            End the response after the reasoning section.
        """),
        markdown=True,
        debug_mode=True,
    )


def main() -> None:
    agent = build_agent()
    agent.print_response("What are the latest news on GLD?", stream=True)


if __name__ == "__main__":
    main()
    