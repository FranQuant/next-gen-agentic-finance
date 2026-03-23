"""Example 9: conservative Agno + Tavily remote MCP issuer-intelligence demo."""

import argparse
import asyncio
import os
import sys
from textwrap import dedent

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIResponses

load_dotenv()

DEFAULT_MODEL_ID = os.getenv("EXAMPLE9_MODEL_ID", os.getenv("OPENAI_MODEL_ID", "gpt-5.4"))
DEFAULT_ISSUER = "Nu Holdings (Nubank)"
DEFAULT_FOCUS = (
    "recent business developments, regulatory or credit-relevant issues, "
    "strategic initiatives, and near-term catalysts"
)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Example 9 issuer-intelligence demo.")
    parser.add_argument(
        "--issuer",
        default=DEFAULT_ISSUER,
        help="Issuer name for the brief.",
    )
    parser.add_argument(
        "--focus",
        default=DEFAULT_FOCUS,
        help="Research focus for the brief.",
    )
    parser.add_argument(
        "--prompt",
        help="Optional full prompt override. If provided, it replaces the default issuer/focus query.",
    )
    return parser.parse_args()


def build_default_query(issuer: str, focus: str) -> str:
    return (
        f"Build a concise issuer intelligence brief on {issuer}. "
        f"Focus on {focus}. "
        "Use public web sources and return a structured summary."
    )


def build_agent(mcp_tools: object, openai_api_key: str) -> Agent:
    return Agent(
        name="issuer-intelligence-demo",
        model=OpenAIResponses(id=DEFAULT_MODEL_ID, api_key=openai_api_key),
        tools=[mcp_tools],
        instructions=dedent("""\
            You are a conservative issuer-intelligence analyst working only from public web sources.

            Use the available Tavily MCP tools to gather recent issuer-specific information.
            Start with search-oriented retrieval and use page extraction only when a clearly relevant source needs verification beyond the search snippet.
            Prefer stronger public sources when available, including company disclosures, regulatory sources,
            official statements, established business press, and clearly attributable reporting.
            Do not overstate weak commentary, duplicated claims, stale items, or unclear evidence.
            Separate confirmed developments from thin or ambiguous evidence.
            Keep the final output compact.

            Return exactly these sections, in this order, and do not add any others:

            QUERY
            RELEVANT SOURCES
            EXTRACTED DEVELOPMENTS
            SOURCE QUALITY / WEAK ITEMS
            OPEN QUESTIONS

            Section rules:
            - `QUERY`: restate the issuer and the requested research focus in 1 to 2 bullets.
            - `RELEVANT SOURCES`: list the most relevant public sources used, with source name, date if available, and why each source matters.
            - `EXTRACTED DEVELOPMENTS`: summarize confirmed recent business developments, regulatory or credit-relevant issues, strategic initiatives, and near-term catalysts.
            - `SOURCE QUALITY / WEAK ITEMS`: isolate thin, stale, low-confidence, duplicated, or weakly attributable items, and say why they are weaker.
            - `OPEN QUESTIONS`: list the main unresolved diligence questions remaining after the public-source review.

            Additional rules:
            - Use public web sources only.
            - Prefer recent, issuer-specific, higher-signal sources when available.
            - Do not fabricate dates, quotes, source details, facts, or causal claims.
            - Do not present weak commentary as confirmed fact.
            - Keep confirmed developments separate from unclear evidence.
            - Do not include valuation views, portfolio commentary, or investment advice.
            - End immediately after `OPEN QUESTIONS`.
        """),
        markdown=True,
    )


async def run_demo(prompt: str) -> None:
    openai_api_key = _require_env("OPENAI_API_KEY")
    _require_env("TAVILY_API_KEY")
    tavily_mcp_url = _require_env("EXAMPLE9_TAVILY_MCP_URL")

    try:
        from agno.tools.mcp import MCPTools
    except ImportError as exc:
        raise RuntimeError(
            "Agno MCP support is unavailable because the `mcp` package is not installed. "
            "Install `mcp` and rerun this example."
        ) from exc

    mcp_tools = MCPTools(
        transport="streamable-http",
        url=tavily_mcp_url,
        timeout_seconds=30,
    )
    agent = build_agent(mcp_tools, openai_api_key=openai_api_key)

    try:
        await mcp_tools.connect()
        await agent.aprint_response(prompt, stream=True)
    finally:
        await mcp_tools.close()


def main() -> None:
    args = parse_args()
    prompt = args.prompt or build_default_query(args.issuer, args.focus)

    try:
        asyncio.run(run_demo(prompt))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
