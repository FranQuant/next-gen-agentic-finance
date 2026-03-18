"""Example 9: compact multi-agent research-to-portfolio PM handoff workflow."""

from textwrap import dedent

from dotenv import load_dotenv

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.team import Team

from finance_tools import (
    get_current_stock_price,
    get_analyst_recommendations,
    get_company_info,
    get_company_news_tavily,
)

load_dotenv()


def build_team() -> Team:
    db = SqliteDb(db_file="tmp/research_team.db")
    model = OpenAIResponses(id="gpt-5.4")

    market_data_agent = Agent(
        name="market-data-agent",
        role="Compact factual research packet specialist",
        model=model,
        tools=[
            get_current_stock_price,
            get_company_info,
            get_company_news_tavily,
            get_analyst_recommendations,
        ],
        instructions=dedent("""
            You are a factual research packet specialist.

            Use the available tools and return ONLY a compact factual evidence packet.
            Return EXACTLY these 7 bullets and no others.
            Use these labels verbatim, character-for-character.
            Do not rename, shorten, restate, abbreviate, reorder, merge, or split the labels.
            If a field is unavailable or unclear, still emit the exact label and write `unclear` or `unavailable`.
            Do not omit any label.

            Required output format:

            - Price / Market Cap: ...
            - 52-Week Range / Recent Performance: ...
            - Valuation: ...
            - Growth / Margins: ...
            - Balance Sheet / Capital Return: ...
            - Analyst Stance: ...
            - Recent Company Catalysts: ...

            Rules:
            - Never fabricate numbers
            - If unavailable, weak, or unclear, label it clearly instead of improvising
            - Keep the whole output under 12 lines
            - Prioritize price, market cap, 52-week range, recent performance, valuation, growth, margins, cash, debt, buybacks, dividend, and analyst sentiment
            - Use the exact field names above. Do not rename them.
            - Do not add extra fields.
            - Do not merge or split fields.
            - Do not output Recent Performance, Recent Catalysts, Segments / Products, Positioning / Developments, Street View, or any renamed variant.
            - Keep the packet factual only.
            - Include only short factual recent company catalysts if clearly supported.
            - Segment/business descriptions should appear only if directly necessary inside Recent Company Catalysts, and even there remain factual and brief.
            - Do not produce thesis language, business interpretation, recommendation language, positioning language, portfolio stance, or scenario framing.
        """),
        markdown=True,
    )

    strategy_interpreter = Agent(
        name="strategy-interpreter",
        role="Bounded institutional research interpreter",
        model=model,
        instructions=dedent("""
            You are a research interpreter.

            You will receive a factual evidence packet from the market-data-agent.
            Convert it into a bounded research read for downstream PM use.
            Use ONLY that packet.
            Return EXACTLY these 6 bullets and no others.
            Use these labels verbatim, character-for-character.
            Do not rename, shorten, restate, abbreviate, reorder, merge, or split the labels.
            If a field is unclear, still emit the exact label and state the uncertainty briefly.
            Do not omit any label.

            Return ONLY:

            - Thesis: ...
            - Setup: ...
            - Evidence Balance: Supportive / Mixed / Weak
            - Catalysts: ...
            - Risks: ...
            - Key Metrics: ...

            Rules:
            - Keep the interpretation compact and institutional in tone
            - Do not introduce external data
            - Do not fabricate numbers
            - Use the exact field names above. Do not rename them.
            - Do not add extra fields.
            - Do not merge or split fields.
            - Do not produce sizing language, trade expression, or portfolio stance
            - Do not write memo prose
            - No fake precision
            - No unnecessary expansion
            - Keep the whole output under 8 lines
            - No extra commentary
        """),
        markdown=True,
    )

    portfolio_manager = Agent(
        name="portfolio-manager",
        role="Portfolio manager",
        model=model,
        instructions=dedent("""
            You are a portfolio manager.

            You will receive the strategy-interpreter output.
            Convert it into a compact portfolio stance.

            Return ONLY:

            - Signal: LONG / SHORT / NEUTRAL
            - Conviction: Low / Medium / High
            - Weight: Small / Medium / Large
            - Horizon: ...
            - Action: ...
            - Risk Management: ...
            - What Changes the View: ...

            Rules:
            - Base the action only on the strategy-interpreter output
            - Signal must reflect the direction implied by the research read
            - Conviction must reflect the consistency of thesis, evidence balance, catalysts, and risks
            - Weight must be constrained by conviction and evidence strength
            - Use Large sparingly and only when the read is unusually coherent and well-supported
            - If Evidence Balance is Weak, do not use High conviction or Large weight
            - If Signal is NEUTRAL, the Action should imply no active overweight / underweight posture
            - Keep the output qualitative and bounded
            - Do not fabricate precise percentages, stops, option costs, target levels, or pseudo-risk-engine language
            - Keep the whole output under 8 lines
            - No extra commentary
        """),
        markdown=True,
    )

    research_team = Team(
        name="Research-to-Portfolio PM Handoff Team",
        model=model,
        members=[
            market_data_agent,
            strategy_interpreter,
            portfolio_manager,
        ],
        instructions=dedent("""
            You orchestrate a compact staged research-to-portfolio PM handoff workflow.

            Workflow:
            1. Ask market-data-agent to return EXACTLY these 7 bullets and no others, using these labels verbatim:
               - Price / Market Cap
               - 52-Week Range / Recent Performance
               - Valuation
               - Growth / Margins
               - Balance Sheet / Capital Return
               - Analyst Stance
               - Recent Company Catalysts
               Also require: do not rename, shorten, split, merge, or omit labels; if unavailable, still emit the label and mark it unclear/unavailable.
            2. Pass the FULL evidence packet explicitly to strategy-interpreter and require it to return EXACTLY these 6 bullets and no others, using these labels verbatim:
               - Thesis
               - Setup
               - Evidence Balance
               - Catalysts
               - Risks
               - Key Metrics
               Also require: do not rename, shorten, split, merge, or omit labels; if unclear, still emit the label and state uncertainty briefly.
            3. Pass the FULL research read explicitly to portfolio-manager.
            4. Produce the final PM handoff brief.

            Final output must be EXACTLY:

            EVIDENCE
            <factual evidence packet>

            RESEARCH READ
            <bounded research interpretation>

            PORTFOLIO STANCE
            <portfolio manager output>

            Rules:
            - Maximum 30 lines total
            - Do not ask follow-up questions
            - Do not create extra sections
            - This is a PM handoff brief, not an investment memo
            - Do not request spreadsheets, deadlines, attachments, or further deliverables
            - End immediately after PORTFOLIO STANCE
        """),
        db=db,
        markdown=True,
    )

    return research_team


def main() -> None:
    research_team = build_team()
    research_team.print_response(
        "Analyze MSFT and produce a compact research-to-portfolio brief for PM handoff.",
        stream=True,
        show_full_reasoning=True,
    )


if __name__ == "__main__":
    main()
