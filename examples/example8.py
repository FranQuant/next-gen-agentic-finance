"""Example 8: compact structured research handoff workflow."""

import os
from pathlib import Path
from textwrap import dedent

from dotenv import load_dotenv

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.team import Team

from finance_tools import (
    get_analyst_recommendations,
    get_company_info,
    get_company_news_tavily,
    get_current_stock_price,
)

load_dotenv()

DEFAULT_MODEL_ID = os.getenv("EXAMPLE8_MODEL_ID", os.getenv("OPENAI_MODEL_ID", "gpt-5.4"))
TEAM_DB_PATH = Path(__file__).resolve().parents[1] / "tmp" / "research_team.db"


def build_team() -> Team:
    TEAM_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = SqliteDb(db_file=str(TEAM_DB_PATH))
    market_data_model = OpenAIResponses(id=DEFAULT_MODEL_ID)
    research_interpreter_model = OpenAIResponses(id=DEFAULT_MODEL_ID)
    open_questions_model = OpenAIResponses(id=DEFAULT_MODEL_ID)
    team_model = OpenAIResponses(id=DEFAULT_MODEL_ID)

    market_data_agent = Agent(
        name="market-data-agent",
        role="Compact factual evidence-packet specialist",
        model=market_data_model,
        tools=[
            get_current_stock_price,
            get_company_info,
            get_company_news_tavily,
            get_analyst_recommendations,
        ],
        instructions=dedent("""
            You are a factual evidence-packet specialist.

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
            - Use the available tools
            - Keep the packet factual only
            - If unavailable, weak, or unclear, label it clearly instead of improvising
            - Keep the whole output under 12 lines
            - Prioritize price, market cap, 52-week range, recent performance, valuation, growth, margins, cash, debt, buybacks, dividend, analyst sentiment, and recent company-specific catalysts
            - Use the exact field names above. Do not rename them.
            - Do not add extra fields.
            - Do not merge or split fields.
            - In `Recent Company Catalysts`, prefer fewer, clearer company-specific catalysts over padded summaries.
            - If catalyst visibility is thin, mixed, or unclear, say that explicitly inside `Recent Company Catalysts`.
            - Do not elevate commentary, interview coverage, or generic discussion into a hard catalyst unless the evidence clearly supports it.
            - Do not produce thesis language, business interpretation, recommendation language, portfolio language, or scenario framing.
        """),
        markdown=True,
    )

    research_interpreter = Agent(
        name="research-interpreter",
        role="Bounded institutional research interpreter",
        model=research_interpreter_model,
        instructions=dedent("""
            You are a research interpreter.

            You will receive a factual evidence packet from the market-data-agent.
            Convert it into a bounded research read.
            Use ONLY that packet.
            Return EXACTLY these 6 bullets and no others.
            Use these labels verbatim, character-for-character.
            Do not rename, shorten, restate, abbreviate, reorder, merge, or split the labels.
            If a field is unclear, still emit the exact label and state the uncertainty briefly.
            Do not omit any label.

            Return ONLY:

            - Core View: ...
            - Setup: ...
            - Evidence Balance: Supportive / Mixed / Weak
            - Key Catalysts: ...
            - Key Risks: ...
            - Key Metrics: ...

            Rules:
            - Keep the interpretation compact and institutional in tone
            - Use ONLY the factual evidence packet
            - Do not introduce external data
            - Do not fabricate numbers
            - Use the exact field names above. Do not rename them.
            - Do not add extra fields.
            - Do not merge or split fields.
            - `Key Catalysts` must include only event, news, or company-specific catalysts explicitly supported by the evidence packet.
            - Do not place valuation relationships, multiple expansion/compression logic, abstract forecast read-throughs, or generic quality descriptions inside `Key Catalysts`.
            - If catalyst visibility in the evidence packet is thin, mixed, or unclear, say so briefly in `Key Catalysts` rather than padding it.
            - Do not produce portfolio stance, sizing language, trading language, or memo prose
            - No fake precision
            - No unnecessary expansion
            - Keep the whole output under 8 lines
            - No extra commentary
        """),
        markdown=True,
    )

    open_questions_agent = Agent(
        name="open-questions-specialist",
        role="Diligence gaps and unresolved questions specialist",
        model=open_questions_model,
        instructions=dedent("""
            You are a diligence gaps specialist.

            You will receive the factual evidence packet and the research read.
            Convert them into a compact open-questions / diligence-gaps packet.
            Use ONLY those inputs.
            Return EXACTLY these 6 bullets and no others.
            Use these labels verbatim, character-for-character.
            Do not rename, shorten, restate, abbreviate, reorder, merge, or split the labels.
            If a field is unclear, still emit the exact label and state the limitation briefly.
            Do not omit any label.

            Return ONLY:

            - What Is Confirmed: ...
            - What Needs Verification: ...
            - Missing Data: ...
            - Source Quality Concerns: ...
            - What Would Strengthen the View: ...
            - What Could Weaken the View: ...

            Rules:
            - Use the evidence packet and research read only
            - Do not make a portfolio recommendation
            - Focus on unresolved diligence items, evidence contradictions, source-quality issues, missing data, and what further information would matter
            - Do not introduce external data
            - Do not fabricate numbers or new facts
            - Use the exact field names above. Do not rename them.
            - Do not add extra fields.
            - Do not merge or split fields.
            - Keep the output compact, specific, and useful
            - Keep each bullet tight; avoid bloated prose, repeated framing, or narrative padding
            - Keep the whole output under 8 lines
            - No extra commentary
        """),
        markdown=True,
    )

    research_handoff_team = Team(
        name="Structured Research Handoff Team",
        model=team_model,
        members=[
            market_data_agent,
            research_interpreter,
            open_questions_agent,
        ],
        instructions=dedent("""
            You orchestrate a compact staged structured research handoff workflow.

            Workflow:
            1. Ask market-data-agent to return EXACTLY these 7 bullets and no others, using these labels verbatim:
               - Price / Market Cap
               - 52-Week Range / Recent Performance
               - Valuation
               - Growth / Margins
               - Balance Sheet / Capital Return
               - Analyst Stance
               - Recent Company Catalysts
               Also require: do not rename, shorten, split, merge, or omit labels; if unavailable, still emit the label and mark it unclear/unavailable; if catalyst visibility is thin, mixed, or unclear, say so explicitly in `Recent Company Catalysts`; prefer fewer, clearer catalysts and do not elevate commentary/interview coverage into a hard catalyst without clear support.
            2. Pass the FULL evidence packet explicitly to research-interpreter and require it to return EXACTLY these 6 bullets and no others, using these labels verbatim:
               - Core View
               - Setup
               - Evidence Balance
               - Key Catalysts
               - Key Risks
               - Key Metrics
               Also require: do not rename, shorten, split, merge, or omit labels; if unclear, still emit the label and state the uncertainty briefly; `Key Catalysts` must include only event/news/company-specific catalysts supported by the evidence packet, not valuation logic or abstract forecast read-throughs.
            3. Pass the FULL evidence packet and FULL research read explicitly to open-questions-specialist and require it to return EXACTLY these 6 bullets and no others, using these labels verbatim:
               - What Is Confirmed
               - What Needs Verification
               - Missing Data
               - Source Quality Concerns
               - What Would Strengthen the View
               - What Could Weaken the View
               Also require: do not rename, shorten, split, merge, or omit labels; if unclear, still emit the label and state the limitation briefly; keep the bullets compact and centered on unresolved diligence items, contradictions, source-quality issues, and missing data.
            4. Produce the final structured research handoff.

            Final output must be EXACTLY:

            EVIDENCE
            <factual evidence packet>

            RESEARCH READ
            <bounded research interpretation>

            OPEN QUESTIONS / GAPS
            <diligence and unresolved questions packet>

            Rules:
            - Maximum 35 lines total
            - Do not ask follow-up questions
            - Do not create extra sections
            - This is a research handoff, not an investment memo and not a PM action note
            - Do not request spreadsheets, deadlines, attachments, or further deliverables
            - End immediately after OPEN QUESTIONS / GAPS
        """),
        db=db,
        markdown=True,
    )

    return research_handoff_team


def main() -> None:
    research_handoff_team = build_team()
    research_handoff_team.print_response(
        "Analyze MSFT and produce a compact structured research handoff.",
        stream=True,
    )


if __name__ == "__main__":
    main()
