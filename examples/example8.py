"""Example 8: compact structured research handoff workflow."""

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Sequence

from dotenv import load_dotenv

from agno.agent import Agent
from agno.team import Team

try:
    from agno.db.sqlite import SqliteDb
except ImportError:
    SqliteDb = None  # type: ignore[assignment,misc]

try:
    from agno.models.openai import OpenAIResponses
except ImportError:
    OpenAIResponses = None  # type: ignore[assignment,misc]

try:
    from finance_tools import (
        get_analyst_recommendations,
        get_company_info,
        get_company_news_tavily,
        get_current_stock_price,
    )
except ImportError:
    get_analyst_recommendations = None  # type: ignore[assignment,misc]
    get_company_info = None  # type: ignore[assignment,misc]
    get_company_news_tavily = None  # type: ignore[assignment,misc]
    get_current_stock_price = None  # type: ignore[assignment,misc]

load_dotenv()

DEFAULT_MODEL_ID = os.getenv("EXAMPLE8_MODEL_ID", os.getenv("OPENAI_MODEL_ID", "gpt-5.4"))
TEAM_DB_PATH = Path(__file__).resolve().parents[1] / "tmp" / "research_team.db"
RUN_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "example8_runs"

EVIDENCE_LABELS = (
    "Price / Market Cap",
    "52-Week Range / Recent Performance",
    "Valuation",
    "Growth / Margins",
    "Balance Sheet / Capital Return",
    "Analyst Stance",
    "Recent Company Catalysts",
)

RESEARCH_READ_LABELS = (
    "Core View",
    "Setup",
    "Evidence Balance",
    "Key Catalysts",
    "Key Risks",
    "Key Metrics",
)

OPEN_QUESTIONS_LABELS = (
    "What Is Confirmed",
    "What Needs Verification",
    "Missing Data",
    "Source Quality Concerns",
    "What Would Strengthen the View",
    "What Could Weaken the View",
)

FINAL_SECTION_TITLES = ("EVIDENCE", "RESEARCH READ", "OPEN QUESTIONS / GAPS")

VALIDATION_CONFIG = {
    "strict_section_titles": True,
    "strict_bullet_labels": True,
    "allow_extra_text": False,
    "allow_duplicate_titles": False,
    "allow_duplicate_labels": False,
}

SECTION_LABELS = {
    FINAL_SECTION_TITLES[0]: EVIDENCE_LABELS,
    FINAL_SECTION_TITLES[1]: RESEARCH_READ_LABELS,
    FINAL_SECTION_TITLES[2]: OPEN_QUESTIONS_LABELS,
}

DEMO_PROMPT = "Analyze MSFT and produce a compact structured research handoff."


def _render_bullet_template(labels: Sequence[str], indent: str = "") -> str:
    return "\n".join(f"{indent}- {label}: ..." for label in labels)


def _render_label_list(labels: Sequence[str], indent: str = "") -> str:
    return "\n".join(f"{indent}- {label}" for label in labels)


def _render_final_handoff_outline(indent: str = "") -> str:
    return "\n".join(
        [
            f"{indent}{FINAL_SECTION_TITLES[0]}",
            f"{indent}<factual evidence packet>",
            "",
            f"{indent}{FINAL_SECTION_TITLES[1]}",
            f"{indent}<bounded research interpretation>",
            "",
            f"{indent}{FINAL_SECTION_TITLES[2]}",
            f"{indent}<diligence and unresolved questions packet>",
        ]
    )


def validate_bulleted_packet(text: str, required_labels: Sequence[str]) -> dict[str, Any]:
    required_labels = tuple(required_labels)
    required_set = set(required_labels)
    label_counts: Counter[str] = Counter()
    bullet_details: list[dict[str, Any]] = []
    non_bullet_lines: list[dict[str, Any]] = []
    malformed_bullets: list[dict[str, Any]] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if not stripped.startswith("- "):
            non_bullet_lines.append({"line_number": line_number, "text": raw_line})
            continue
        if ":" not in stripped[2:]:
            malformed_bullets.append({"line_number": line_number, "text": raw_line})
            continue

        label, value = stripped[2:].split(":", 1)
        label = label.strip()
        value = value.strip()
        bullet_details.append(
            {
                "line_number": line_number,
                "label": label,
                "value": value,
                "text": raw_line,
            }
        )
        if label in required_set:
            label_counts[label] += 1

    found_labels = [entry["label"] for entry in bullet_details]
    present_labels = [label for label in required_labels if label_counts[label] > 0]
    missing_labels = [label for label in required_labels if label_counts[label] == 0]
    duplicate_labels = [label for label in required_labels if label_counts[label] > 1]
    unexpected_labels = [label for label in found_labels if label not in required_set]

    ok = not (
        missing_labels
        or duplicate_labels
        or unexpected_labels
        or malformed_bullets
        or non_bullet_lines
    )

    return {
        "ok": ok,
        "required_labels": list(required_labels),
        "found_labels": found_labels,
        "present_labels": present_labels,
        "missing_labels": missing_labels,
        "duplicate_labels": duplicate_labels,
        "unexpected_labels": unexpected_labels,
        "malformed_bullets": malformed_bullets,
        "non_bullet_lines": non_bullet_lines,
        "bullet_details": bullet_details,
        "label_counts": {label: label_counts[label] for label in required_labels},
    }


def validate_final_handoff(text: str) -> dict[str, Any]:
    section_lines: dict[str, list[dict[str, Any]]] = {title: [] for title in FINAL_SECTION_TITLES}
    seen_titles: list[str] = []
    title_counts: Counter[str] = Counter()
    orphan_lines: list[dict[str, Any]] = []
    current_title: str | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            if current_title is not None:
                section_lines[current_title].append({"line_number": line_number, "text": raw_line})
            continue

        if stripped in FINAL_SECTION_TITLES:
            current_title = stripped
            seen_titles.append(stripped)
            title_counts[stripped] += 1
            continue

        if current_title is None:
            orphan_lines.append({"line_number": line_number, "text": raw_line})
        else:
            section_lines[current_title].append({"line_number": line_number, "text": raw_line})

    section_diagnostics: dict[str, Any] = {}
    for title, labels in SECTION_LABELS.items():
        body_text = "\n".join(line["text"] for line in section_lines[title]).strip()
        section_diagnostics[title] = {
            "body": body_text,
            "body_lines": section_lines[title],
            "bullet_validation": validate_bulleted_packet(body_text, labels),
        }

    missing_section_titles = [title for title in FINAL_SECTION_TITLES if title_counts[title] == 0]
    duplicate_section_titles = [title for title in FINAL_SECTION_TITLES if title_counts[title] > 1]
    section_sequence_ok = seen_titles == list(FINAL_SECTION_TITLES)
    section_validation_ok = all(
        section["bullet_validation"]["ok"] for section in section_diagnostics.values()
    )
    ok = (
        section_sequence_ok
        and not missing_section_titles
        and not duplicate_section_titles
        and not orphan_lines
        and section_validation_ok
    )

    return {
        "ok": ok,
        "expected_section_titles": list(FINAL_SECTION_TITLES),
        "found_section_titles": seen_titles,
        "missing_section_titles": missing_section_titles,
        "duplicate_section_titles": duplicate_section_titles,
        "section_sequence_ok": section_sequence_ok,
        "orphan_lines": orphan_lines,
        "section_diagnostics": section_diagnostics,
    }


def _capture_last_run_text(team: Team) -> tuple[Any | None, str | None, str | None, str | None]:
    try:
        run_output = team.get_last_run_output()
    except Exception as exc:
        return None, None, None, str(exc)

    if run_output is None:
        return None, None, None, None

    run_text = getattr(run_output, "content", None)
    capture_method = None
    capture_error = None

    if isinstance(run_text, str):
        capture_method = "team.get_last_run_output().content"
    else:
        get_content_as_string = getattr(run_output, "get_content_as_string", None)
        if callable(get_content_as_string):
            try:
                run_text = get_content_as_string()
                capture_method = "team.get_last_run_output().get_content_as_string()"
            except Exception as exc:
                run_text = None
                capture_error = str(exc)

    return run_output, run_text, capture_method, capture_error


def _write_run_artifact(team: Team, prompt: str) -> Path | None:
    timestamp = datetime.now(timezone.utc)
    run_output, final_text, capture_method, capture_error = _capture_last_run_text(team)

    try:
        artifact = {
            "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
            "prompt": prompt,
            "team_name": team.name,
            "team_model_id": getattr(getattr(team, "model", None), "id", None),
            "member_names": [member.name for member in team.members],
            "member_model_ids": {
                member.name: getattr(getattr(member, "model", None), "id", None)
                for member in team.members
            },
            "required_labels": {
                "EVIDENCE": list(EVIDENCE_LABELS),
                "RESEARCH READ": list(RESEARCH_READ_LABELS),
                "OPEN QUESTIONS / GAPS": list(OPEN_QUESTIONS_LABELS),
            },
            "final_section_titles": list(FINAL_SECTION_TITLES),
            "validation_config": VALIDATION_CONFIG,
            "run_id": getattr(run_output, "run_id", None),
            "session_id": getattr(run_output, "session_id", None),
            "run_status": getattr(getattr(run_output, "status", None), "value", getattr(run_output, "status", None)),
            "final_text_capture_method": capture_method,
            "final_text_capture_error": capture_error,
            "final_text": final_text,
            "validation_results": validate_final_handoff(final_text) if final_text else None,
        }

        RUN_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp_slug = timestamp.strftime("%Y%m%dT%H%M%SZ")
        run_id = artifact["run_id"]
        artifact_name = f"example8_{timestamp_slug}"
        if run_id:
            artifact_name += f"_{run_id}"
        artifact_path = RUN_ARTIFACT_DIR / f"{artifact_name}.json"
        artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        return artifact_path
    except Exception:
        return None


def build_team() -> Team:
    if SqliteDb is None:
        raise RuntimeError(
            "Agno SQLite support is unavailable because `sqlalchemy` is not installed. "
            "Install `sqlalchemy` and rerun this example."
        )
    if OpenAIResponses is None:
        raise RuntimeError(
            "Agno OpenAI support is unavailable because the `openai` package is not installed. "
            "Install `openai` and rerun this example."
        )
    if any(
        tool is None
        for tool in (
            get_current_stock_price,
            get_company_info,
            get_company_news_tavily,
            get_analyst_recommendations,
        )
    ):
        raise RuntimeError(
            "Shared finance tools are unavailable because one or more example dependencies are missing. "
            "Install the example dependencies and rerun this example."
        )

    TEAM_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = SqliteDb(db_file=str(TEAM_DB_PATH))
    evidence_packet_model = OpenAIResponses(id=DEFAULT_MODEL_ID)
    research_interpreter_model = OpenAIResponses(id=DEFAULT_MODEL_ID)
    open_questions_model = OpenAIResponses(id=DEFAULT_MODEL_ID)
    team_model = OpenAIResponses(id=DEFAULT_MODEL_ID)

    evidence_packet_agent = Agent(
        name="evidence-packet-agent",
        role="Compact evidence-packet specialist for facts, fundamentals, news, and analyst context",
        model=evidence_packet_model,
        tools=[
            get_current_stock_price,
            get_company_info,
            get_company_news_tavily,
            get_analyst_recommendations,
        ],
        instructions=dedent(f"""
            You are a factual evidence-packet specialist.

            Use the available tools and return ONLY a compact factual evidence packet.
            Return EXACTLY these 7 bullets and no others.
            Use these labels verbatim, character-for-character.
            Do not rename, shorten, restate, abbreviate, reorder, merge, or split the labels.
            If a field is unavailable or unclear, still emit the exact label and write `unclear` or `unavailable`.
            Do not omit any label.

            Required output format:

{_render_bullet_template(EVIDENCE_LABELS, indent="            ")}

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
        instructions=dedent(f"""
            You are a research interpreter.

            You will receive a factual evidence packet from the evidence-packet-agent.
            Convert it into a bounded research read.
            Use ONLY that packet.
            Return EXACTLY these 6 bullets and no others.
            Use these labels verbatim, character-for-character.
            Do not rename, shorten, restate, abbreviate, reorder, merge, or split the labels.
            If a field is unclear, still emit the exact label and state the uncertainty briefly.
            Do not omit any label.

            Return ONLY:

{_render_bullet_template(RESEARCH_READ_LABELS, indent="            ")}

            Rules:
            - Keep the interpretation compact and institutional in tone
            - Use ONLY the factual evidence packet
            - Do not introduce external data
            - Do not fabricate numbers
            - Use the exact field names above. Do not rename them.
            - Do not add extra fields.
            - Do not merge or split fields.
            - `Evidence Balance` must be exactly one of: Supportive / Mixed / Weak.
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
        instructions=dedent(f"""
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

{_render_bullet_template(OPEN_QUESTIONS_LABELS, indent="            ")}

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
            evidence_packet_agent,
            research_interpreter,
            open_questions_agent,
        ],
        instructions=dedent(f"""
            You orchestrate a compact staged structured research handoff workflow.

            Workflow:
            1. Ask evidence-packet-agent to return EXACTLY these 7 bullets and no others, using these labels verbatim:
{_render_label_list(EVIDENCE_LABELS, indent="               ")}
               Also require: do not rename, shorten, split, merge, or omit labels; if unavailable, still emit the label and mark it unclear/unavailable; if catalyst visibility is thin, mixed, or unclear, say so explicitly in `Recent Company Catalysts`; prefer fewer, clearer catalysts and do not elevate commentary/interview coverage into a hard catalyst without clear support.
            2. Pass the FULL evidence packet explicitly to research-interpreter and require it to return EXACTLY these 6 bullets and no others, using these labels verbatim:
{_render_label_list(RESEARCH_READ_LABELS, indent="               ")}
               Also require: do not rename, shorten, split, merge, or omit labels; if unclear, still emit the label and state the uncertainty briefly; `Key Catalysts` must include only event/news/company-specific catalysts supported by the evidence packet, not valuation logic or abstract forecast read-throughs.
            3. Pass the FULL evidence packet and FULL research read explicitly to open-questions-specialist and require it to return EXACTLY these 6 bullets and no others, using these labels verbatim:
{_render_label_list(OPEN_QUESTIONS_LABELS, indent="               ")}
               Also require: do not rename, shorten, split, merge, or omit labels; if unclear, still emit the label and state the limitation briefly; keep the bullets compact and centered on unresolved diligence items, contradictions, source-quality issues, and missing data.
            4. Produce the final structured research handoff.

            Final output must be EXACTLY:

{_render_final_handoff_outline(indent="            ")}

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


def run_demo(prompt: str) -> None:
    research_handoff_team = build_team()
    try:
        research_handoff_team.print_response(prompt, stream=True)
    finally:
        _write_run_artifact(research_handoff_team, prompt)


def main() -> None:
    run_demo(DEMO_PROMPT)


if __name__ == "__main__":
    main()
