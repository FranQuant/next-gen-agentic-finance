"""Example 8: compact structured research handoff workflow."""

import argparse
import io
import json
import os
import re
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent, shorten
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
        selective_extract_shortlisted_urls_tavily,
    )
except ImportError:
    get_analyst_recommendations = None  # type: ignore[assignment,misc]
    get_company_info = None  # type: ignore[assignment,misc]
    get_company_news_tavily = None  # type: ignore[assignment,misc]
    get_current_stock_price = None  # type: ignore[assignment,misc]
    selective_extract_shortlisted_urls_tavily = None  # type: ignore[assignment,misc]

load_dotenv()

DEFAULT_MODEL_ID = os.getenv("EXAMPLE8_MODEL_ID", os.getenv("OPENAI_MODEL_ID", "gpt-5.4"))
REPO_ROOT = Path(__file__).resolve().parents[1]
TEAM_DB_PATH = REPO_ROOT / "tmp" / "research_team.db"
RUN_ARTIFACT_DIR = REPO_ROOT / "tmp" / "example8_runs"

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


def _compact_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return " ".join(text.split()) or None


def _renderable_image_description(value: Any) -> str | None:
    text = _compact_text(value)
    if not text:
        return None

    lowered = text.lower()
    if len(text) > 120:
        return None
    if lowered in {"...", "n/a", "na", "none", "null", "unknown", "unavailable"}:
        return None
    if not re.search(r"[a-z0-9]", lowered):
        return None
    if "..." in text or "…" in text:
        return None
    if re.search(r"https?://|www\.", lowered):
        return None
    if "placeholder" in lowered:
        return None
    if lowered.startswith("image description") and any(
        token in lowered for token in ("missing", "unavailable", "n/a", "none", "unknown")
    ):
        return None

    return text


def _invoke_tool(tool: Any, *args: Any, **kwargs: Any) -> Any:
    if callable(tool):
        return tool(*args, **kwargs)

    entrypoint = getattr(tool, "entrypoint", None)
    if callable(entrypoint):
        return entrypoint(*args, **kwargs)

    raise TypeError("Tool object is not callable and has no callable entrypoint.")


def _invoke_tool_silently(tool: Any, *args: Any, **kwargs: Any) -> Any:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        return _invoke_tool(tool, *args, **kwargs)


def _infer_symbol_from_prompt(prompt: str) -> str | None:
    match = re.search(r"\b([A-Z]{2,5})\b", prompt or "")
    if match is None:
        return None
    return match.group(1)


def _source_card_bucket_marker(bucket: str | None) -> str:
    if bucket == "high_confidence_company_specific":
        return "[HIGH]"
    if bucket == "broader_context":
        return "[MIXED]"
    if bucket == "weak_or_generic":
        return "[WEAK]"
    return "[MIXED]"


def _source_card_caution(card: dict[str, Any]) -> str | None:
    if card.get("extraction_status") == "failed":
        return "selective extraction failed; verify directly"

    bucket = card.get("relevance_bucket")
    if bucket == "broader_context":
        return "contextual coverage; not a confirmed company-specific catalyst"
    if bucket == "weak_or_generic":
        return "low-confidence / generic coverage"
    if bucket is None:
        return "confidence not established"
    return None


def _normalize_source_card(
    item: dict[str, Any],
    *,
    source_type: str = "tavily_news_search",
    extracted: bool = False,
    extraction_status: str = "not_requested",
    extraction_error: str | None = None,
    selected_rank: int | None = None,
) -> dict[str, Any]:
    ranking_score = item.get("ranking_score")
    if ranking_score is None:
        ranking_score = item.get("score")

    card: dict[str, Any] = {
        "title": _compact_text(item.get("title")),
        "publisher": _compact_text(item.get("publisher")),
        "date": _compact_text(item.get("date")),
        "url": _compact_text(item.get("url")),
        "favicon": _compact_text(item.get("favicon")),
        "image": _compact_text(item.get("image")),
        "image_description": _compact_text(item.get("image_description")),
        "query_category": _compact_text(item.get("query_category")),
        "relevance_bucket": _compact_text(item.get("relevance_bucket")),
        "extracted": extracted,
        "extraction_status": extraction_status,
        "source_type": source_type,
        "ranking_score": ranking_score,
        "caution": _compact_text(item.get("caution")) or _source_card_caution(item),
        "snippet": _compact_text(item.get("snippet")),
    }

    if selected_rank is not None:
        card["selected_rank"] = selected_rank
    if extraction_error:
        card["extraction_error"] = _compact_text(extraction_error)
    if extracted and item.get("content") is not None:
        card["content"] = _compact_text(item.get("content"))

    return card


def _update_card_extraction_state(
    cards: list[dict[str, Any]],
    extraction_records: Sequence[dict[str, Any]],
) -> None:
    card_by_url = {
        _compact_text(card.get("url")): card
        for card in cards
        if _compact_text(card.get("url"))
    }

    for record in extraction_records:
        url = _compact_text(record.get("url"))
        if not url or url not in card_by_url:
            continue

        card = card_by_url[url]
        card["extracted"] = bool(record.get("extracted"))
        card["extraction_status"] = _compact_text(record.get("extraction_status")) or "not_requested"
        if record.get("content") is not None and record.get("extracted"):
            card["content"] = _compact_text(record.get("content"))
        if record.get("extraction_error"):
            card["extraction_error"] = _compact_text(record.get("extraction_error"))
        card["caution"] = _source_card_caution(card)


def _selective_extraction_needed(news_packet: dict[str, Any], source_cards: Sequence[dict[str, Any]]) -> bool:
    if not source_cards:
        return False

    query_failures = news_packet.get("query_failures") or []
    if query_failures:
        return True

    if (news_packet.get("returned_count") or 0) < 2:
        return True

    high_confidence_count = sum(
        1 for card in source_cards if card.get("relevance_bucket") == "high_confidence_company_specific"
    )
    has_non_high_confidence = any(
        card.get("relevance_bucket") != "high_confidence_company_specific" for card in source_cards
    )
    if high_confidence_count < 2 and has_non_high_confidence:
        return True

    quality_note = _compact_text(news_packet.get("news_quality_note")) or ""
    lowered_note = quality_note.lower()
    return any(
        phrase in lowered_note
        for phrase in (
            "mixed result set",
            "weak result set",
            "no sufficiently relevant",
        )
    )


def _build_source_inspection(prompt: str) -> dict[str, Any]:
    symbol = _infer_symbol_from_prompt(prompt)
    if not symbol:
        return {
            "symbol": None,
            "company_name": None,
            "company_info_packet": {"ok": False, "error": "Could not infer a ticker symbol from the prompt."},
            "news_packet": {"ok": False, "error": "Could not infer a ticker symbol from the prompt."},
            "source_cards": [],
            "weak_or_low_confidence_items": [],
            "query_failures": [],
            "selected_extraction_records": [],
            "extraction_results": None,
            "extraction_triggered": False,
            "selective_extraction_needed": False,
            "inspection_warnings": [],
        }

    inspection_warnings: list[dict[str, str]] = []
    company_info_failed = False

    try:
        company_info_packet = _invoke_tool_silently(get_company_info, symbol)
    except Exception:
        company_info_packet = {"symbol": symbol, "ok": False, "error": "company info unavailable"}
        inspection_warnings.append({"stage": "company_info", "message": "company info unavailable"})
        company_info_failed = True

    if not isinstance(company_info_packet, dict):
        company_info_packet = {"symbol": symbol, "ok": False, "error": "company info unavailable"}
        inspection_warnings.append({"stage": "company_info", "message": "company info unavailable"})
        company_info_failed = True

    if not company_info_packet.get("ok") and not company_info_failed:
        inspection_warnings.append({"stage": "company_info", "message": "company info unavailable"})

    company_info = company_info_packet.get("company_info") if isinstance(company_info_packet, dict) else {}
    company_name = _compact_text(company_info.get("longName")) if isinstance(company_info, dict) else None

    try:
        news_packet = _invoke_tool_silently(
            get_company_news_tavily,
            symbol,
            company_name=company_name or "",
            num_stories=5,
        )
        news_search_failed = False
    except Exception:
        news_packet = {"symbol": symbol, "ok": False, "news": [], "query_failures": [], "error": "news search unavailable"}
        news_search_failed = True
        inspection_warnings.append({"stage": "news_search", "message": "news search unavailable"})

    if not isinstance(news_packet, dict):
        news_packet = {"symbol": symbol, "ok": False, "news": [], "query_failures": [], "error": "news search unavailable"}
        news_search_failed = True
        inspection_warnings.append({"stage": "news_search", "message": "news search unavailable"})

    if not news_packet.get("ok") and not news_search_failed:
        inspection_warnings.append({"stage": "news_search", "message": "news search unavailable"})

    news_items = news_packet.get("news") if isinstance(news_packet, dict) else []
    source_cards = [
        _normalize_source_card(item)
        for item in news_items
        if isinstance(item, dict)
    ]
    weak_or_low_confidence_items = [
        card
        for card in source_cards
        if card.get("relevance_bucket") != "high_confidence_company_specific"
    ]

    query_failures = news_packet.get("query_failures") or []
    selective_extraction_needed = _selective_extraction_needed(news_packet, source_cards)
    extraction_results = None
    selected_extraction_records: list[dict[str, Any]] = []
    extraction_triggered = False

    if selective_extract_shortlisted_urls_tavily is not None and selective_extraction_needed:
        shortlist_query = f"{company_name or symbol} selective verification"
        try:
            extraction_results = _invoke_tool_silently(
                selective_extract_shortlisted_urls_tavily,
                source_cards,
                query=shortlist_query,
                max_urls=3,
            )
            extraction_failed = False
        except Exception:
            extraction_results = {
                "ok": False,
                "source": "Tavily selective extraction",
                "results": [],
                "failed_results": [],
                "error": "selective extraction unavailable",
            }
            extraction_failed = True
            inspection_warnings.append({"stage": "selective_extraction", "message": "selective extraction unavailable"})

        if not isinstance(extraction_results, dict):
            extraction_results = {
                "ok": False,
                "source": "Tavily selective extraction",
                "results": [],
                "failed_results": [],
                "error": "selective extraction unavailable",
            }
            extraction_failed = True
            inspection_warnings.append({"stage": "selective_extraction", "message": "selective extraction unavailable"})

        if isinstance(extraction_results, dict):
            selected_extraction_records = [
                record
                for record in [
                    *extraction_results.get("results", []),
                    *extraction_results.get("failed_results", []),
                ]
                if isinstance(record, dict)
            ]
            extraction_triggered = bool(selected_extraction_records)
            _update_card_extraction_state(source_cards, selected_extraction_records)
            weak_or_low_confidence_items = [
                card
                for card in source_cards
                if card.get("relevance_bucket") != "high_confidence_company_specific"
            ]
            if not extraction_results.get("ok") and not extraction_failed:
                inspection_warnings.append({"stage": "selective_extraction", "message": "selective extraction unavailable"})

    return {
        "symbol": symbol,
        "company_name": company_name,
        "company_info_packet": company_info_packet,
        "news_packet": news_packet,
        "source_cards": source_cards,
        "weak_or_low_confidence_items": weak_or_low_confidence_items,
        "query_failures": query_failures,
        "selected_extraction_records": selected_extraction_records,
        "extraction_results": extraction_results,
        "extraction_triggered": extraction_triggered,
        "selective_extraction_needed": selective_extraction_needed,
        "inspection_warnings": inspection_warnings,
    }


def _render_source_card(card: dict[str, Any]) -> str:
    marker = _source_card_bucket_marker(card.get("relevance_bucket"))
    if card.get("extracted"):
        marker = f"{marker} [EXTRACTED]"
    elif card.get("extraction_status") == "failed":
        marker = f"{marker} [FAILED]"

    title = _compact_text(card.get("title")) or "Untitled"
    publisher = _compact_text(card.get("publisher")) or "unknown"
    date = _compact_text(card.get("date")) or "n/a"
    url = _compact_text(card.get("url")) or "n/a"
    query_category = _compact_text(card.get("query_category")) or "n/a"
    relevance_bucket = _compact_text(card.get("relevance_bucket")) or "n/a"
    source_type = _compact_text(card.get("source_type")) or "n/a"
    extraction_status = _compact_text(card.get("extraction_status")) or "n/a"
    ranking_score = card.get("ranking_score")
    snippet = _compact_text(card.get("snippet"))
    content = _compact_text(card.get("content"))
    caution = _compact_text(card.get("caution"))

    lines = [
        f"{marker} {title}",
        f"  publisher: {publisher} | date: {date}",
        f"  url: {url}",
        f"  query_category: {query_category} | relevance_bucket: {relevance_bucket}",
        f"  source_type: {source_type} | extracted: {'yes' if card.get('extracted') else 'no'} | extraction_status: {extraction_status}",
    ]
    if ranking_score is not None:
        lines.append(f"  ranking_score: {ranking_score}")
    if _compact_text(card.get("favicon")):
        lines.append(f"  favicon: {_compact_text(card.get('favicon'))}")
    if _compact_text(card.get("image")):
        lines.append(f"  image: {_compact_text(card.get('image'))}")
    image_description = _renderable_image_description(card.get("image_description"))
    if image_description:
        lines.append(f"  image_description: {image_description}")
    if caution:
        lines.append(f"  caution: {caution}")
    if snippet:
        lines.append(f"  snippet: {shorten(snippet, width=180, placeholder='...')}")
    if content and card.get("extracted"):
        lines.append(f"  content_preview: {shorten(content, width=240, placeholder='...')}")

    return "\n".join(lines)


def _render_card_section(title: str, cards: Sequence[dict[str, Any]]) -> str:
    lines = [title, "-" * len(title)]
    if not cards:
        lines.append("- none")
        return "\n".join(lines)

    for index, card in enumerate(cards, start=1):
        if index > 1:
            lines.append("")
        lines.append(_render_source_card(card))
    return "\n".join(lines)


def _render_query_failures(title: str, query_failures: Sequence[dict[str, Any]]) -> str:
    lines = [title, "-" * len(title)]
    if not query_failures:
        lines.append("- none")
        return "\n".join(lines)

    for failure in query_failures:
        query_category = _compact_text(failure.get("query_category")) or "unknown"
        message = _compact_text(failure.get("message")) or "query unavailable"
        lines.append(f"- {query_category}: {message}")

    return "\n".join(lines)


def _render_inspection_warnings(title: str, warnings: Sequence[dict[str, Any]]) -> str:
    lines = [title, "-" * len(title)]
    if not warnings:
        lines.append("- none")
        return "\n".join(lines)

    for warning in warnings:
        stage = _compact_text(warning.get("stage")) or "inspection"
        message = _compact_text(warning.get("message")) or "unavailable"
        lines.append(f"- {stage}: {message}")

    return "\n".join(lines)


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_run_artifacts(
    team: Team,
    prompt: str,
    source_inspection: dict[str, Any] | None,
    *,
    artifact_root: Path = RUN_ARTIFACT_DIR,
) -> Path | None:
    timestamp = datetime.now(timezone.utc)
    run_output, final_text, capture_method, capture_error = _capture_last_run_text(team)

    try:
        run_id = getattr(run_output, "run_id", None)
        session_id = getattr(run_output, "session_id", None)
        run_status = getattr(getattr(run_output, "status", None), "value", getattr(run_output, "status", None))

        timestamp_slug = timestamp.strftime("%Y%m%dT%H%M%SZ")
        run_dir_name = timestamp_slug
        if run_id:
            run_dir_name += f"_{run_id}"
        run_dir = artifact_root / run_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

        source_cards = list((source_inspection or {}).get("source_cards", []))
        weak_cards = list((source_inspection or {}).get("weak_or_low_confidence_items", []))
        extraction_results = (source_inspection or {}).get("extraction_results")
        query_failures = list((source_inspection or {}).get("query_failures", []))
        inspection_warnings = list((source_inspection or {}).get("inspection_warnings", []))

        source_cards_payload = {
            "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
            "symbol": (source_inspection or {}).get("symbol"),
            "company_name": (source_inspection or {}).get("company_name"),
            "news_quality_note": (source_inspection or {}).get("news_packet", {}).get("news_quality_note")
            if isinstance((source_inspection or {}).get("news_packet"), dict)
            else None,
            "event_diversity_note": (source_inspection or {}).get("news_packet", {}).get("event_diversity_note")
            if isinstance((source_inspection or {}).get("news_packet"), dict)
            else None,
            "query_failures": query_failures,
            "source_cards": source_cards,
            "weak_or_low_confidence_items": weak_cards,
            "selected_extraction_records": list((source_inspection or {}).get("selected_extraction_records", [])),
            "extraction_triggered": bool((source_inspection or {}).get("extraction_triggered")),
            "selective_extraction_needed": bool((source_inspection or {}).get("selective_extraction_needed")),
            "inspection_warnings": inspection_warnings,
        }
        source_cards_path = run_dir / "source_cards.json"
        _write_json_file(source_cards_path, source_cards_payload)

        extraction_path = None
        if isinstance(extraction_results, dict) and (
            extraction_results.get("results") or extraction_results.get("failed_results")
        ):
            extraction_path = run_dir / "extraction_results.json"
            _write_json_file(extraction_path, extraction_results)

        run_manifest = {
            "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
            "prompt": prompt,
            "symbol": (source_inspection or {}).get("symbol"),
            "company_name": (source_inspection or {}).get("company_name"),
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
            "run_id": run_id,
            "session_id": session_id,
            "run_status": run_status,
            "final_text_capture_method": capture_method,
            "final_text_capture_error": capture_error,
            "final_text": final_text,
            "validation_results": validate_final_handoff(final_text) if final_text else None,
            "source_cards_path": _display_path(source_cards_path),
            "extraction_results_path": _display_path(extraction_path) if extraction_path else None,
            "inspection_warnings": inspection_warnings,
            "artifacts": {
                "source_cards": "source_cards.json",
                "extraction_results": "extraction_results.json" if extraction_path else None,
                "run_manifest": "run_manifest.json",
            },
        }
        _write_json_file(run_dir / "run_manifest.json", run_manifest)
        return run_dir
    except Exception:
        return None


def _render_extraction_section(source_inspection: dict[str, Any]) -> str:
    extraction_results = source_inspection.get("extraction_results")
    if isinstance(extraction_results, dict) and (
        extraction_results.get("results") or extraction_results.get("failed_results")
    ):
        extraction_cards = [
            record
            for record in [
                *extraction_results.get("results", []),
                *extraction_results.get("failed_results", []),
            ]
            if isinstance(record, dict)
        ]
        return _render_card_section("EXTRACTIONS", extraction_cards)

    if source_inspection.get("selective_extraction_needed") and selective_extract_shortlisted_urls_tavily is None:
        note = "selective extraction helper unavailable"
    else:
        note = "selective extraction not triggered"

    return "\n".join(["EXTRACTIONS", "-----------", f"- {note}"])


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


def run_demo(
    prompt: str,
    *,
    show_sources: bool = False,
    show_weak_items: bool = False,
    show_extractions: bool = False,
    write_artifacts: bool = False,
    artifact_root: Path = RUN_ARTIFACT_DIR,
) -> None:
    research_handoff_team = build_team()
    capture_team_output = show_sources or show_weak_items or show_extractions or write_artifacts
    if capture_team_output:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            research_handoff_team.print_response(prompt, stream=True)

        _, final_text, _, _ = _capture_last_run_text(research_handoff_team)
        if final_text:
            print(final_text)
        else:
            fallback_text = _compact_text(stdout_buffer.getvalue())
            if fallback_text:
                print(fallback_text)
    else:
        research_handoff_team.print_response(prompt, stream=True)

    if not (show_sources or show_weak_items or show_extractions or write_artifacts):
        return

    source_inspection = _build_source_inspection(prompt)
    rendered_sections: list[str] = []

    if show_sources:
        rendered_sections.append(_render_card_section("SOURCE CARDS", source_inspection["source_cards"]))
    if show_weak_items:
        rendered_sections.append(
            _render_card_section(
                "WEAK / LOW-CONFIDENCE ITEMS",
                source_inspection["weak_or_low_confidence_items"],
            )
        )
    if show_extractions:
        rendered_sections.append(_render_extraction_section(source_inspection))

    if (show_sources or show_weak_items or show_extractions) and source_inspection.get("query_failures"):
        rendered_sections.append(_render_query_failures("QUERY FAILURES", source_inspection["query_failures"]))
    if (show_sources or show_weak_items or show_extractions) and source_inspection.get("inspection_warnings"):
        rendered_sections.append(
            _render_inspection_warnings("INSPECTION WARNINGS", source_inspection["inspection_warnings"])
        )

    if rendered_sections:
        print()
        print("\n\n".join(rendered_sections))

    if write_artifacts:
        artifact_dir = _write_run_artifacts(
            research_handoff_team,
            prompt,
            source_inspection,
            artifact_root=artifact_root,
        )
        if artifact_dir is not None:
            print()
            print(f"Artifacts: {_display_path(artifact_dir)}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the structured research handoff example with optional CLI source inspection."
    )
    parser.add_argument("--show-sources", action="store_true", help="Print normalized source cards.")
    parser.add_argument(
        "--show-weak-items",
        action="store_true",
        help="Print the weak or low-confidence source cards in a separate section.",
    )
    parser.add_argument(
        "--show-extractions",
        action="store_true",
        help="Print selective Tavily extraction results when verification is warranted.",
    )
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="Write JSON sidecar artifacts for the run.",
    )
    args = parser.parse_args(argv)
    run_demo(
        DEMO_PROMPT,
        show_sources=args.show_sources,
        show_weak_items=args.show_weak_items,
        show_extractions=args.show_extractions,
        write_artifacts=args.write_artifacts,
    )


if __name__ == "__main__":
    main()
