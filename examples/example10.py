"""Example 10: bounded holdings/watchlist monitoring demo bootstrap."""

import argparse
import csv
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

CANONICAL_INPUT_PATH = Path("data/example10_watchlist.csv")
CANONICAL_SCHEMA = (
    "ticker",
    "name",
    "weight",
    "thesis",
    "risk_bucket",
    "region",
    "priority",
    "notes",
)
MIN_ROWS = 1
WEIGHT_TOLERANCE = 0.001
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]*$")
TEXT_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
REQUIRED_NON_EMPTY_TEXT_FIELDS = ("ticker", "name", "thesis", "risk_bucket", "region")
MAX_RECENT_ITEMS_PER_ISSUER = 4
DEFAULT_QUERY_CATEGORIES = (
    "broad_company_news",
    "strategic_regulatory_monitoring",
)
HIGH_SIGNAL_QUERY_CATEGORIES = {
    "strategic_regulatory_monitoring",
    "product_strategy",
    "regulatory_legal",
    "management_commentary",
}
LIMITED_QUALITY_PREFIXES = (
    "Sparse issuer-specific evidence",
    "Contextual issuer evidence only",
    "Mixed issuer evidence",
    "Weak issuer evidence",
    "Weak result set",
    "No sufficiently relevant",
    "No sufficiently relevant issuer-specific evidence remained",
)
LOW_CONFIDENCE_SOURCE_DOMAINS = {
    "fool.com",
    "forbes.com",
    "insurancenewsnet.com",
    "investors.com",
    "seekingalpha.com",
    "stocktitan.net",
    "thestreet.com",
    "tipranks.com",
}
MONITORING_TOKEN_STOPWORDS = {
    "development",
    "developments",
    "leader",
    "leaders",
    "monitor",
    "platform",
    "platforms",
}
COMPANY_TOKEN_STOPWORDS = {
    "and",
    "co",
    "company",
    "corp",
    "corporation",
    "financial",
    "financiero",
    "group",
    "grupo",
    "holdings",
    "inc",
    "limited",
    "ltd",
    "sa",
    "sociedad",
    "the",
    "y",
}
GENERIC_ROUNDUP_HINTS = (
    "big things we're watching",
    "market roundup",
    "market wrap",
    "stock market this week",
    "stocks to watch",
    "week ahead",
    "what we're watching",
)


class WatchlistValidationError(Exception):
    """Raised when the Example 10 watchlist CSV fails validation."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


@dataclass(frozen=True)
class WatchlistRow:
    ticker: str
    name: str
    weight: float
    thesis: str
    risk_bucket: str
    region: str
    priority: int
    notes: str


@dataclass(frozen=True)
class EvidenceRecord:
    title: str
    publisher: str
    date: str | None
    url: str
    snippet: str | None
    query_category: str | None
    relevance_bucket: str | None


@dataclass(frozen=True)
class IssuerReview:
    ticker: str
    name: str
    weight: float
    thesis: str
    risk_bucket: str
    region: str
    priority: int
    notes: str
    recent_development_summary: str
    evidence: tuple[EvidenceRecord, ...]
    evidence_quality_note: str
    requires_attention: bool
    attention_reason: str
    retrieval_ok: bool
    retrieval_error: str | None
    query_failures: tuple[str, ...]
    contextual_evidence_only: bool
    filtered_out_count: int
    status: str
    status_note: str


IssuerRetrievalFn = Callable[[WatchlistRow, int], dict[str, Any]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Example 10 holdings/watchlist monitoring demo bootstrap."
    )
    parser.add_argument(
        "--input",
        default=str(CANONICAL_INPUT_PATH),
        help="Path to the Example 10 watchlist CSV.",
    )
    return parser.parse_args(argv)


def load_watchlist_csv(path: str | Path) -> list[WatchlistRow]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise WatchlistValidationError([f"Input file not found: {csv_path}"])

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)

            try:
                header = next(reader)
            except StopIteration as exc:
                raise WatchlistValidationError(["CSV is empty."]) from exc

            if header != list(CANONICAL_SCHEMA):
                expected = ",".join(CANONICAL_SCHEMA)
                raise WatchlistValidationError(
                    [f"Header mismatch. Expected exactly: {expected}"]
                )

            errors: list[str] = []
            rows: list[WatchlistRow] = []
            seen_tickers: dict[str, int] = {}
            total_weight = 0.0
            has_invalid_weight = False
            data_row_count = 0

            for line_number, raw_row in enumerate(reader, start=2):
                if not raw_row or not any(cell.strip() for cell in raw_row):
                    errors.append(f"Row {line_number}: blank row not allowed.")
                    continue

                data_row_count += 1

                if len(raw_row) != len(CANONICAL_SCHEMA):
                    errors.append(
                        f"Row {line_number}: expected {len(CANONICAL_SCHEMA)} columns, found {len(raw_row)}."
                    )
                    continue

                record = {
                    field: value.strip()
                    for field, value in zip(CANONICAL_SCHEMA, raw_row, strict=True)
                }

                row_errors: list[str] = []

                ticker = record["ticker"]
                if not ticker:
                    row_errors.append("ticker is required.")
                else:
                    normalized_ticker = ticker.upper()

                    if ticker != normalized_ticker:
                        row_errors.append("ticker must be uppercase.")
                    if not TICKER_PATTERN.fullmatch(ticker):
                        row_errors.append("ticker format is invalid.")

                    prior_line = seen_tickers.get(normalized_ticker)
                    if prior_line is not None:
                        row_errors.append(
                            f"duplicate ticker '{ticker}' also appears on row {prior_line}."
                        )
                    else:
                        seen_tickers[normalized_ticker] = line_number

                for field in REQUIRED_NON_EMPTY_TEXT_FIELDS[1:]:
                    if not record[field]:
                        row_errors.append(f"{field} is required.")

                weight: float | None = None
                try:
                    weight = float(record["weight"])
                except ValueError:
                    row_errors.append("weight must be a number.")
                    has_invalid_weight = True
                else:
                    if weight <= 0 or weight > 1:
                        row_errors.append("weight must be > 0 and <= 1.")
                        has_invalid_weight = True
                    else:
                        total_weight += weight

                priority: int | None = None
                try:
                    priority = int(record["priority"])
                except ValueError:
                    row_errors.append("priority must be an integer >= 1.")
                else:
                    if priority < 1:
                        row_errors.append("priority must be an integer >= 1.")

                if row_errors:
                    errors.extend(f"Row {line_number}: {error}" for error in row_errors)
                    continue

                rows.append(
                    WatchlistRow(
                        ticker=ticker,
                        name=record["name"],
                        weight=weight,
                        thesis=record["thesis"],
                        risk_bucket=record["risk_bucket"],
                        region=record["region"],
                        priority=priority,
                        notes=record["notes"],
                    )
                )

            if data_row_count < MIN_ROWS:
                errors.append(f"CSV must contain at least {MIN_ROWS} data row.")
            if not has_invalid_weight and data_row_count >= MIN_ROWS:
                if abs(total_weight - 1.0) > WEIGHT_TOLERANCE:
                    errors.append(
                        f"Total weight must sum to 1.0 +/- {WEIGHT_TOLERANCE:.3f}; got {total_weight:.4f}."
                    )

            if errors:
                raise WatchlistValidationError(errors)

            return rows
    except WatchlistValidationError:
        raise
    except UnicodeDecodeError as exc:
        raise WatchlistValidationError(
            [f"Could not decode CSV as UTF-8: {csv_path}"]
        ) from exc
    except csv.Error as exc:
        raise WatchlistValidationError(
            [f"CSV parse error in {csv_path}: {exc}"]
        ) from exc
    except OSError as exc:
        raise WatchlistValidationError(
            [f"Could not read input file: {csv_path}"]
        ) from exc


def order_watchlist_rows(rows: Sequence[WatchlistRow]) -> list[WatchlistRow]:
    return sorted(rows, key=lambda row: (row.priority, -row.weight, row.ticker))


def build_portfolio_watchlist_summary(rows: Sequence[WatchlistRow]) -> str:
    top_weights = sorted(rows, key=lambda row: (-row.weight, row.ticker))[:3]
    priority_counts = Counter(row.priority for row in rows)
    region_counts = Counter(row.region for row in rows)
    risk_counts = Counter(row.risk_bucket for row in rows)

    priority_mix = ", ".join(
        f"P{priority}: {priority_counts[priority]}" for priority in sorted(priority_counts)
    )
    region_mix = ", ".join(
        f"{region}: {region_counts[region]}" for region in sorted(region_counts)
    )
    risk_mix = ", ".join(
        f"{risk}: {risk_counts[risk]}" for risk in sorted(risk_counts)
    )
    top_weight_text = ", ".join(
        f"{row.ticker} {row.weight * 100:.1f}%" for row in top_weights
    )

    lines = [
        "## Portfolio / Watchlist Summary",
        f"- Names: {len(rows)}",
        f"- Total weight: {sum(row.weight for row in rows) * 100:.1f}%",
        f"- Top weights: {top_weight_text}",
        f"- Priority mix: {priority_mix}",
        f"- Region mix: {region_mix}",
        f"- Risk mix: {risk_mix}",
    ]
    return "\n".join(lines)


def _default_retrieval_fn(row: WatchlistRow, max_items: int) -> dict[str, Any]:
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise RuntimeError("tavily package is unavailable.") from exc

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "symbol": row.ticker,
            "ok": False,
            "error": "TAVILY_API_KEY not set.",
        }

    client = TavilyClient(api_key=api_key)
    query_failures: list[dict[str, str]] = []
    collected: list[dict[str, Any]] = []
    max_results_per_query = max(1, min(max_items, 2))

    for query_info in build_default_retrieval_queries(row):
        try:
            response = client.search(
                query=query_info["query"],
                topic="news",
                days=30,
                max_results=max_results_per_query,
                include_raw_content=False,
            )
        except Exception as exc:
            query_failures.append(
                {
                    "query_category": query_info["query_category"],
                    "error": str(exc),
                }
            )
            continue

        results = response.get("results", []) if isinstance(response, dict) else []
        for item in results:
            if not isinstance(item, dict):
                continue

            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue

            publisher = str(item.get("source") or _publisher_from_url(url)).strip() or _publisher_from_url(url)
            collected.append(
                {
                    "title": title,
                    "publisher": publisher,
                    "date": str(item.get("published_date") or item.get("publishedDate") or "").strip() or None,
                    "url": url,
                    "snippet": str(item.get("content") or "").strip() or None,
                    "query_category": query_info["query_category"],
                    "relevance_bucket": (
                        "broader_context"
                        if query_info["query_category"] == "broad_company_news"
                        else "high_confidence_company_specific"
                    ),
                }
            )

    if not collected and query_failures:
        return {
            "symbol": row.ticker,
            "ok": False,
            "error": "All Tavily company-news queries failed.",
            "query_failures": query_failures,
            "queries_used": list(build_default_retrieval_queries(row)),
        }

    deduped = _dedupe_news_items(collected)
    return {
        "symbol": row.ticker,
        "ok": True,
        "news": deduped[:max_items],
        "returned_count": min(len(deduped), max_items),
        "query_failures": query_failures,
        "queries_used": list(build_default_retrieval_queries(row)),
    }


def build_default_retrieval_queries(row: WatchlistRow) -> tuple[dict[str, str], ...]:
    company_label = f"{row.name} ({row.ticker})"
    return (
        {
            "query_category": "broad_company_news",
            "query": f"{company_label} latest company news",
        },
        {
            "query_category": "strategic_regulatory_monitoring",
            "query": (
                f"{company_label} strategy regulatory legal antitrust partnership "
                "product launch guidance"
            ),
        },
    )


def _publisher_from_url(url: str) -> str:
    hostname = urlparse(url).netloc.lower()
    return hostname.removeprefix("www.") or "Unknown source"


def _source_domain(record: EvidenceRecord) -> str:
    hostname = urlparse(record.url).netloc.lower()
    return hostname.removeprefix("www.")


def _dedupe_news_items(news_items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for item in news_items:
        url_key = str(item.get("url") or "").strip().lower()
        title_key = str(item.get("title") or "").strip().lower()
        key = url_key or title_key
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(item)

    return deduped


def retrieve_recent_developments(
    row: WatchlistRow,
    retrieval_fn: IssuerRetrievalFn | None = None,
    max_items: int = MAX_RECENT_ITEMS_PER_ISSUER,
) -> dict[str, Any]:
    active_retrieval_fn = retrieval_fn or _default_retrieval_fn

    try:
        result = active_retrieval_fn(row, max_items)
    except Exception as exc:
        return {
            "symbol": row.ticker,
            "ok": False,
            "error": f"Retrieval exception: {exc}",
        }

    if not isinstance(result, dict):
        return {
            "symbol": row.ticker,
            "ok": False,
            "error": "Retrieval returned a non-dict result.",
        }

    return result


def _normalize_evidence_records(news_items: Any) -> tuple[EvidenceRecord, ...]:
    if not isinstance(news_items, list):
        return ()

    evidence: list[EvidenceRecord] = []
    for item in news_items:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        if not title or not url:
            continue

        publisher = str(item.get("publisher") or "Unknown source").strip() or "Unknown source"
        date = str(item.get("date")).strip() if item.get("date") else None
        snippet = str(item.get("snippet")).strip() if item.get("snippet") else None
        query_category = str(item.get("query_category")).strip() if item.get("query_category") else None
        relevance_bucket = str(item.get("relevance_bucket")).strip() if item.get("relevance_bucket") else None

        evidence.append(
            EvidenceRecord(
                title=title,
                publisher=publisher,
                date=date,
                url=url,
                snippet=snippet,
                query_category=query_category,
                relevance_bucket=relevance_bucket,
            )
        )

    return tuple(evidence[:MAX_RECENT_ITEMS_PER_ISSUER])


def _tokenize_text(value: str) -> tuple[str, ...]:
    return tuple(TEXT_TOKEN_PATTERN.findall(value.lower()))


def _company_name_tokens(name: str) -> tuple[str, ...]:
    tokens = [
        token
        for token in _tokenize_text(name)
        if token not in COMPANY_TOKEN_STOPWORDS and len(token) >= 3
    ]
    return tuple(_dedupe_preserve_order(tokens))


def _monitoring_brief_tokens(row: WatchlistRow) -> tuple[str, ...]:
    tokens = [
        token
        for token in _tokenize_text(f"{row.thesis} {row.notes}")
        if token not in COMPANY_TOKEN_STOPWORDS
        and token not in MONITORING_TOKEN_STOPWORDS
        and len(token) >= 5
    ]
    return tuple(_dedupe_preserve_order(tokens))


def _record_text(record: EvidenceRecord) -> str:
    return " ".join(part for part in (record.title, record.snippet or "", record.url) if part)


def _ticker_match_for_row(row: WatchlistRow, record: EvidenceRecord) -> bool:
    ticker = row.ticker.lower()
    title_tokens = set(_tokenize_text(record.title))
    text_tokens = set(_tokenize_text(_record_text(record)))

    if len(ticker) <= 3:
        return ticker in title_tokens
    return ticker in text_tokens


def _is_generic_roundup(record: EvidenceRecord) -> bool:
    text = _record_text(record).lower()
    return any(hint in text for hint in GENERIC_ROUNDUP_HINTS)


def _record_has_strong_match(row: WatchlistRow, record: EvidenceRecord) -> bool:
    text = _record_text(record)
    text_tokens = set(_tokenize_text(text))
    normalized_text = " ".join(_tokenize_text(text))
    ticker_match = _ticker_match_for_row(row, record)
    company_tokens = _company_name_tokens(row.name)
    matched_tokens = [token for token in company_tokens if token in text_tokens]
    full_name_match = " ".join(company_tokens) in normalized_text if company_tokens else False

    if _is_generic_roundup(record):
        return False

    return (
        ticker_match
        or full_name_match
        or len(matched_tokens) >= 2
        or any(len(token) >= 6 for token in matched_tokens)
    )


def _record_has_contextual_match(row: WatchlistRow, record: EvidenceRecord) -> bool:
    text = _record_text(record)
    text_tokens = set(_tokenize_text(text))
    ticker_match = _ticker_match_for_row(row, record)
    company_tokens = _company_name_tokens(row.name)
    matched_tokens = [token for token in company_tokens if token in text_tokens]

    return bool(ticker_match or matched_tokens)


def _downgrade_to_contextual(record: EvidenceRecord) -> EvidenceRecord:
    return EvidenceRecord(
        title=record.title,
        publisher=record.publisher,
        date=record.date,
        url=record.url,
        snippet=record.snippet,
        query_category=record.query_category,
        relevance_bucket="broader_context",
    )


def _filter_evidence_for_row(
    row: WatchlistRow,
    evidence: Sequence[EvidenceRecord],
) -> tuple[tuple[EvidenceRecord, ...], bool, int]:
    strong_matches: list[EvidenceRecord] = []
    contextual_matches: list[EvidenceRecord] = []
    rejected_count = 0

    for record in evidence:
        if _record_has_strong_match(row, record):
            strong_matches.append(record)
            continue

        if _record_has_contextual_match(row, record):
            contextual_matches.append(_downgrade_to_contextual(record))
            continue

        rejected_count += 1

    if strong_matches:
        rejected_count += len(contextual_matches)
        return tuple(strong_matches[:MAX_RECENT_ITEMS_PER_ISSUER]), False, rejected_count

    if contextual_matches:
        retained_contextual = tuple(contextual_matches[:1])
        rejected_count += max(0, len(contextual_matches) - len(retained_contextual))
        return retained_contextual, True, rejected_count

    return (), False, rejected_count


def _normalize_query_failures(raw_query_failures: Any) -> tuple[str, ...]:
    if not isinstance(raw_query_failures, list):
        return ()

    normalized: list[str] = []
    for item in raw_query_failures:
        if isinstance(item, dict):
            category = str(item.get("query_category") or "unknown").strip()
            error = str(item.get("error") or "unknown error").strip()
            normalized.append(f"{category}: {error}")
        elif item:
            normalized.append(str(item).strip())

    return tuple(item for item in normalized if item)


def _ensure_terminal_period(text: str) -> str:
    if text.endswith((".", "!", "?")):
        return text
    return f"{text}."


def _summarize_recent_developments(
    evidence: Sequence[EvidenceRecord],
    retrieval_ok: bool,
    retrieval_error: str | None,
) -> str:
    if not retrieval_ok:
        return f"Recent issuer retrieval failed: {_ensure_terminal_period(retrieval_error or 'unknown error')}"

    if not evidence:
        return "No recent issuer-specific developments were retrieved."

    highlighted_titles = [record.title for record in evidence[:2]]
    if len(highlighted_titles) == 1:
        return f"Recent coverage highlighted: {highlighted_titles[0]}."

    return f"Recent coverage highlighted: {highlighted_titles[0]}; {highlighted_titles[1]}."


def _is_low_confidence_source(record: EvidenceRecord) -> bool:
    return _source_domain(record) in LOW_CONFIDENCE_SOURCE_DOMAINS


def _matches_monitoring_brief(row: WatchlistRow, record: EvidenceRecord) -> bool:
    monitoring_tokens = _monitoring_brief_tokens(row)
    if not monitoring_tokens:
        return False

    text_tokens = set(_tokenize_text(_record_text(record)))
    return any(token in text_tokens for token in monitoring_tokens)


def _material_records(row: WatchlistRow, evidence: Sequence[EvidenceRecord]) -> tuple[EvidenceRecord, ...]:
    material: list[EvidenceRecord] = []
    for record in evidence:
        if record.relevance_bucket != "high_confidence_company_specific":
            continue
        if _is_low_confidence_source(record):
            continue
        if record.query_category == "strategic_regulatory_monitoring" or _matches_monitoring_brief(row, record):
            material.append(record)

    return tuple(material)


def _derive_evidence_quality_note(
    row: WatchlistRow,
    *,
    retrieval_ok: bool,
    retrieval_error: str | None,
    filtered_evidence: Sequence[EvidenceRecord],
    contextual_only: bool,
    rejected_count: int,
    query_failures: Sequence[str],
) -> str:
    if not retrieval_ok:
        return f"Retrieval failed: {retrieval_error or 'unknown error'}"

    material_records = _material_records(row, filtered_evidence)

    if not filtered_evidence:
        note = "No sufficiently relevant issuer-specific evidence remained after local relevance filtering."
    elif contextual_only:
        note = "Contextual issuer evidence only: generic or low-signal coverage remained."
    elif len(material_records) >= 2 and len(material_records) == len(filtered_evidence):
        note = "Relevant issuer-specific evidence found across multiple items."
    elif len(material_records) >= 2:
        note = "Mixed issuer evidence: multiple materially relevant items were retained, alongside lower-signal coverage."
    elif len(material_records) == 1 and len(filtered_evidence) == 1:
        note = "Sparse issuer-specific evidence: one materially relevant item remained after local relevance filtering."
    elif len(material_records) == 1:
        note = "Mixed issuer evidence: one materially relevant item was retained alongside weaker coverage."
    elif len(filtered_evidence) == 1:
        note = "Sparse issuer-specific evidence: one weakly relevant item remained after local relevance filtering."
    else:
        note = "Weak issuer evidence: issuer match was retained, but material relevance to the monitoring brief is limited."

    if rejected_count > 0 and filtered_evidence:
        note = f"{note} Weak or wrong-entity items were removed."
    if query_failures:
        note = f"{note} Partial query failures occurred."
    return note


def _quality_is_limited(evidence_quality_note: str) -> bool:
    return any(
        evidence_quality_note.startswith(prefix) for prefix in LIMITED_QUALITY_PREFIXES
    )


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def determine_attention(review: IssuerReview) -> tuple[bool, str, str]:
    material_records = _material_records(
        WatchlistRow(
            ticker=review.ticker,
            name=review.name,
            weight=review.weight,
            thesis=review.thesis,
            risk_bucket=review.risk_bucket,
            region=review.region,
            priority=review.priority,
            notes=review.notes,
        ),
        review.evidence,
    )

    if not review.retrieval_ok:
        return False, "Gap", "Retrieval failed."
    if not review.evidence:
        if review.filtered_out_count > 0:
            return False, "Gap", "No relevant issuer-specific evidence remained after filtering."
        return False, "Gap", "No relevant issuer-specific evidence was retained."
    if review.contextual_evidence_only:
        return False, "Gap", "Contextual coverage only."
    if review.evidence_quality_note.startswith("Sparse issuer-specific evidence"):
        return False, "Gap", "Sparse evidence."
    if review.evidence_quality_note.startswith("Weak issuer evidence"):
        return False, "Gap", "Weak evidence quality."
    if len(material_records) >= 2:
        lead_category = material_records[0].query_category or "issuer-specific"
        category_label = lead_category.replace("_", " ")
        return True, "Attention", f"Multiple relevant {category_label} items retained."
    if len(material_records) == 1:
        return False, "Routine", "One relevant item retained; continue routine monitoring."
    return False, "Routine", "Issuer-specific coverage retained, but no elevated monitoring signal was identified."


def build_issuer_review(row: WatchlistRow, retrieval_result: dict[str, Any]) -> IssuerReview:
    retrieval_ok = bool(retrieval_result.get("ok"))
    retrieval_error = (
        str(retrieval_result.get("error")).strip()
        if retrieval_result.get("error")
        else None
    )
    query_failures = _normalize_query_failures(retrieval_result.get("query_failures"))
    raw_evidence = _normalize_evidence_records(retrieval_result.get("news"))
    evidence, contextual_only, rejected_count = _filter_evidence_for_row(row, raw_evidence)
    evidence_quality_note = _derive_evidence_quality_note(
        row,
        retrieval_ok=retrieval_ok,
        retrieval_error=retrieval_error,
        filtered_evidence=evidence,
        contextual_only=contextual_only,
        rejected_count=rejected_count,
        query_failures=query_failures,
    )

    base_review = IssuerReview(
        ticker=row.ticker,
        name=row.name,
        weight=row.weight,
        thesis=row.thesis,
        risk_bucket=row.risk_bucket,
        region=row.region,
        priority=row.priority,
        notes=row.notes,
        recent_development_summary=_summarize_recent_developments(
            evidence,
            retrieval_ok=retrieval_ok,
            retrieval_error=retrieval_error,
        ),
        evidence=evidence,
        evidence_quality_note=evidence_quality_note,
        requires_attention=False,
        attention_reason="routine monitoring only",
        retrieval_ok=retrieval_ok,
        retrieval_error=retrieval_error,
        query_failures=query_failures,
        contextual_evidence_only=contextual_only,
        filtered_out_count=rejected_count,
        status="Routine",
        status_note="Routine monitoring.",
    )

    requires_attention, status, status_note = determine_attention(base_review)
    return IssuerReview(
        ticker=base_review.ticker,
        name=base_review.name,
        weight=base_review.weight,
        thesis=base_review.thesis,
        risk_bucket=base_review.risk_bucket,
        region=base_review.region,
        priority=base_review.priority,
        notes=base_review.notes,
        recent_development_summary=base_review.recent_development_summary,
        evidence=base_review.evidence,
        evidence_quality_note=base_review.evidence_quality_note,
        requires_attention=requires_attention,
        attention_reason=status_note,
        retrieval_ok=base_review.retrieval_ok,
        retrieval_error=base_review.retrieval_error,
        query_failures=base_review.query_failures,
        contextual_evidence_only=base_review.contextual_evidence_only,
        filtered_out_count=base_review.filtered_out_count,
        status=status,
        status_note=status_note,
    )


def order_issuer_reviews(reviews: Sequence[IssuerReview]) -> list[IssuerReview]:
    return sorted(
        reviews,
        key=lambda review: (review.priority, -review.weight, review.ticker),
    )


def build_issuer_reviews(
    rows: Sequence[WatchlistRow],
    retrieval_fn: IssuerRetrievalFn | None = None,
    max_items: int = MAX_RECENT_ITEMS_PER_ISSUER,
) -> list[IssuerReview]:
    reviews: list[IssuerReview] = []
    for row in rows:
        retrieval_result = retrieve_recent_developments(
            row,
            retrieval_fn=retrieval_fn,
            max_items=max_items,
        )
        reviews.append(build_issuer_review(row, retrieval_result))

    return order_issuer_reviews(reviews)


def select_names_requiring_attention(reviews: Sequence[IssuerReview]) -> list[IssuerReview]:
    return [review for review in order_issuer_reviews(reviews) if review.status == "Attention"]


def _format_evidence_list(evidence: Sequence[EvidenceRecord]) -> str:
    if not evidence:
        return "none"

    return "; ".join(
        f"{record.publisher} ({record.date or 'undated'}): {record.title}"
        for record in evidence
    )


def build_names_requiring_attention_section(reviews: Sequence[IssuerReview]) -> str:
    lines = ["## Names Requiring Attention"]
    flagged_reviews = select_names_requiring_attention(reviews)

    if not flagged_reviews:
        lines.append("- No issuers currently in Attention status.")
        return "\n".join(lines)

    for review in flagged_reviews:
        lines.append(
            f"- {review.ticker} ({review.name}) | {review.weight * 100:.1f}% | "
            f"Priority {review.priority} | {review.status_note}"
        )

    return "\n".join(lines)


def build_issuer_event_review_section(reviews: Sequence[IssuerReview]) -> str:
    lines = ["## Issuer-Event Review"]

    for review in order_issuer_reviews(reviews):
        lines.extend(
            [
                f"**{review.ticker} | {review.name} | {review.weight * 100:.1f}% | Priority {review.priority} | Status: {review.status}**",
                f"Thesis: {review.thesis}",
                f"Monitoring note: {review.notes or 'none'}",
                f"Recent developments: {review.recent_development_summary}",
                f"Evidence quality: {review.evidence_quality_note}",
                f"Sources: {_format_evidence_list(review.evidence)}",
            ]
        )
        if review.status_note:
            lines.append(f"Status note: {review.status_note}")
        lines.append("")

    return "\n".join(lines).rstrip()


def build_monitoring_gaps_evidence_limitations_section(reviews: Sequence[IssuerReview]) -> str:
    lines = ["## Monitoring Gaps / Evidence Limitations"]
    gaps: list[str] = []

    for review in order_issuer_reviews(reviews):
        gap_parts: list[str] = []

        if not review.retrieval_ok:
            gap_parts.append("retrieval failed")
        elif not review.evidence:
            if review.filtered_out_count > 0:
                gap_parts.append("no relevant issuer-specific evidence remained after filtering")
                gap_parts.append("wrong-entity or weak contamination was removed")
            else:
                gap_parts.append("no relevant issuer-specific evidence was retained")
            if review.query_failures:
                gap_parts.append("partial query failures occurred")
        else:
            if review.contextual_evidence_only:
                gap_parts.append("generic contextual evidence only")
            if review.evidence_quality_note.startswith("Sparse issuer-specific evidence"):
                gap_parts.append("sparse evidence")
            if review.evidence_quality_note.startswith("Weak issuer evidence"):
                gap_parts.append("weak evidence quality")
            if review.query_failures:
                gap_parts.append("partial query failures occurred")

        if gap_parts:
            gaps.append(f"{review.ticker}: {'; '.join(_dedupe_preserve_order(gap_parts))}.")

    deduped_gaps = _dedupe_preserve_order(gaps)
    if not deduped_gaps:
        lines.append("- No material evidence gaps were identified in this bounded pass.")
        return "\n".join(lines)

    lines.extend(f"- {gap}" for gap in deduped_gaps)
    return "\n".join(lines)


def build_monitoring_report(
    rows: Sequence[WatchlistRow],
    retrieval_fn: IssuerRetrievalFn | None = None,
) -> str:
    ordered_rows = order_watchlist_rows(rows)
    reviews = build_issuer_reviews(ordered_rows, retrieval_fn=retrieval_fn)

    sections = [
        build_portfolio_watchlist_summary(ordered_rows),
        build_names_requiring_attention_section(reviews),
        build_issuer_event_review_section(reviews),
        build_monitoring_gaps_evidence_limitations_section(reviews),
    ]
    return "\n\n".join(sections)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        rows = load_watchlist_csv(args.input)
    except WatchlistValidationError as exc:
        if len(exc.errors) == 1:
            print(f"Error: {exc.errors[0]}", file=sys.stderr)
        else:
            print("Error:", file=sys.stderr)
            for error in exc.errors:
                print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)

    print(build_monitoring_report(rows))


if __name__ == "__main__":
    main()
