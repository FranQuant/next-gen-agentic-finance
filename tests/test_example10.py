import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.example10 import (  # noqa: E402
    CANONICAL_SCHEMA,
    DEFAULT_QUERY_CATEGORIES,
    MAX_RECENT_ITEMS_PER_ISSUER,
    IssuerReview,
    WatchlistRow,
    WatchlistValidationError,
    build_default_retrieval_queries,
    build_issuer_reviews,
    build_monitoring_report,
    build_portfolio_watchlist_summary,
    load_watchlist_csv,
    order_watchlist_rows,
    select_names_requiring_attention,
)


def write_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def make_row(
    ticker: str,
    *,
    name: str | None = None,
    weight: float = 0.5,
    thesis: str = "Bounded demo thesis",
    risk_bucket: str = "High",
    region: str = "LatAm",
    priority: int = 1,
    notes: str = "",
) -> WatchlistRow:
    return WatchlistRow(
        ticker=ticker,
        name=name or f"{ticker} Corp",
        weight=weight,
        thesis=thesis,
        risk_bucket=risk_bucket,
        region=region,
        priority=priority,
        notes=notes,
    )


def test_load_watchlist_csv_accepts_canonical_file() -> None:
    rows = load_watchlist_csv(ROOT / "data" / "example10_watchlist.csv")

    assert len(rows) == 6
    assert rows[0].ticker == "AAPL"
    assert rows[-1].ticker == "META"
    assert sum(row.weight for row in rows) == pytest.approx(1.0)


def test_load_watchlist_csv_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(WatchlistValidationError, match="Input file not found"):
        load_watchlist_csv(tmp_path / "missing.csv")


def test_load_watchlist_csv_rejects_header_mismatch(tmp_path: Path) -> None:
    csv_path = tmp_path / "watchlist.csv"
    write_csv(
        csv_path,
        ("ticker", "name", "bad_column"),
        [["NU", "Nu Holdings", "x"]],
    )

    with pytest.raises(WatchlistValidationError, match="Header mismatch"):
        load_watchlist_csv(csv_path)


def test_load_watchlist_csv_rejects_duplicate_ticker(tmp_path: Path) -> None:
    csv_path = tmp_path / "watchlist.csv"
    write_csv(
        csv_path,
        CANONICAL_SCHEMA,
        [
            ["NU", "Nu Holdings", "0.5", "Fintech", "High", "Brazil", "1", ""],
            ["NU", "Nu Holdings", "0.5", "Fintech", "High", "Brazil", "2", ""],
        ],
    )

    with pytest.raises(WatchlistValidationError, match="duplicate ticker"):
        load_watchlist_csv(csv_path)


def test_load_watchlist_csv_rejects_invalid_weight(tmp_path: Path) -> None:
    csv_path = tmp_path / "watchlist.csv"
    write_csv(
        csv_path,
        CANONICAL_SCHEMA,
        [
            ["NU", "Nu Holdings", "1.2", "Fintech", "High", "Brazil", "1", ""],
        ],
    )

    with pytest.raises(WatchlistValidationError, match="weight must be > 0 and <= 1"):
        load_watchlist_csv(csv_path)


def test_load_watchlist_csv_rejects_bad_total_weight(tmp_path: Path) -> None:
    csv_path = tmp_path / "watchlist.csv"
    write_csv(
        csv_path,
        CANONICAL_SCHEMA,
        [
            ["NU", "Nu Holdings", "0.4", "Fintech", "High", "Brazil", "1", ""],
            ["MELI", "MercadoLibre", "0.5", "Platform", "High", "LatAm", "2", ""],
        ],
    )

    with pytest.raises(WatchlistValidationError, match="Total weight must sum to 1.0"):
        load_watchlist_csv(csv_path)


def test_load_watchlist_csv_rejects_blank_required_field(tmp_path: Path) -> None:
    csv_path = tmp_path / "watchlist.csv"
    write_csv(
        csv_path,
        CANONICAL_SCHEMA,
        [
            ["NU", "", "1.0", "Fintech", "High", "Brazil", "1", ""],
        ],
    )

    with pytest.raises(WatchlistValidationError, match="name is required"):
        load_watchlist_csv(csv_path)


def test_load_watchlist_csv_allows_blank_notes_value(tmp_path: Path) -> None:
    csv_path = tmp_path / "watchlist.csv"
    write_csv(
        csv_path,
        CANONICAL_SCHEMA,
        [
            ["NU", "Nu Holdings", "1.0", "Fintech", "High", "Brazil", "1", ""],
        ],
    )

    rows = load_watchlist_csv(csv_path)

    assert len(rows) == 1
    assert rows[0].notes == ""


def test_load_watchlist_csv_rejects_invalid_priority(tmp_path: Path) -> None:
    csv_path = tmp_path / "watchlist.csv"
    write_csv(
        csv_path,
        CANONICAL_SCHEMA,
        [
            ["NU", "Nu Holdings", "1.0", "Fintech", "High", "Brazil", "0", ""],
        ],
    )

    with pytest.raises(WatchlistValidationError, match="priority must be an integer >= 1"):
        load_watchlist_csv(csv_path)


def test_order_watchlist_rows_sorts_priority_then_weight() -> None:
    rows = [
        WatchlistRow("BBB", "Name B", 0.2, "Thesis", "Medium", "LatAm", 2, ""),
        WatchlistRow("AAA", "Name A", 0.1, "Thesis", "Medium", "LatAm", 1, ""),
        WatchlistRow("CCC", "Name C", 0.3, "Thesis", "Medium", "LatAm", 1, ""),
    ]

    ordered = order_watchlist_rows(rows)

    assert [row.ticker for row in ordered] == ["CCC", "AAA", "BBB"]


def test_build_portfolio_watchlist_summary_outputs_expected_basics() -> None:
    rows = load_watchlist_csv(ROOT / "data" / "example10_watchlist.csv")

    summary = build_portfolio_watchlist_summary(order_watchlist_rows(rows))

    assert "## Portfolio / Watchlist Summary" in summary
    assert "- Names: 6" in summary
    assert "- Total weight: 100.0%" in summary
    assert "- Top weights: MSFT 20.0%, AAPL 18.0%, NVDA 18.0%" in summary
    assert "- Priority mix: P1: 3, P2: 3" in summary
    assert "- Region mix: US: 6" in summary
    assert "- Risk mix:" in summary


def test_build_default_retrieval_queries_uses_two_default_categories() -> None:
    queries = build_default_retrieval_queries(make_row("AAPL", name="Apple Inc", weight=1.0))

    assert len(queries) == 2
    assert [query["query_category"] for query in queries] == list(DEFAULT_QUERY_CATEGORIES)


def test_build_issuer_reviews_uses_stubbed_retrieval_and_returns_structured_reviews() -> None:
    rows = [make_row("NU", name="Nu Holdings", weight=1.0, notes="Core monitoring name")]
    calls: list[tuple[str, int]] = []

    def stub_retrieval(row: WatchlistRow, max_items: int) -> dict:
        calls.append((row.ticker, max_items))
        return {
            "symbol": row.ticker,
            "ok": True,
            "news": [
                {
                    "title": "Nu Holdings launches new credit product",
                    "publisher": "Reuters",
                    "date": "2026-03-15",
                    "url": "https://example.com/nu-product",
                    "snippet": "Nu expanded its consumer credit offering.",
                    "query_category": "product_strategy",
                    "relevance_bucket": "high_confidence_company_specific",
                }
            ],
            "news_quality_note": "Strong company-specific coverage found.",
        }

    reviews = build_issuer_reviews(rows, retrieval_fn=stub_retrieval)

    assert calls == [("NU", MAX_RECENT_ITEMS_PER_ISSUER)]
    assert len(reviews) == 1
    assert isinstance(reviews[0], IssuerReview)
    assert reviews[0].ticker == "NU"
    assert reviews[0].evidence[0].title == "Nu Holdings launches new credit product"
    assert reviews[0].evidence_quality_note.startswith("Sparse issuer-specific evidence:")
    assert reviews[0].status == "Gap"
    assert reviews[0].requires_attention is False
    assert reviews[0].status_note == "Sparse evidence."


def test_select_names_requiring_attention_returns_attention_status_only() -> None:
    rows = [
        make_row("AAA", weight=0.3, priority=2),
        make_row("BBB", weight=0.4, priority=1),
        make_row("CCC", weight=0.3, priority=1),
    ]

    def stub_retrieval(row: WatchlistRow, max_items: int) -> dict:
        if row.ticker == "AAA":
            return {
                "symbol": row.ticker,
                "ok": True,
                "news": [
                    {
                        "title": "AAA expands regulatory filing program",
                        "publisher": "Reuters",
                        "date": "2026-03-14",
                        "url": "https://example.com/aaa-1",
                        "snippet": "AAA expanded its regulatory filing program and capital plan.",
                        "query_category": "strategic_regulatory_monitoring",
                        "relevance_bucket": "high_confidence_company_specific",
                    },
                    {
                        "title": "AAA announces strategic partnership",
                        "publisher": "Bloomberg",
                        "date": "2026-03-12",
                        "url": "https://example.com/aaa-2",
                        "snippet": "AAA announced a strategic partnership tied to capital deployment.",
                        "query_category": "strategic_regulatory_monitoring",
                        "relevance_bucket": "high_confidence_company_specific",
                    },
                ],
            }
        if row.ticker == "BBB":
            return {
                "symbol": row.ticker,
                "ok": True,
                "news": [
                    {
                        "title": "BBB launches new credit product",
                        "publisher": "Reuters",
                        "date": "2026-03-13",
                        "url": "https://example.com/bbb-1",
                        "snippet": "BBB launched a new credit product tied to growth.",
                        "query_category": "strategic_regulatory_monitoring",
                        "relevance_bucket": "high_confidence_company_specific",
                    },
                    {
                        "title": "BBB mentioned in market outlook",
                        "publisher": "TipRanks",
                        "date": "2026-03-11",
                        "url": "https://example.com/bbb-2",
                        "snippet": "BBB was mentioned in a broader market outlook.",
                        "query_category": "broad_company_news",
                        "relevance_bucket": "high_confidence_company_specific",
                    },
                ],
            }
        return {
            "symbol": row.ticker,
            "ok": False,
            "error": "simulated retrieval failure",
        }

    reviews = build_issuer_reviews(rows, retrieval_fn=stub_retrieval)
    flagged = select_names_requiring_attention(reviews)

    assert [review.ticker for review in flagged] == ["AAA"]
    assert [review.status for review in reviews] == ["Routine", "Gap", "Attention"]


def test_build_monitoring_report_contains_all_required_sections() -> None:
    rows = [
        make_row("NU", name="Nu Holdings", weight=0.5, priority=1),
        make_row("MELI", name="MercadoLibre", weight=0.5, priority=2),
    ]

    def stub_retrieval(row: WatchlistRow, max_items: int) -> dict:
        return {
            "symbol": row.ticker,
            "ok": True,
            "news": [
                {
                    "title": f"{row.name} recent update",
                    "publisher": "Reuters",
                    "date": "2026-03-10",
                    "url": f"https://example.com/{row.ticker.lower()}",
                    "snippet": "Recent issuer-specific development.",
                    "query_category": "broad_company_news",
                    "relevance_bucket": "broader_context",
                }
            ],
            "news_quality_note": "Mixed result set: company-specific items found, with some contextual coverage retained.",
        }

    report = build_monitoring_report(rows, retrieval_fn=stub_retrieval)

    assert "## Portfolio / Watchlist Summary" in report
    assert "## Names Requiring Attention" in report
    assert "## Issuer-Event Review" in report
    assert "## Monitoring Gaps / Evidence Limitations" in report
    assert "## Source Quality / Limitations" not in report
    assert report.count("## Monitoring Gaps / Evidence Limitations") == 1


def test_build_monitoring_report_surfaces_sparse_and_failed_retrieval_cleanly() -> None:
    rows = [
        make_row("NU", name="Nu Holdings", weight=0.6, priority=1),
        make_row("PBR", name="Petrobras", weight=0.4, priority=2),
    ]

    def stub_retrieval(row: WatchlistRow, max_items: int) -> dict:
        if row.ticker == "NU":
            return {
                "symbol": row.ticker,
                "ok": False,
                "error": "TAVILY_API_KEY not set.",
            }

        return {
            "symbol": row.ticker,
            "ok": True,
            "news": [],
            "news_quality_note": "No sufficiently relevant company-news items found.",
            "query_failures": [
                {
                    "query_category": "management_commentary",
                    "error": "timeout",
                }
            ],
        }

    report = build_monitoring_report(rows, retrieval_fn=stub_retrieval)

    assert "Retrieval failed: TAVILY_API_KEY not set." in report
    assert "No recent issuer-specific developments were retrieved." in report
    assert "## Monitoring Gaps / Evidence Limitations" in report
    assert "- NU: retrieval failed." in report
    assert "- PBR: no relevant issuer-specific evidence was retained; partial query failures occurred." in report


def test_build_monitoring_report_uses_routine_monitoring_when_nothing_is_flagged() -> None:
    rows = [make_row("CIB", name="Bancolombia", weight=1.0, priority=2)]

    def stub_retrieval(row: WatchlistRow, max_items: int) -> dict:
        return {
            "symbol": row.ticker,
            "ok": True,
            "news": [
                {
                    "title": "Bancolombia mentioned in regional market wrap",
                    "publisher": "Reuters",
                    "date": "2026-03-12",
                    "url": "https://example.com/cib-wrap",
                    "snippet": "Regional market context.",
                    "query_category": "broad_company_news",
                    "relevance_bucket": "broader_context",
                }
            ],
            "news_quality_note": "Mixed result set: company-specific items found, with some contextual coverage retained.",
        }

    report = build_monitoring_report(rows, retrieval_fn=stub_retrieval)

    assert "## Names Requiring Attention\n- No issuers currently in Attention status." in report


def test_wrong_entity_match_is_rejected() -> None:
    rows = [make_row("GGAL", name="Grupo Financiero Galicia", weight=1.0, priority=2)]

    def stub_retrieval(row: WatchlistRow, max_items: int) -> dict:
        return {
            "symbol": row.ticker,
            "ok": True,
            "news": [
                {
                    "title": "Spain's Grupo Gallo takes broths into US retail",
                    "publisher": "Just Food",
                    "date": "2026-03-12",
                    "url": "https://example.com/grupo-gallo",
                    "snippet": "Grupo Gallo expands its food distribution footprint.",
                    "query_category": "product_strategy",
                    "relevance_bucket": "high_confidence_company_specific",
                }
            ],
            "news_quality_note": "Strong company-specific coverage found.",
        }

    reviews = build_issuer_reviews(rows, retrieval_fn=stub_retrieval)

    assert reviews[0].evidence == ()
    assert reviews[0].evidence_quality_note.startswith(
        "No sufficiently relevant issuer-specific evidence remained"
    )
    assert reviews[0].requires_attention is False


def test_generic_market_roundup_is_downgraded_to_weak_contextual_evidence() -> None:
    rows = [make_row("MELI", name="MercadoLibre", weight=1.0, priority=2)]

    def stub_retrieval(row: WatchlistRow, max_items: int) -> dict:
        return {
            "symbol": row.ticker,
            "ok": True,
            "news": [
                {
                    "title": "Here are the 5 big things we're watching in the stock market this week",
                    "publisher": "CNBC",
                    "date": "2026-03-20",
                    "url": "https://example.com/meli-roundup",
                    "snippet": "MercadoLibre was mentioned among several regional names to watch.",
                    "query_category": "broad_company_news",
                    "relevance_bucket": "high_confidence_company_specific",
                }
            ],
            "news_quality_note": "Strong company-specific coverage found.",
        }

    reviews = build_issuer_reviews(rows, retrieval_fn=stub_retrieval)

    assert len(reviews[0].evidence) == 1
    assert reviews[0].evidence[0].relevance_bucket == "broader_context"
    assert reviews[0].evidence_quality_note.startswith("Contextual issuer evidence only")
    assert reviews[0].status == "Gap"
    assert reviews[0].requires_attention is False


def test_irrelevant_article_is_rejected() -> None:
    rows = [make_row("CIB", name="Bancolombia", weight=1.0, priority=2)]

    def stub_retrieval(row: WatchlistRow, max_items: int) -> dict:
        return {
            "symbol": row.ticker,
            "ok": True,
            "news": [
                {
                    "title": "Where design meets biology: an interview on SPIKA",
                    "publisher": "Nature",
                    "date": "2026-03-14",
                    "url": "https://example.com/cib-architecture",
                    "snippet": "A feature on microbe-mediated architecture and a non-issuer CIB acronym.",
                    "query_category": "management_commentary",
                    "relevance_bucket": "high_confidence_company_specific",
                }
            ],
            "news_quality_note": "Strong company-specific coverage found.",
        }

    reviews = build_issuer_reviews(rows, retrieval_fn=stub_retrieval)

    assert reviews[0].evidence == ()
    assert reviews[0].evidence_quality_note.startswith(
        "No sufficiently relevant issuer-specific evidence remained"
    )
    assert reviews[0].requires_attention is False


def test_weak_and_sparse_evidence_are_reported_in_limitations() -> None:
    rows = [
        make_row("MELI", name="MercadoLibre", weight=0.5, priority=2),
        make_row("GGAL", name="Grupo Financiero Galicia", weight=0.5, priority=2),
    ]

    def stub_retrieval(row: WatchlistRow, max_items: int) -> dict:
        if row.ticker == "MELI":
            return {
                "symbol": row.ticker,
                "ok": True,
                "news": [
                    {
                        "title": "Here are the 5 big things we're watching in the stock market this week",
                        "publisher": "CNBC",
                        "date": "2026-03-20",
                        "url": "https://example.com/meli-roundup",
                        "snippet": "MercadoLibre was mentioned among several regional names to watch.",
                        "query_category": "broad_company_news",
                        "relevance_bucket": "high_confidence_company_specific",
                    }
                ],
                "news_quality_note": "Strong company-specific coverage found.",
            }

        return {
            "symbol": row.ticker,
            "ok": True,
            "news": [
                {
                    "title": "Spain's Grupo Gallo takes broths into US retail",
                    "publisher": "Just Food",
                    "date": "2026-03-12",
                    "url": "https://example.com/grupo-gallo",
                    "snippet": "Grupo Gallo expands its food distribution footprint.",
                    "query_category": "product_strategy",
                    "relevance_bucket": "high_confidence_company_specific",
                }
            ],
            "news_quality_note": "Strong company-specific coverage found.",
        }

    report = build_monitoring_report(rows, retrieval_fn=stub_retrieval)

    assert "- MELI: generic contextual evidence only." in report
    assert "- GGAL: no relevant issuer-specific evidence remained after filtering; wrong-entity or weak contamination was removed." in report


def test_status_categories_cover_attention_routine_and_gap() -> None:
    rows = [
        make_row("AAA", name="AAA Holdings", weight=0.34, priority=1, notes="Monitor capital returns and regulatory pressure."),
        make_row("BBB", name="BBB Holdings", weight=0.33, priority=2, notes="Monitor capital returns and regulatory pressure."),
        make_row("CCC", name="CCC Holdings", weight=0.33, priority=2, notes="Monitor capital returns and regulatory pressure."),
    ]

    def stub_retrieval(row: WatchlistRow, max_items: int) -> dict:
        if row.ticker == "AAA":
            return {
                "symbol": row.ticker,
                "ok": True,
                "news": [
                    {
                        "title": "AAA expands capital returns plan",
                        "publisher": "Reuters",
                        "date": "2026-03-14",
                        "url": "https://example.com/aaa-1",
                        "snippet": "AAA expanded its capital returns plan after a regulatory review.",
                        "query_category": "strategic_regulatory_monitoring",
                        "relevance_bucket": "high_confidence_company_specific",
                    },
                    {
                        "title": "AAA faces new regulatory review",
                        "publisher": "Bloomberg",
                        "date": "2026-03-13",
                        "url": "https://example.com/aaa-2",
                        "snippet": "AAA is working through a new regulatory review.",
                        "query_category": "strategic_regulatory_monitoring",
                        "relevance_bucket": "high_confidence_company_specific",
                    },
                ],
            }
        if row.ticker == "BBB":
            return {
                "symbol": row.ticker,
                "ok": True,
                "news": [
                    {
                        "title": "BBB updates capital returns plan",
                        "publisher": "Reuters",
                        "date": "2026-03-12",
                        "url": "https://example.com/bbb-1",
                        "snippet": "BBB updated its capital returns plan.",
                        "query_category": "strategic_regulatory_monitoring",
                        "relevance_bucket": "high_confidence_company_specific",
                    },
                    {
                        "title": "BBB listed in broad market commentary",
                        "publisher": "TipRanks",
                        "date": "2026-03-11",
                        "url": "https://example.com/bbb-2",
                        "snippet": "BBB was listed in a broader market commentary piece.",
                        "query_category": "broad_company_news",
                        "relevance_bucket": "high_confidence_company_specific",
                    },
                ],
            }
        return {
            "symbol": row.ticker,
            "ok": True,
            "news": [
                {
                    "title": "Here are the 5 big things we're watching in the stock market this week",
                    "publisher": "CNBC",
                    "date": "2026-03-20",
                    "url": "https://example.com/ccc-roundup",
                    "snippet": "CCC was mentioned among several names to watch.",
                    "query_category": "broad_company_news",
                    "relevance_bucket": "high_confidence_company_specific",
                }
            ],
        }

    reviews = build_issuer_reviews(rows, retrieval_fn=stub_retrieval)

    assert [review.status for review in reviews] == ["Attention", "Routine", "Gap"]


def test_canonical_watchlist_builds_report_with_stubbed_retrieval() -> None:
    rows = load_watchlist_csv(ROOT / "data" / "example10_watchlist.csv")

    def stub_retrieval(row: WatchlistRow, max_items: int) -> dict:
        return {
            "symbol": row.ticker,
            "ok": True,
            "news": [
                {
                    "title": f"{row.name} update",
                    "publisher": "Reuters",
                    "date": "2026-03-10",
                    "url": f"https://example.com/{row.ticker.lower()}",
                    "snippet": f"{row.name} update tied to {row.notes}",
                    "query_category": "strategic_regulatory_monitoring",
                    "relevance_bucket": "high_confidence_company_specific",
                },
                {
                    "title": f"{row.name} second update",
                    "publisher": "Bloomberg",
                    "date": "2026-03-09",
                    "url": f"https://example.com/{row.ticker.lower()}-2",
                    "snippet": f"{row.name} second update tied to {row.thesis}",
                    "query_category": "strategic_regulatory_monitoring",
                    "relevance_bucket": "high_confidence_company_specific",
                },
            ],
        }

    report = build_monitoring_report(rows, retrieval_fn=stub_retrieval)

    assert "META | Meta Platforms Inc" in report
    assert "Status: Attention" in report
