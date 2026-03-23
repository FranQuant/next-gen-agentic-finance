import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.example10 import (  # noqa: E402
    CANONICAL_SCHEMA,
    WatchlistRow,
    WatchlistValidationError,
    build_portfolio_watchlist_summary,
    load_watchlist_csv,
    order_watchlist_rows,
)


def write_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_load_watchlist_csv_accepts_canonical_file() -> None:
    rows = load_watchlist_csv(ROOT / "data" / "example10_watchlist.csv")

    assert len(rows) == 6
    assert rows[0].ticker == "NU"
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
    assert "- Top weights: MELI 26.0%, NU 22.0%, SQM 15.0%" in summary
    assert "- Priority mix: P1: 3, P2: 3" in summary
    assert "- Region mix:" in summary
    assert "- Risk mix:" in summary
