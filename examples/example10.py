"""Example 10: bounded holdings/watchlist monitoring demo bootstrap."""

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
REQUIRED_NON_EMPTY_TEXT_FIELDS = ("ticker", "name", "thesis", "risk_bucket", "region")


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


def build_pass2_output(rows: Sequence[WatchlistRow]) -> str:
    return (
        f"{build_portfolio_watchlist_summary(rows)}\n\n"
        "Retrieval not yet implemented in this pass."
    )


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

    ordered_rows = order_watchlist_rows(rows)
    print(build_pass2_output(ordered_rows))


if __name__ == "__main__":
    main()
