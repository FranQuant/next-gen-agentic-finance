# Example 10 Design Note

## Scope
- Add `examples/example10.py` as a one-shot monitoring demo over a tiny local holdings/watchlist CSV.
- Load and validate `data/example10_watchlist.csv`, retrieve recent issuer developments for each row, and synthesize one structured monitoring report.
- Treat the CSV as the source of watchlist metadata; use recent public issuer-specific news as the live evidence layer.
- Reuse only the minimum realistic shared utility surface from `examples/finance_tools.py`, with `get_company_news_tavily` as the primary recent-issuer retrieval path.

## Non-goals
- Not an optimizer, risk engine, PM system, trade recommendation engine, or engine-repo design.
- No portfolio construction logic, scenario engine, target prices, valuation calls, or trade actions.
- No redesign of Example 9 or the shared utility layer beyond minimal bounded reuse.

## Canonical Input
- Path: `data/example10_watchlist.csv`
- Header, exact order: `ticker,name,weight,thesis,risk_bucket,region,priority,notes`
- Demo size: `1` to `15` rows.

## Validation Rules
- File must exist, be readable, and parse as UTF-8 CSV with a header row.
- Header must exactly match the canonical schema and order above.
- No blank rows.
- `ticker`: non-empty, uppercase, unique case-insensitively, regex `^[A-Z][A-Z0-9.-]*$`.
- `name`, `thesis`, `risk_bucket`, `region`: non-empty trimmed strings.
- `weight`: parseable float, `> 0`, `<= 1`, and total portfolio weight must sum to `1.0 +/- 0.001`.
- `priority`: integer `>= 1`.
- `notes`: column is required; value may be blank but must load as a string field.
- Collect all validation errors first; exit `1` before any live retrieval or model call if validation fails.

## Output Sections
- Final output is markdown only.
- Use exactly these top-level headings, in this order:
- `## Portfolio / Watchlist Summary`
- `## Names Requiring Attention`
- `## Issuer-Event Review`
- `## Source Quality / Limitations`
- `## Open Questions / Follow-Up`
- Under each heading, use single-level bullets only. No tables, JSON, nested bullets, or closing prose after the final section.
- `Portfolio / Watchlist Summary`: count of names, total weight, top weights, and simple priority / region / risk mix from the CSV.
- `Names Requiring Attention`: only names with a confirmed material recent issuer development, weak or mixed evidence on a high-priority name, or a clear tension between retrieved developments and the CSV thesis/notes. If none qualify, emit one bullet saying routine monitoring only.
- `Issuer-Event Review`: one bullet per input row, ordered by `priority` ascending then `weight` descending; each bullet must include ticker, name, and a concise recent-development read or a clear limited-signal note.
- `Source Quality / Limitations`: source strength, recency gaps, noisy or weak items, partial query failures, and demo limitations.
- `Open Questions / Follow-Up`: unresolved diligence questions that would materially improve monitoring clarity.

## CLI Behavior
- Run: `python examples/example10.py`
- Optional flag: `--input PATH` with default `data/example10_watchlist.csv`
- No interactive mode and no extra v1 CLI flags beyond `--input`.
- Required env: `OPENAI_API_KEY` and `TAVILY_API_KEY`
- Optional env: `EXAMPLE10_MODEL_ID`, falling back to `OPENAI_MODEL_ID`, then `gpt-5.4`
- On CSV, env, or bootstrap errors: write a concise error to stderr and exit `1`.
- Partial issuer-retrieval failures do not abort the run if at least one issuer packet is usable; surface the gap in `Source Quality / Limitations`.
- If all issuer retrievals fail, exit `1`.

## Test Checklist
- Add focused `pytest` coverage; the repo currently has no committed `tests/` tree, so Example 10 should introduce one narrowly for this demo.
- Loader / validator accepts the canonical CSV.
- Loader / validator rejects: missing file, header mismatch, duplicate ticker, invalid weight, bad total weight, blank required field, and invalid priority.
- Ordering helper sorts rows by `priority` ascending then `weight` descending.
- Evidence-packet assembly works with mocked Tavily results and preserves CSV metadata per name.
- CLI success test with mocked retrieval and mocked model output asserts the five exact section headings in order.
- CLI failure test asserts non-zero exit on invalid CSV.
- Manual smoke run on the canonical CSV with live keys produces the full five-section report.

## Closure Checklist
- `examples/example10.py` exists and stays within the bounded demo scope above.
- Canonical CSV is the default input and validation behaves as specified.
- Output matches the exact five-section contract and respects the non-goals.
- Automated tests above exist and pass.
- Manual smoke run succeeds on the canonical CSV.
- README is updated last with one bounded Example 10 entry.
