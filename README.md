# next-gen-agentic-finance

**Layer 1 examples for public/demo-data retrieval, bounded synthesis, and structured handoffs.**

Built with Agno, OpenAI, Tavily, yfinance, Python 3.13

---

## Context

This repository is a pedagogical Layer 1 repo for agentic finance research intelligence. It focuses on public/demo-data retrieval, bounded synthesis, and structured handoffs for human review.

It is intentionally limited in scope: it is not a full research stack, not a backtesting platform, not a portfolio orchestration system, and not an execution engine.

`example8.py` is the canonical structured handoff pattern. `example7.py` is the bridge example. `example9.py` is an adjacent MCP issuer-intelligence demo, not the repo's central contract.

---

## Architectural Contract

- `examples/example8.py` is the governing pattern: evidence packet, bounded interpretation, and open questions / gaps.
- `examples/example0_setup_check.py` through `examples/example6.py` are learning primitives and scaffolding.
- `examples/example7.py` is the bridge example for staged multi-agent research.
- `examples/example9.py` is adjacent MCP-based issuer intelligence, not the core contract.
- `examples/finance_tools.py` and `examples/news_filter.py` are shared read-only helpers, not a platform layer.
- Outputs are research handoffs for human review, not PM action notes.

## Governance Posture

- Read-only tools only.
- Public or demo data only.
- Human review required before any decision.
- No autonomous portfolio action.
- No execution, trading, or capital-allocation workflows in this repo.

---

## Stack

| Component | Role |
|---|---|
| [Agno](https://github.com/agno-agi/agno) | Agent and multi-agent orchestration framework |
| OpenAI Responses API | LLM backend for the examples |
| [Tavily](https://tavily.com) | Live web and news retrieval |
| yfinance | Market data, company fundamentals, analyst recommendation records, and options chains |
| DuckDB / CsvTools | SQL-style queries over local structured data |
| MCP (optional) | Remote Tavily transport used only by `example9.py` |

---

## Setup

Requires **Python 3.13** and [`uv`](https://docs.astral.sh/uv/).

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# Clone and install
git clone https://github.com/FranQuant/next-gen-agentic-finance.git
cd next-gen-agentic-finance

uv venv --python 3.13 .venv
source .venv/bin/activate
uv sync

# Optional: install the adjacent MCP demo extra for example9
uv sync --extra example9

# Verify installation
uv run examples/example0_setup_check.py
```

---

## Environment Variables

```bash
cp .env.example .env
```

| Variable | Required for |
|---|---|
| `OPENAI_API_KEY` | examples 1–9 |
| `TAVILY_API_KEY` | examples 2–4 and 7–9 |
| `EXAMPLE9_TAVILY_MCP_URL` | example9 (remote MCP) |

See `.env.example` for per-example model override options.

---

## Example Progression

The series builds one concept at a time. Each example adds exactly one new pattern.

| Example | Pattern | What it introduces |
|---|---|---|
| `example0_setup_check.py` | Sanity check | Verifies Agno installs correctly — no API calls |
| `example1.py` | Prompt-only agent | Structured sentiment analysis over static headlines — no tools |
| `example2.py` | Tool-enabled agent | Live news retrieval via Tavily + same sentiment rubric |
| `example3.py` | Debug / inspection | `debug_mode=True` — how to inspect tool calls and retrieval quality |
| `example4.py` | Interactive console | Turns the retrieval agent into an ad hoc analyst CLI |
| `example5.py` | Structured data agent | SQL-style queries over a local LatAm equities CSV via DuckDB |
| `example6.py` | Custom tool definition | Options max-pain computation: pure function → `@tool` → agent |
| `example7.py` | Bridge example | 3-agent research brief: market data → company info → orchestrator |
| `example8.py` | Canonical structured handoff | Evidence packet → interpretation → open questions; explicit gap tracking |
| `example9.py` | Adjacent MCP demo | Issuer intelligence via Agno + Tavily over async MCP transport |

### The learning arc

```
ex0–1   Static prompts, no tools          → understand the baseline
ex2–4   Live retrieval, debug, interactive → add tools, inspect behavior
ex5–6   Structured data + custom tools    → connect agents to real data sources
ex7     Bridge to staged multi-agent flow  → coordinate specialist agents
ex8     Canonical structured handoff      → structured evidence, read, and gaps
ex9     Adjacent MCP demo                  → issuer intelligence over async transport
```

---

## Shared Tool Layer

`examples/finance_tools.py` provides the shared finance utility functions used across examples:

- `get_current_stock_price` — price, OHLCV, session timing
- `get_analyst_recommendations` — recent analyst recommendation records and aggregate stance context
- `get_company_info` — sector, fundamentals, valuation ratios
- `get_company_news` — yfinance news feed
- `get_company_news_tavily` — deduplicated, quality-scored news via 4 sequential Tavily queries

This is a demo utility module. It is not a production SDK.

---

## Repository Structure

```text
next-gen-agentic-finance/
├── data/
│   └── latamstocks.csv          # LatAm equities dataset (example5)
├── examples/
│   ├── finance_tools.py         # Shared tool layer
│   ├── news_filter.py           # News scoring and filtering logic
│   ├── example0_setup_check.py
│   ├── example1.py  →  example9.py
├── README.md
├── pyproject.toml
└── uv.lock
```

---

## Where to Start

If you want the repo's governing pattern, start with `example8.py`.

If you want the full learning sequence, run `example0_setup_check.py` first, then go in order. The progression is intentional — each example assumes familiarity with the one before it.

If you want to jump to a specific pattern:
- **Canonical structured handoff** → `example8.py`
- **Bridge into multi-agent orchestration** → `example7.py`
- **Custom tool design** → `example6.py`
- **Adjacent MCP demo** → `example9.py`

---

## Disclaimer

This repository is for research and educational purposes only. Nothing in this codebase or its outputs constitutes investment advice. All examples are bounded demonstrations — not deployable investment infrastructure.
