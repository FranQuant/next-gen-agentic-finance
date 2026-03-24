# next-gen-agentic-finance

**A progressive example series for Layer 1 agentic finance: AI-powered research intelligence, live retrieval, multi-agent orchestration, and structured monitoring workflows.**

Built with Agno · OpenAI · Tavily · yfinance · Python 3.13

---

## Context

The current frontier of applied AI in finance is **agentic research** — agents that retrieve live market intelligence, reason over it, and surface structured findings for human review. This is happening in production at major institutions today.

This repository demonstrates that layer systematically, from a single prompt-only agent to multi-agent orchestration and issuer monitoring workflows. It is designed as **Layer 1 of a quant research stack**: intelligence gathering and synthesis that feeds human-supervised investment decisions.

It is not a trading system, portfolio optimizer, or execution engine. Those layers are downstream — and they still require human oversight before any capital action is taken.

---

## What this repository is / is not

**What it is**
- A focused, progressive demo series covering the core patterns of agentic finance research
- Designed to be readable, extensible, and honest about what agents can and cannot do reliably
- A foundation layer: research intelligence designed to inform, not to act

**What it is not**
- Not a production research platform, live trading system, or portfolio management tool
- Not investment advice
- Not a framework — it uses [Agno](https://github.com/agno-agi/agno) as the agent layer

---

## Stack

| Component | Role |
|---|---|
| [Agno](https://github.com/agno-agi/agno) | Agent and multi-agent orchestration framework |
| OpenAI (gpt-4o / o-series) | LLM backend via OpenAI Responses API |
| [Tavily](https://tavily.com) | Live web and news retrieval |
| yfinance | Market data, options chains, analyst recommendations |
| DuckDB / CsvTools | SQL-style queries over local structured data |
| MCP (remote) | Tavily via Model Context Protocol in example9 |

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
| `OPENAI_API_KEY` | All examples |
| `TAVILY_API_KEY` | example2 – example4, example9, example10 |
| `EXAMPLE9_TAVILY_MCP_URL` | example9 (remote MCP) |
| `FRED_API_KEY` | Reserved for future macro examples |

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
| `example7.py` | Multi-agent team | 3-agent research brief: market data → company info → orchestrator |
| `example8.py` | Structured handoff | Evidence packet → interpretation → open questions; explicit gap tracking |
| `example9.py` | Remote MCP | Issuer intelligence via Agno + Tavily over async MCP transport |
| `example10.py` | Watchlist monitoring | Load and validate a ticker watchlist → retrieve developments → structured alert report |

### The learning arc

```
ex0–1   Static prompts, no tools          → understand the baseline
ex2–4   Live retrieval, debug, interactive → add tools, inspect behavior
ex5–6   Structured data + custom tools    → connect agents to real data sources
ex7–8   Multi-agent orchestration         → coordinate specialist agents with JSON handoffs
ex9–10  MCP + monitoring workflows        → production-adjacent patterns with validation
```

---

## Shared Tool Layer

`examples/finance_tools.py` provides the shared finance utility functions used across examples:

- `get_current_stock_price` — price, OHLCV, session timing
- `get_analyst_recommendations` — buy/hold/sell counts, price targets
- `get_company_info` — sector, fundamentals, valuation ratios
- `get_company_news` — yfinance news feed
- `get_company_news_tavily` — deduplicated, quality-scored news via 4 parallel Tavily queries

This is a demo utility module. It is not a production SDK.

---

## Repository Structure

```text
next-gen-agentic-finance/
├── data/
│   └── latamstocks.csv          # LatAm equities dataset (example5)
├── examples/
│   ├── finance_tools.py         # Shared tool layer
│   ├── example0_setup_check.py
│   ├── example1.py  →  example10.py
├── README.md
├── pyproject.toml
└── uv.lock
```

---

## Where to Start

Run `example0_setup_check.py` first, then go in order. The progression is intentional — each example assumes familiarity with the one before it.

If you want to jump to a specific pattern:
- **Custom tool design** → `example6.py`
- **Multi-agent orchestration** → `example7.py`
- **MCP integration** → `example9.py`
- **Watchlist monitoring** → `example10.py`

---

## Roadmap

Patterns under consideration for future examples:

- Pydantic-validated structured output (replacing natural language output constraints)
- Async parallel tool execution
- Agent error recovery and tool fallback patterns
- Evaluation and testing of agent outputs
- Earnings call transcript analysis
- Macro indicator retrieval (FRED)

---

## Disclaimer

This repository is for research and educational purposes only. Nothing in this codebase or its outputs constitutes investment advice. All examples are bounded demonstrations — not deployable investment infrastructure.
