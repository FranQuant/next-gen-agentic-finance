# Agentic Finance Examples

A focused learning repository for agentic finance patterns: prompt engineering, live retrieval, retrieval debugging, interactive analyst workflows, structured data access, custom finance tools, multi-agent research orchestration, and structured research handoff design.

Built with:

- Agno agents and teams
- OpenAI Responses models
- Tavily web search
- yfinance market data
- local CSV / SQL-style querying for structured data examples

## What this repository is / is not

**What it is**
- A focused educational demos repository for agentic finance patterns, covering prompt engineering, live retrieval, retrieval debugging, structured data access, custom finance tools, multi-agent orchestration, and structured research handoff.
- Intended for learning, experimentation, and bounded prototype workflows.

**What it is not**
- Not a production research platform, live trading or execution system, portfolio optimizer, or OMS/EMS.
- Not investment advice; `example7.py` and `example8.py` should be understood as bounded prototype workflows, not deployable PM infrastructure.

## 1. Setup

This repository expects **Python 3.13** and uses `uv` for environment and dependency management.

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

Create the environment and install dependencies:

```bash
git clone https://github.com/FranQuant/next-gen-agentic-finance.git
cd next-gen-agentic-finance

uv venv --python 3.13 .venv
source .venv/bin/activate
uv sync
```

Run the installation sanity check:

```bash
uv run examples/example0_setup_check.py
```

## 2. Environment Variables

Copy the template:

```bash
cp .env.example .env
```

Set the keys needed for the examples you want to run:

- `OPENAI_API_KEY` — required for OpenAI-powered examples
- `TAVILY_API_KEY` — required for web search and live news retrieval examples
- `EXAMPLE9_TAVILY_MCP_URL` — required for `example9.py` remote Tavily MCP connection
- `FRED_API_KEY` — optional for future macro-oriented demos if added later

See `.env.example` for the current template and optional per-example model overrides.

## 3. Example Guide

| Example | Focus |
|---|---|
| `example0_setup_check.py` | Minimal installation sanity check |
| `example1.py` | Prompt-only baseline over static finance headlines |
| `example2.py` | Tool-enabled live finance-news retrieval and sentiment scoring |
| `example3.py` | Retrieval debug, traceability, and quality inspection |
| `example4.py` | Interactive analyst console for ad hoc market-topic queries |
| `example5.py` | Local CSV / SQL-style LatAm equities data agent |
| `example6.py` | Custom finance tool example: options maximum pain |
| `example7.py` | Clean multi-agent stock research orchestration |
| `example8.py` | Compact structured research handoff |
| `example9.py` | Agno + Tavily remote MCP issuer-intelligence demo |

## 4. Progression

- `example0_setup_check.py` → environment sanity check
- `example1.py` → prompt-only baseline
- `example2.py` → first live retrieval workflow
- `example3.py` → inspect and debug retrieval behavior
- `example4.py` → turn retrieval into an interactive analyst workflow
- `example5.py` → structured local data querying
- `example6.py` → custom domain-specific finance tool
- `example7.py` → first clean multi-agent research brief
- `example8.py` → structured research handoff with explicit open questions and gaps
- `example9.py` → issuer-intelligence workflow via Agno + Tavily remote MCP

## 5. Shared Utility Layer

`examples/finance_tools.py` provides a lightweight shared tool layer used across multiple examples for market data, analyst context, company snapshots, and web/news retrieval.

It is a compact demo utility module, not a production SDK.

## 6. Current Repository Structure

```text
next-gen-agentic-finance/
├── data/
├── examples/
│   ├── example0_setup_check.py
│   ├── example1.py
│   ├── example2.py
│   ├── example3.py
│   ├── example4.py
│   ├── example5.py
│   ├── example6.py
│   ├── example7.py
│   ├── example8.py
│   ├── example9.py
│   └── finance_tools.py
├── README.md
├── pyproject.toml
└── uv.lock
```

## 7. Where to Start

- Run `example0_setup_check.py` first.
- Then go in order from `example1.py` upward.
- Use `example3.py` when you want to inspect live retrieval behavior.
- Use `example5.py` and `example6.py` to see structured-data and custom-tool patterns.
- Use `example7.py` and `example8.py` for the more advanced multi-agent workflows.
- Use `example9.py` for a bounded issuer-intelligence demo using Agno with Tavily remote MCP.

## Disclaimer

This repository is for research and educational use only. Nothing in this codebase or its outputs constitutes investment advice.
