# Agentic Finance Examples

A curated set of agentic finance examples for tool-enabled research workflows, market and macro retrieval, multi-agent analysis, and an MCP-native finance research capstone.

Built with:

- Agno agents and teams
- OpenAI Responses models
- Tavily web search
- yfinance market data
- MCP-based web and macro integrations in `example10`

## 1. Setup

This repository expects **Python 3.13** and uses `uv` for environment and dependency management.

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

Clone the repository and create the environment:

```bash
git clone https://github.com/FranQuant/next-gen-agentic-finance.git
cd next-gen-agentic-finance

uv venv --python 3.13 .venv
source .venv/bin/activate
uv sync
```

Run the setup check:

```bash
python examples/example0_setup_check.py
```

## 2. Environment Variables

```bash
cp .env.example .env
```

Set the keys needed for the examples you want to run:

* `OPENAI_API_KEY` — OpenAI-powered examples
* `TAVILY_API_KEY` — web search and news retrieval
* `FRED_API_KEY` — optional macro data flows

See `.env.example` for the current template.

## 3. Example Guide

| Example | Focus |
|---|---|
| `example0` | Setup check |
| `example1` to `example4` | Single-agent and tool-usage foundations |
| `example5` and `example6` | Finance-shaped tool demos |
| `example7` to `example9` | Multi-agent finance workflows |
| `example10` | MCP-native finance research capstone |

Progression:

- `example0` → setup check
- `example1` to `example4` → foundations
- `example5` and `example6` → finance-oriented demos
- `example7` to `example9` → multi-agent workflows
- `example10` → flagship capstone

## 4. Shared Utility Layer

`examples/finance_tools.py` provides a lightweight shared tool layer used across multiple examples for market data, company snapshots, and web/news retrieval.

It is intended as a compact demo utility module, not a production SDK.

## 5. Current Repository Structure

```text
agentic_finance/
├── data/
├── examples/
│   ├── example0_setup_check.py
│   ├── example1.py ... example9.py
│   ├── finance_tools.py
│   ├── example10/
│   │   ├── example10.py
│   │   ├── team.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── adapters/
│   │   ├── agents/
│   │   └── services/
├── README.md
├── pyproject.toml
└── uv.lock
```

## Where to Start

- Run `example0_setup_check.py` to confirm the environment is working.
- Start with `example1.py` to `example4.py` for foundations.
- Use `example10` for the MCP-native finance research capstone.
