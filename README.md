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

Copy the example file and fill in the keys you want to use:

```bash
cp .env.example .env
```

Common variables:

OPENAI_API_KEY — required for examples that use OpenAI models

TAVILY_API_KEY — required for web search and news retrieval

FRED_API_KEY — optional, used for macro data flows

Some examples may run with only a subset of these variables, depending on the tools they use. See .env.example for the current template.

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

### `examples/finance_tools.py`

Shared finance tool layer used by the mid-repo AGNO examples. It currently provides:

- latest price snapshots
- analyst recommendations
- curated company info
- normalized company news from yfinance
- Tavily-backed company news retrieval

This file is best understood as a reusable demo tool boundary for examples such as `example7` and `example9`, rather than as a general-purpose production data library.

## 5. Current Repository Structure

```text
agentic_finance/
├── data/
│   └── latamstocks.csv
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
├── tmp/
└── uv.lock
```

Runtime artifacts are written under `tmp/` and are not part of the core source tree.

## 6. Notes on the Current Architecture

The repository changes shape as it progresses:

- **Examples 1-6** are mostly single-file AGNO and tool-calling demos.
- **Examples 7-9** introduce shared finance tools and multi-agent workflows.
- **Example 10** packages the workflow into adapters, schemas, storage, and reporting.

The later examples are best read as compact architecture demos rather than as production systems. In `example10`, the packaged "agents" are deterministic analyzers inside an orchestrator stack, and the example is designed to degrade explicitly when live external data is unavailable.

## 7. Validation

This repository does not currently ship a dedicated automated test suite. Useful local checks:

```bash
python -m compileall examples
python examples/example0_setup_check.py
uv run examples/example7.py
EXAMPLE10_PLAIN_REPORT=1 uv run examples/example10/example10.py "Analyze MSFT for a 3-12 month portfolio view."
```

## 8. Where to Start

If you are new to the repository:

- start with `example0_setup_check.py`
- read `example1.py` through `example4.py` for the smallest AGNO demos
- jump to `example7.py` through `example9.py` for multi-agent finance workflows
- read `example10/` for the packaged orchestrator path

## 9. Disclaimer

This repository is for research and educational use only. Nothing in this codebase or its outputs constitutes investment advice.
