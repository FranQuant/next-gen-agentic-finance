# Agentic Finance Examples

A curated set of agentic finance examples covering tool-enabled research workflows, market and macro retrieval, multi-agent analysis, and an MCP-native finance research capstone.

The repository brings together:

- AGNO agents and teams
- OpenAI Responses models
- Tavily search and web intelligence
- yfinance market and company data
- MCP-based web and macro integrations in `example10`

## Repository Purpose

This repository is a reference library of agentic finance examples focused on:

- agent orchestration patterns
- tool-calling in finance-oriented workflows
- market, macro, and web-intelligence retrieval
- compact research pipelines
- adapter and orchestrator design
- structured reporting and run persistence

It is intended for experimentation, learning, and architectural reference rather than for live trading or production deployment.

## 1. Setup

### Prerequisites

Install `uv` first if it is not already available on your system:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

### Clone and install

```bash
git clone https://github.com/FranQuant/next-gen-agentic-finance.git
cd next-gen-agentic-finance
uv venv .venv
source .venv/bin/activate
uv sync
```

### Minimal setup check

```bash
python examples/example0_setup_check.py
```

Expected output:

```text
minimal Agno setup check passed
```

## 2. Environment Variables

Create a `.env` file in the project root for the networked examples:

```bash
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
FRED_API_KEY=your_key_here
```

Notes:

OPENAI_API_KEY is used by the AGNO/OpenAI examples in examples 1–9.

TAVILY_API_KEY is used by Tavily-backed search and web-intelligence flows.

FRED_API_KEY is optional for live macro data in example10.

example10 is designed to degrade safely when live external data is unavailable.

## 3. Example Guide

| Example | Focus | Entry Point |
| --- | --- | --- |
| `0` | Environment and AGNO setup check | `python examples/example0_setup_check.py` |
| `1` | Fixed-headline sentiment demo | `uv run examples/example1.py` |
| `2` | Tavily-backed sentiment demo | `uv run examples/example2.py` |
| `3` | Tool-visible sentiment demo | `uv run examples/example3.py` |
| `4` | Interactive CLI sentiment agent | `uv run examples/example4.py` |
| `5` | LatAm CSV assistant | `uv run examples/example5.py` |
| `6` | Options maximum-pain tool demo | `uv run examples/example6.py` |
| `7` | Two-member research team using shared finance tools | `uv run examples/example7.py` |
| `8` | Persistent multi-agent investment memo workflow | `uv run examples/example8.py` |
| `9` | Compact research-to-portfolio workflow | `uv run examples/example9.py` |
| `10` | MCP-native finance research capstone | `uv run examples/example10/example10.py "Assess whether current macro and web evidence support a tactical overweight to equities"` |

### Common entry points

```bash
python examples/example0_setup_check.py
uv run examples/example7.py
uv run examples/example10/example10.py "Assess whether current macro and web evidence support a tactical overweight to equities"
EXAMPLE10_PLAIN_REPORT=1 uv run examples/example10/example10.py "Analyze MSFT for a 3-12 month portfolio view."
```

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
