# Agentic Finance Examples

Multi-Agent Workflows for Market Data, Research, and Strategy Prototyping

This repository provides a set of agentic finance examples built with:

* AGNO (multi-agent and AgentOS framework) https://www.agno.com 
* LM Studio https://lmstudio.ai
* OpenAI GPT-5.1 models
* Local LLMs via LM Studio
* Oanda API (market data)
* Tavily API (web research) https://www.tavily.com
* Pandas / NumPy (data processing)
* Matplotlib (plots)
* FastAPI / Uvicorn (service layer)

Each example is independent, minimal, and fully runnable.
The goal is not to ship a trading system, but to give a clear reference library for:

* Agent orchestration
* Tool-calling
* Market data retrieval
* Research pipelines
* Simple backtesting and diagnostics

---

## 1. Environment Setup

This project uses **UV** for reproducible Python environments.

### 1.1 Clone the repository

```bash
git clone https://github.com/FranQuant/next-gen-agentic-finance.git
cd 10_agentic_finance
```

### 1.2 Create and sync the environment

```bash
uv sync
```

### 1.3 Activate the environment

```bash
source .venv/bin/activate
```

Quick check:

```bash
python -c "import agno, openai, pandas; print('Environment OK')"
```

If you see `Environment OK`, you are ready to run the examples.

---

## 2. API Keys and Configuration

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
OANDA_API_KEY=your_key_here
OANDA_ACCOUNT_ID=your_account
OANDA_ENV=practice
```

Notes:

* `.env` is ignored by git (see `.gitignore`).
* Do **not** commit any keys or credentials.

---

## 3. Local LLM Setup (Examples 1–6)

Examples 1–6 use **LM Studio** as a local inference backend.

Requirements:

* LM Studio installed and running
* Any Llama-compatible model (for example: Llama-3, Mistral)
* In LM Studio, enable:

  * “Local server”
  * Endpoint: `http://localhost:1234/v1`

Once LM Studio is running, you can execute:

```bash
uv run examples/example1.py
```

---

## 4. Running the Examples

The general pattern is:

```bash
uv run examples/exampleX.py
```

### 4.1 Examples 1–6 — Local LLM Agent Patterns

Showcases:

* Basic agent definitions
* Simple tools and reasoning
* Short, local inference loops via LM Studio

Example:

```bash
uv run examples/example3.py
```

---

### 4.2 Example 7 — GPT-5.1 Research Agent with Tools

A research agent that calls structured tools (market data, fundamentals, etc.) and produces a compact memo.

```bash
uv run examples/example7.py
```

---

### 4.3 Example 8 — Extended Research Pipeline

Builds on Example 7 with richer prompts and metadata, still using GPT-5.1.

```bash
uv run examples/example8.py
```

---

### 4.4 Example 9 — AgentOS FastAPI Server

Runs an AgentOS-based microservice that exposes a hedge-fund style research pipeline over HTTP.

Start the server:

```bash
uv run examples/example9.py
```

In another terminal:

```bash
curl -X POST "http://127.0.0.1:8000/agents/hedge-fund-os/runs/json" \
     -H "Content-Type: application/json" \
     -d '{"message": "Analyze NVDA and produce a hedge-fund-grade summary."}'
```

---

### 4.5 Example 10 — Oanda + Tavily Research Engine

Single-asset FX research pipeline:

* Fetches historical data from Oanda
* Builds technical features and quick diagnostics
* Pulls macro / FX headlines from Tavily
* Produces a structured research memo with GPT-5.1

```bash
uv run examples/example10_oanda_tavily.py --symbol EUR_USD
```

Charts are written to `plots/`.

---

### 4.6 Example 11 — Multi-Asset Research Engine

Compact multi-asset engine for FX, equity index, and gold:

* Oanda price history
* Feature engineering and mini-backtest
* Simple regime readout
* GPT-5.1 macro / FX memo per asset

Supported assets:

* `EURUSD`
* `USDMXN`
* `SPX500`
* `XAUUSD`

Run a single asset:

```bash
uv run examples/example11_multi_asset_engine.py --asset EURUSD
```

Process all assets in sequence:

```bash
uv run examples/example11_multi_asset_engine.py --all
```

Custom lookback (default 120 days):

```bash
uv run examples/example11_multi_asset_engine.py --asset EURUSD --lookback 180
```

---

## 5. Repository Structure

```text
10_agentic_finance/
│
├── examples/
│   ├── main.py                          # Local LLM starter
│   ├── example1.py                      # Basic agent
│   ├── example2.py                      # Multi-step reasoning
│   ├── example3.py                      # Tool-using agent
│   ├── example4.py                      # CLI agent interface
│   ├── example5.py                      # Pain-level / stateful agent example
│   ├── example6.py                      # Local LLM agent patterns
│   ├── example6a.py                     # Improved stateful agent
│   ├── example7.py                      # GPT-5.1 research agent
│   ├── example8.py                      # Long-form analyst agent
│   ├── example9.py                      # AgentOS FastAPI server
│   ├── example10_oanda_tavily.py        # Single-asset FX engine
│   ├── example11_multi_asset_engine.py  # Multi-asset agentic quant engine
│   └── finance_tools.py                 # Indicators, backtests, chart utilities
│
├── plots/                               # Auto-generated charts
├── tmp/                                 # AgentOS DBs, caches
├── pyproject.toml
├── uv.lock
└── README.md
```

The `plots/` and `tmp/` directories are generated at runtime and can be deleted at any time.

---

## 6. Extending the Project

Possible extensions:

* Add portfolio-construction agents (risk-based or forecast-based)
* Extend the multi-asset engine with more instruments and richer diagnostics
* Plug in CSV or database sources (for example, LatAm equities, fundamentals)
* Swap GPT-5.1 for fully local models in production setups
* Add RAG layers for filings, earnings transcripts, or broker research

Every example is intentionally small so you can treat this repo as a starting point for your own agentic finance stack.

---

## 7. Disclaimer

This repository is for research and educational use only.
Nothing in this codebase or its outputs constitutes investment advice.
