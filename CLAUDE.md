# next-gen-agentic-finance

## What This Project Is
A progressive demo series for agentic finance research — Layer 1 of a quant research stack. Agents retrieve live market intelligence, reason over it, and surface structured findings for human review.

This is NOT a trading system. It informs human decisions, it does not make them. Never suggest code that executes trades or manages capital.

## Stack — Use These, Not Alternatives
- **Language:** Python 3.13 (strict — do not suggest 3.11 patterns)
- **Agent framework:** Agno — NOT LangChain, NOT LlamaIndex
- **LLM backend:** OpenAI gpt-4o / o-series via OpenAI Responses API
- **Live retrieval:** Tavily (web + news search)
- **Market data:** yfinance
- **Structured data:** DuckDB + CsvTools
- **MCP transport:** async remote MCP (example9 pattern)
- **Package manager:** uv — never suggest pip install directly
- **Virtual env:** .venv at project root

## How to Run Things
Always use uv to run examples:
  uv run examples/example7.py
Never use python directly — always uv run.

## Project Structure
examples/
  finance_tools.py        # Shared tool layer — edit this for new tools
  example0–example9.py    # Progressive examples, each adds ONE pattern
data/
  latamstocks.csv         # LatAm equities dataset (example5 only)
pyproject.toml            # Dependencies — use uv add, not pip
.env                      # API keys — never commit, never print

## Architecture — The Example Progression
Each example adds exactly ONE new pattern. This is intentional.
When adding new examples, follow this constraint strictly.

Shared tool layer: finance_tools.py — all reusable finance functions go here.
Never duplicate tool logic inside individual examples.

Agent pattern: Always use Agno's Agent/Team abstractions.
Pattern: Agent(model=..., tools=[...], instructions=...)

## Environment Variables
Stored in .env — never hardcode, never print, never log.
- OPENAI_API_KEY — required for all examples
- TAVILY_API_KEY — required for example2–4, example9
- EXAMPLE9_TAVILY_MCP_URL — required for example9 remote MCP
- FRED_API_KEY — reserved, not yet used

Load with: from dotenv import load_dotenv at top of each file.

## Key Gotchas & Constraints
- Agno is NOT LangChain — do not use LangChain patterns or imports
- yfinance can return empty DataFrames — always check before processing
- Tavily returns a list of results — always handle empty list case
- DuckDB queries in example5 run in-memory — no persistent DB
- MCP transport in example9 is async — use async/await correctly
- When adding a new tool to finance_tools.py, follow the existing @tool decorator pattern exactly
