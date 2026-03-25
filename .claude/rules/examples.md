---
paths:
  - "examples/*.py"
  - "examples/**/*.py"
---

# Rules for Working on Examples

## The One-Pattern Rule
Each example introduces exactly ONE new concept. Never add multiple new
patterns to a single example. If a new example needs both a new tool AND
a new agent pattern, split it into two examples.

## Agno Agent Structure
Always follow this pattern when creating or editing agents:
```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    tools=[...],
    instructions=["instruction 1", "instruction 2"],
    show_tool_calls=True,
    markdown=True,
)
```

Never use LangChain, LlamaIndex, or any other agent framework.

## Adding a New Tool to finance_tools.py
Always use the @tool decorator pattern:
```python
from agno.tools import tool

@tool
def get_something(ticker: str) -> str:
    """
    One-line description of what this tool does.
    Args:
        ticker: stock ticker symbol (e.g. AAPL)
    Returns:
        description of what gets returned
    """
    # implementation
    return result
```

Never define tools as plain functions without @tool.
Never define tool logic inline inside an Agent — it goes in finance_tools.py.

## Multi-Agent Pattern (example7+ style)
When creating agent teams:
```python
from agno.team import Team

team = Team(
    members=[agent1, agent2, orchestrator],
    mode="coordinate",
)
```

The orchestrator agent always goes last in the members list.

## Running and Testing
Always run with: uv run examples/exampleN.py
Never use: python examples/exampleN.py

## Environment Variables
Always load at the top of every example file:
from dotenv import load_dotenv
load_dotenv()

Never hardcode API keys. Never print or log API key values.

## Error Handling for External Tools
Always handle empty/failed responses from yfinance and Tavily:
- yfinance: check if DataFrame is empty before processing
- Tavily: check if results list is empty before iterating
