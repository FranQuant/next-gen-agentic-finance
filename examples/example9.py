# examples/example9.py
# Hedge Fund OS – AgentOS Server using GPT-5.1 (JSON-only API)

from textwrap import dedent

from fastapi import Body
from agno.agent import Agent
from agno.team import Team
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIResponses
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools
from agno.os import AgentOS


# ---------------------------------------------------------
# 1. Persistent storage
# ---------------------------------------------------------
db = SqliteDb(db_file="tmp/agentos_hedge_fund_team.db")


# ---------------------------------------------------------
# 2. Market Data Agent
# ---------------------------------------------------------
market_data_agent = Agent(
    id="market-data-agent",
    name="Market Data Agent",
    role="Financial Data Retrieval Specialist",
    model=OpenAIResponses(id="gpt-5.1"),
    tools=[YFinanceTools()],
    instructions=dedent("""
        Fetch and summarize all market + fundamental data for a ticker.
        ALWAYS call tools for data. Never fabricate numbers.
        Explicitly state unavailable fields.
    """),
    markdown=True,
    db=db,
)


# ---------------------------------------------------------
# 3. Strategy Agent
# ---------------------------------------------------------
strategy_agent = Agent(
    id="quant-strategy-agent",
    name="Quantitative Strategist",
    role="Trading Strategy Developer",
    model=OpenAIResponses(id="gpt-5.1"),
    instructions=dedent("""
        Using ONLY the Market Data Agent output, produce:
        - stance (buy/sell/hold)
        - entry/exit zones (ranges)
        - position sizing logic
        - 3-scenario qualitative framework
        - risk management plan
        Never invent missing quantitative data.
    """),
    markdown=True,
    db=db,
)


# ---------------------------------------------------------
# 4. Team Pipeline (internal, not exposed)
# ---------------------------------------------------------
hedge_fund_team = Team(
    name="AI Hedge Fund Analysis Team",
    model=OpenAIResponses(id="gpt-5.1"),
    members=[market_data_agent, strategy_agent],
    tools=[ReasoningTools(add_instructions=True)],
    instructions=dedent("""
        Workflow:
        1. Delegate to Market Data Agent.
        2. Delegate to Strategy Agent.
        3. Produce final investment memo.
    """),
    db=db,
    markdown=True,
)


# ---------------------------------------------------------
# 5. Exposed Orchestrator Agent (JSON input only)
# ---------------------------------------------------------
class HedgeFundOSAgent(Agent):

    async def run(self, input_data: str):
        """Accepts a JSON string and passes it to the team pipeline."""
        result = hedge_fund_team.run(input_data)
        return result


hedge_fund_os = HedgeFundOSAgent(
    id="hedge-fund-os",
    name="Hedge Fund OS",
    role="Team Coordinator",
    model=OpenAIResponses(id="gpt-5.1"),
    instructions="You orchestrate the internal hedge_fund_team pipeline.",
    markdown=True,
    db=db,
)


# ---------------------------------------------------------
# 6. AgentOS registration
# ---------------------------------------------------------
agent_os = AgentOS(agents=[hedge_fund_os])
app = agent_os.get_app()


# ---------------------------------------------------------
# 7. Override POST /agents/{id}/runs to use JSON instead of multipart
# ---------------------------------------------------------
@app.post("/agents/{agent_id}/runs/json")
async def run_json(agent_id: str, payload: dict = Body(...)):
    """
    Clean JSON endpoint:
    {
        "message": "Analyze AAPL"
    }
    """
    msg = payload.get("message", "")
    agent = agent_os.get_agent(agent_id)
    return await agent.run(msg)


# ---------------------------------------------------------
# 8. Health check endpoint (optional)
# ---------------------------------------------------------
@app.get("/hedge_fund_os_alive")
async def alive():
    return {
        "status": "Hedge Fund OS is running (JSON mode)",
        "agents": ["hedge-fund-os"],
    }


# ---------------------------------------------------------
# 9. Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
