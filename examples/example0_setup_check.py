"""Example 0: minimal Agno installation sanity check."""

try:
    from agno.agent import Agent
except ImportError:
    print("Agno is not installed or could not be imported. Install the base Agno package and rerun this example.")
    raise SystemExit(1)

agent = Agent(name="SetupCheck")

print("Base Agno installation check passed.")
print("This example verifies base Agno installation only. It does not test model providers, external APIs, or network retrieval.")
print(f"Agent name: {agent.name}")
