"""Example 0: minimal Agno installation and environment sanity check."""


from agno.agent import Agent

# Simple test agent for your setup
agent = Agent(
    name="SetupCheck",
    description="A simple test agent running in my custom quant environment."
)

print("minimal Agno setup check passed")
print(f"Agent name: {agent.name}")
