# Example 2 - Running Agents
from agno.agent import Agent, RunOutput
from agno.models.lmstudio import LMStudio
from agno.tools.duckduckgo import DuckDuckGoTools
from textwrap import dedent


agent = Agent(
    model=LMStudio(
        id="meta-llama-3.1-8b-instruct",
        cache_response=True,
        cache_ttl=3600,
    ),
    tools = [DuckDuckGoTools()],
    instructions=dedent("""\
        You are a News Sentiment Decoding Assistant.

    When using tools:
    1. First decide **which tool** to call and output the tool call.
    2. WAIT for the tool output.
    3. THEN generate the final answer using the tool results.

    Decode the news and provide the sentiment ranging from +10 (very positive)
    to -10 (very negative) in a table format with the following columns:
    Date, Time, News, Source, and Score.

    After the table, provide a point-by-point reasoning explaining how you
    assigned the sentiment score.
    """),
    markdown=True,
)


# use run for production
response: RunOutput = agent.run("What is the latest new on Gold?")

# print the response
print(response.content)


