# Example 3 - Debugging Agents
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
        Decode the news and provide the sentiment ranging from +10 (very positive)
        to -10 (very negative) in a table format with the following columns:
        Date, Time, News, Source, and Score.
        
        After the table, provide a point-by-point reasoning explaining how you
        assigned the sentiment score.
    """),
    markdown=True,
    debug_mode=True,
)


agent.print_response("what is the latest news on Gold?", stream=True)
