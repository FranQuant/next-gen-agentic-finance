"""Example 5: interactive LatAm stocks SQL agent over a local CSV dataset."""

from textwrap import dedent

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.csv_toolkit import CsvTools

load_dotenv()


def build_agent() -> Agent:
    csv_tool = CsvTools(csvs=["data/latamstocks.csv"])

    return Agent(
        model=OpenAIResponses(id="gpt-5.4"),
        tools=[csv_tool],
        instructions=dedent("""\
            You are a LatAm equities data assistant that queries a local CSV dataset with SQL.

            Workflow:
            1. Write exactly one valid SQL query.
            2. After the query executes:
               - show the SQL in a fenced sql block
               - display the returned rows in one markdown table
               - provide exactly one concise conclusion based only on the returned data

            Rules:
            - Table name is latamstocks
            - Available columns are: Date, Ticker, Close, Volume
            - Ticker values are uppercase symbols such as MELI, VALE, EC, and PBR
            - Date is a DATE column
            - If the requested ticker is not present in the dataset, say so clearly
            - Use only the returned data
            - Use exactly one SQL statement
            - Do not use semicolons
            - Do not invent columns
            - Do not add extra headings, sections, or multiple summary layers
        """),
        markdown=True,
    )


def main() -> None:
    agent = build_agent()
    agent.cli_app(stream=True)


if __name__ == "__main__":
    main()
