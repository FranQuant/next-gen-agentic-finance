"""Example 5: interactive LatAm stocks SQL agent over a local CSV dataset."""

import os
from pathlib import Path
from textwrap import dedent

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from agno.tools.csv_toolkit import CsvTools

load_dotenv()

MODEL_ID = os.getenv("EXAMPLE5_MODEL_ID", os.getenv("OPENAI_MODEL_ID", "gpt-5.4"))
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "latamstocks.csv"


def build_agent() -> Agent:
    csv_tool = CsvTools(csvs=[str(DATA_PATH)])

    return Agent(
        model=OpenAIResponses(id=MODEL_ID),
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
            - If the query returns no rows, say so clearly in the conclusion
            - Use only the returned data
            - Keep the conclusion strictly grounded in the returned rows
            - Use exactly one SQL statement
            - Do not use semicolons
            - Do not invent columns
            - Do not add unsupported analysis, calculations, or assumptions
            - Do not ask follow-up questions
            - Do not add "what would you like to know next" or similar follow-up messaging
            - End the response immediately after the single concise conclusion
            - Do not add extra headings, sections, or multiple summary layers
        """),
        markdown=True,
    )


def main() -> None:
    agent = build_agent()
    agent.cli_app(stream=True)


if __name__ == "__main__":
    main()
