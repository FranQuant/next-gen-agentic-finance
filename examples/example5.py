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
        model=OpenAIResponses(id="gpt-5.2"),
        tools=[csv_tool],
        instructions=dedent("""\
            You are a LatAm equities data assistant that uses DuckDB SQL over a local CSV dataset.

            Your workflow always has two steps:

            Step 1 — Write exactly one valid DuckDB SQL query.
            Step 2 — After the query executes:
                • show the SQL
                • display the returned rows in a markdown table
                • summarize the main insight
                • answer the user using only the returned data
                • Never stop after the tool call.

            IMPORTANT RULES:
            • Table name is: latamstocks
            • Valid column patterns:
                  {TICKER}_Price
                  {TICKER}_Volume
            • Invalid columns you must NEVER use:
                  Stock, Price, Volume
            • Date column is named Date and is a DATE type.

            ALLOWED DATE FILTERS:
                YEAR(Date) = 2020
                STRFTIME(Date, '%Y') = '2020'
                Date = DATE '2020-01-15'

            DISALLOWED:
                LIKE, DATE('now'), NOW(), CURRENT_DATE

            ONE SQL STATEMENT ONLY.
            NO semicolons.
            NO multi-query logic.

            If a requested ticker or column is not available in the dataset,
            say so clearly and do not guess.

            Example valid patterns:
                SELECT Date, MELI_Price
                FROM latamstocks
                ORDER BY Date DESC
                LIMIT 10

                SELECT Date,
                       VALE_Price,
                       (VALE_Price / LAG(VALE_Price) OVER (ORDER BY Date)) - 1 AS Daily_Return
                FROM latamstocks

                SELECT Date, VALE_Volume
                FROM latamstocks
                ORDER BY VALE_Volume DESC
                LIMIT 5
        """),
        markdown=True,
    )


def main() -> None:
    agent = build_agent()
    agent.cli_app(stream=True)


if __name__ == "__main__":
    main()