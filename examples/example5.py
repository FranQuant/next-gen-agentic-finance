# example5.py — LatAm Stocks Agent Demo (Corrected + Agno-Compatible)

from agno.agent import Agent
from agno.tools.csv_toolkit import CsvTools
from agno.models.lmstudio import LMStudio
from textwrap import dedent

# ---------------------------------------------------------
# 1. Load LatAm stocks CSV
# ---------------------------------------------------------
csv_tool = CsvTools(
    csvs=["../latamstocks.csv"]  # relative path from /examples/
)

# ---------------------------------------------------------
# 2. Create Agent using LMStudio local model
# ---------------------------------------------------------
agent = Agent(
    model=LMStudio(
        id="meta-llama-3.1-8b-instruct",
        cache_response=True,
        cache_ttl=3600,
    ),
    tools=[csv_tool],

    instructions=dedent("""
        You are a DuckDB SQL assistant. Your workflow ALWAYS has two steps:

        STEP 1 — Generate exactly ONE valid DuckDB SQL query following all rules below.
        STEP 2 — After the SQL query is executed, you ALWAYS:
            • Read the returned rows.
            • Display them in a markdown table.
            • Summarize the main insight.
            • Provide a final answer to the user using the actual returned data.
            • Never stop after the tool call. Always continue with analysis.

        IMPORTANT RULES:
        ----------------
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

        Example valid patterns:
            SELECT Date, MELI_Price
            FROM latamstocks
            ORDER BY Date DESC
            LIMIT 10

            SELECT Date,
                   VALE_Price,
                   VALE_Price - LAG(VALE_Price) OVER (ORDER BY Date) AS Daily_Return
            FROM latamstocks

            SELECT Date, VALE_Volume
            FROM latamstocks
            ORDER BY VALE_Volume DESC
            LIMIT 5
    """),

    markdown=True,
)

# ---------------------------------------------------------
# 3. CLI interactive app
# ---------------------------------------------------------
if __name__ == "__main__":
    agent.cli_app(stream=True)
