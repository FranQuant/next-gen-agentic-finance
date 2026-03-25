# Review Example File

Review the example file specified: $ARGUMENTS

If no file is specified, find the most recently modified file in examples/ and review that.

## What to Check

### The One-Pattern Rule
- Does this example introduce exactly ONE new concept compared to the previous example?
- If multiple new patterns exist, name each one explicitly

### Agno Structure
- Is Agent() using: model, tools, instructions, show_tool_calls, markdown?
- Is the model OpenAIResponses (not OpenAIChat)?
- Does the agent have name and role fields?
- Are multi-line instructions using dedent("""...""")?
- Is markdown=False on member agents, markdown=True only on the orchestrator?
- Are section headers using # ==== style comments?

### Tool Conventions
- Are all tools imported from finance_tools.py — nothing defined inline?
- Do new tools in finance_tools.py use the @tool decorator?
- Does each @tool function have a docstring with Args and Returns?

### Environment Variables
- Is load_dotenv() at the top of the file?
- Does each model use its own env var with a DEFAULT_MODEL_ID fallback?
- Are any API keys hardcoded? Flag immediately if yes.

### Error Handling
- Are yfinance calls checking for empty DataFrames?
- Are Tavily calls checking for empty results list?

### Runnability
- Is there a if __name__ == "__main__": block?
- Would uv run examples/exampleN.py work without errors?

## Output Format

For each issue:
FILE:LINE | SEVERITY | description
→ exact fix

Severities: bug | convention | missing | suggestion

Finish with either:
✅ READY TO COMMIT — no issues found
❌ X issues found — fix before committing
