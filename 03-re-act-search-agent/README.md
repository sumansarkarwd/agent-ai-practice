# 03 — ReAct Search Agent

A ReAct-style agent that answers questions by searching the web with Tavily and returns a structured response (final answer plus the list of source URLs it used).

## What it demonstrates

- Building an agent with `langchain.agents.create_agent`
- Using `TavilySearch` as a tool the agent can call
- Enforcing a typed output schema with Pydantic (`AgentReponse` with `answer` and `sources`)

## Files

- `main.py` — defines the response schema, builds the agent, and asks it about the weather in Kolkata and Tokyo

## Requirements

- `OPENAI_API_KEY` in the repo-root `.env`
- `TAVILY_API_KEY` in the repo-root `.env` — get one at [tavily.com](https://tavily.com)

The script validates both keys up front and raises a clear error if either is missing.

## Run

From the repo root:

```bash
uv run python 03-re-act-search-agent/main.py
```

The default question is _"Tell me the weather in Kolkata and Tokyo"_ — edit the `HumanMessage` in `main.py` to ask something else.
