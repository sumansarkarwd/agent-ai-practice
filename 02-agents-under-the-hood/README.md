# 02 — Agents Under the Hood

A shopping-cart assistant implemented two ways to show what an "agent" actually is: a model + a tool-calling loop. Both variants use the same two tools — `get_product_price` and `get_discounted_price` — and the same system rules.

## What it demonstrates

- The agent loop: model decides -> tool call -> observation -> repeat -> final answer
- Two implementations of the same loop:
  - `main.py` — uses LangChain's native tool-calling (`bind_tools` + `tool_calls`)
  - `using_regex_main.py` — a raw ReAct-style prompt with `Action:` / `Action Input:` / `Observation:` parsed via regex
- LangSmith tracing via `@traceable` decorators

## Files

- `main.py` — LangChain tool-calling agent loop (`init_chat_model`, `bind_tools`)
- `using_regex_main.py` — Raw OpenAI client + regex-parsed ReAct prompt

## Requirements

- `OPENAI_API_KEY` in the repo-root `.env` (required)
- `LANGSMITH_*` keys (optional, enables traces in LangSmith)

## Run

From the repo root:

```bash
uv run python 02-agents-under-the-hood/main.py
uv run python 02-agents-under-the-hood/using_regex_main.py
```

Both scripts default to the query: _"What is the price of the laptop after applying gold discount?"_

## Available test data

The tools have a tiny built-in catalog and discount table:

| Product | Price |
|---------|-------|
| laptop  | 1000  |
| bag     | 5     |
| charger | 50    |

| Tier   | Discount |
|--------|----------|
| gold   | 70%      |
| silver | 30%      |
| bronze | 10%      |
