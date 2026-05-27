# Agent AI Practice

Udacity course exercises consolidated from multiple branches into numbered project folders. Shared project configuration lives at the repo root.

## Projects

| Folder | Description |
|--------|-------------|
| `01-hello-world/` | LLM biographical summarizer using LangChain and a Stephen Hawking dataset |
| `02-agents-under-the-hood/` | Shopping cart assistant with tool-calling (LangChain agent loop and regex-based ReAct variant) |
| `03-re-act-search-agent/` | ReAct agent with Tavily web search and structured output |

## Setup

1. Copy environment template and fill in your keys:

   ```bash
   cp .env.example .env
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

## Run

From the repo root:

```bash
uv run python 01-hello-world/main.py
uv run python 02-agents-under-the-hood/main.py
uv run python 02-agents-under-the-hood/using_regex_main.py
uv run python 03-re-act-search-agent/main.py
```

## Environment variables

- `OPENAI_API_KEY` — required for all projects
- `LANGSMITH_*` — optional tracing for projects in `02-agents-under-the-hood/`
- `TAVILY_API_KEY` — required for `03-re-act-search-agent/`
