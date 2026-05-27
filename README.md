# Agent AI Practice

Udacity course exercises consolidated from multiple branches into numbered project folders. Shared project configuration (dependencies, env, lockfile, VS Code settings) lives at the repo root.

## Projects

| Folder | Description |
|--------|-------------|
| [`01-hello-world/`](01-hello-world/README.md) | LLM biographical summarizer using LangChain (`PromptTemplate` + `ChatOpenAI`) |
| [`02-agents-under-the-hood/`](02-agents-under-the-hood/README.md) | Shopping-cart agent loop — once with LangChain tool-calling, once with a raw regex-parsed ReAct prompt |
| [`03-re-act-search-agent/`](03-re-act-search-agent/README.md) | ReAct agent with Tavily web search and a typed Pydantic response schema |

See each project's `README.md` for what it demonstrates and how to run it.

## Setup

1. Copy the environment template and fill in your keys:

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
- `LANGSMITH_TRACING`, `LANGSMITH_ENDPOINT`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` — optional, enables LangSmith tracing (used by project 02)
- `TAVILY_API_KEY` — required for project 03

## Layout

```
.
├── 01-hello-world/
│   ├── README.md
│   ├── main.py
│   └── data/stephen_hawking.txt
├── 02-agents-under-the-hood/
│   ├── README.md
│   ├── main.py
│   └── using_regex_main.py
├── 03-re-act-search-agent/
│   ├── README.md
│   └── main.py
├── .vscode/
├── .env.example
├── pyproject.toml
├── uv.lock
└── README.md
```
