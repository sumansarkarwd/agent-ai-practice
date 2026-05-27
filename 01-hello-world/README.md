# 01 — Hello World

A first-look LangChain example: feed a biography of Stephen Hawking to an LLM and ask for a short summary plus two interesting facts.

## What it demonstrates

- Loading a `PromptTemplate` with an input variable
- Wiring a prompt into a `ChatOpenAI` model via the `|` (LCEL) operator
- Reading a local text file (`data/stephen_hawking.txt`) as model input

## Files

- `main.py` — entry point that builds the prompt, invokes the chain, and prints the response
- `data/stephen_hawking.txt` — biography passed into the prompt

## Requirements

- `OPENAI_API_KEY` set in the repo-root `.env`

## Run

From the repo root:

```bash
uv run python 01-hello-world/main.py
```

The data file is resolved relative to `main.py`, so it works from any working directory.
