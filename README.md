# NovaCell Support Assistant (RAG chatbot)

A self-serve chat assistant that answers Tier-1 telecom questions using only NovaCell's FAQ,
resolved tickets and PDF guides. Built to `PRD.md`, plus the escalation and account-data rules
from the NovaCell PRD.

## Setup (Windows)

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
copy .env.example .env        # then put your GROQ_API_KEY in .env
.venv\Scripts\python ingest_all.py
```

## Run

```bash
.venv\Scripts\streamlit run app.py      # browser UI
.venv\Scripts\python main.py            # CLI, type 'quit' to exit
```

## How a question is handled

1. **Guardrails** (`guardrails.py`)
   - Lost/stolen phone, fraud or emergency: tell the customer to call 611 straight away, with no troubleshooting.
   - "Talk to a human": give the contact route straight away.
   - Questions about the customer's own account ("my bill", "my balance"): general causes only, never amounts. The bot offers an agent.
2. **Retrieval** (`retriever.py`): the top 3 from each of `faq`, `tickets` and `guides`, searched in parallel. That gives 9 source-labelled passages.
3. **Confidence gate**: if the best match scores below `MIN_RELEVANCE` (default 0.30), the bot declines and escalates without calling the LLM.
4. **Generation** (`chain.py`): `qwen/qwen3.8-27b` on Groq at temperature 0. PRD.md names `qwen3.6-27b`, but Groq no longer serves it; set `GROQ_MODEL` in `.env` to change the model. It must answer from the passages only and never invent prices or policies. The answer streams to the UI, with an expandable Sources list.
5. **Logging**: every turn and every 👍/👎 is written to `logs/interactions.jsonl`. Each entry records the route, the sources, the top score, whether the bot escalated, and the latency.

## Updating knowledge (support ops)

| Change | Do this |
|---|---|
| FAQ entry | Edit `data/faq.csv` (`id,question,answer,category`), then run `python ingest_faq.py` |
| Resolved ticket | Insert a row in `data/tickets.db` with `status='resolved'`, then run `python ingest_tickets.py` |
| PDF guide | Drop any `.pdf` into `data/`, then run `python ingest_guides.py` |

Ingest scripts rebuild their collection from scratch, so re-running them is safe. Restart the app afterwards.
To add a new source, write an `ingest_<name>.py` and add the name to `COLLECTIONS` in `config.py`.
