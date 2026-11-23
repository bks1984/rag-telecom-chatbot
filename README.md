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

## Refund agent (`agent_PRD.md`)

Every message first goes through **intent detection** (`intent.py`), which uses the same Groq LLM with structured output:

- `requirement_inquiry` goes to the RAG pipeline above, unchanged.
- `refund_request` goes to the **Refund Agent** (`refund_agent.py`).

| Piece | File | Notes |
|---|---|---|
| Simulated accounts | `refund_data.py` | `user_123` (Rahul) has a ₹999 and a ₹499 recharge. `user_old` and `user_refunded` cover scenarios 3 and 4. The signed-in user is `DEMO_USER_ID`; the customer is never asked for it. |
| Tools | `refund_tools.py` | `get_user_account`, `validate_refund`, `process_refund`, `submit_to_support`. Only the first two are given to the LLM. |
| Validation | `refund_tools.validate_refund` | Deterministic rules: completed recharge, within 7 days, amount > 0 and ≤ the recharge, and not already refunded. An exact amount match picks that recharge. |
| Approval limit | `config.REFUND_APPROVAL_LIMIT` (₹499) | For ₹499 or less, the agent shows **Confirm Refund / Cancel**, then calls `process_refund`. Above ₹499, it calls `submit_to_support` with no confirmation step. It never reports a Support Team outcome. |
| Workflow state | `refund_agent.RefundState` | Tracks the §10 states. Only code moves between them, and `process_refund` re-validates before acting. |

The LLM reads the conversation and works out which amount the customer means. Application code makes every decision and writes every outcome message. A refund is only processed after a short, explicit "yes" ("Yes, proceed" or the Confirm button). Anything unclear is asked again. The agent never acts on an amount the customer didn't state, unless that amount is one of their recharges.

Conversation context: the last `HISTORY_MESSAGES` messages go to both routes. Retrieval still uses only the current question. If a follow-up ("should I do that before I travel?") scores below the relevance gate on its own, it is answered from the previous answer's sources. **Clear conversation** resets the history and any pending refund. The sidebar's *Demo account* panel switches the simulated user and resets the demo data.

Workflow steps (`[Intent] …`, `[Agent] validate_refund(amount=499)`, `[Refund] success …`) are logged to the console and to `logs/agent.log`.

```bash
.venv\Scripts\python -m unittest discover -s tests -v   # offline tests (scripted fake LLM)
.venv\Scripts\python scenario_check.py                   # the PRD's 7 scenarios against the live LLM
```

## Updating knowledge (support ops)

| Change | Do this |
|---|---|
| FAQ entry | Edit `data/faq.csv` (`id,question,answer,category`), then run `python ingest_faq.py` |
| Resolved ticket | Insert a row in `data/tickets.db` with `status='resolved'`, then run `python ingest_tickets.py` |
| PDF guide | Drop any `.pdf` into `data/`, then run `python ingest_guides.py` |

Ingest scripts rebuild their collection from scratch, so re-running them is safe. Restart the app afterwards.
To add a new source, write an `ingest_<name>.py` and add the name to `COLLECTIONS` in `config.py`.
