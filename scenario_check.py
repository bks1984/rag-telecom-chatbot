"""Run the agent_PRD §8 demonstration scenarios against the live LLM.

Run:  .venv\\Scripts\\python scenario_check.py
Each scenario starts from fresh demo data. Transient network errors are retried.
"""
import os
import sys
import time

import groq

import config
import refund_data
from assistant import Assistant
from refund_agent import RefundState

SCENARIOS = {
    "1 successful refund": ("user_123", "I want a refund of ₹499.", "Yes, proceed"),
    "2 exceeds recharge": ("user_123", "Give me a refund of ₹2,000."),
    "3 too old": ("user_old", "I want a refund of ₹299."),
    "4 already refunded": ("user_123", "I want a refund of ₹499.", "Yes, proceed",
                           "I want a refund of ₹499."),
    "5 RAG": ("user_123", "Why is my mobile internet slow?"),
    "6 above approval limit": ("user_123", "I want a refund of ₹999."),
    "7 follow-up": ("user_123", "How do I activate international roaming?",
                    "Should I do that before I travel?"),
    "missing amount + ambiguous confirmation": ("user_123", "I want a refund.", "499", "maybe",
                                                "No"),
    "how-to question stays an inquiry": ("user_123", "How do I cancel a recharge?"),
}


def respond(a: Assistant, message: str, history: list, state: RefundState, user_id: str):
    for attempt in range(4):                 # retry transient network errors only
        try:
            turn = a.respond(message, history, state, user_id)
            return turn, "".join(turn.stream)
        except groq.APIConnectionError:
            if attempt == 3:
                raise
            time.sleep(2)


def convo(a: Assistant, user_id: str, *messages: str) -> None:
    state, history = RefundState(), []
    for m in messages:
        turn, answer = respond(a, m, history, state, user_id)
        print(f"\nUSER: {m}\nBOT [{turn.intent}/{turn.route}/{state.status}]: {answer}")
        history += [{"role": "user", "content": m},
                    {"role": "assistant", "content": answer, "hits": turn.hits}]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    config.setup_logging()
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set.")
    a = Assistant()
    for name, (uid, *msgs) in SCENARIOS.items():
        refund_data.reset_demo_data()
        print(f"\n===== Scenario {name} ({uid})", flush=True)
        convo(a, uid, *msgs)
        print("  recharges:", [(r["recharge_id"], r["refund_status"])
                               for r in refund_data.USERS[uid]["recharges"]],
              "| support queue:", [q["ticket_id"] for q in refund_data.SUPPORT_QUEUE], flush=True)


if __name__ == "__main__":
    main()
