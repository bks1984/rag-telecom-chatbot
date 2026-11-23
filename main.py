"""Interactive CLI (FR-18). Type 'quit' to exit (FR-19)."""
import os
import sys

import config
from assistant import Assistant
from chain import run_turn
from refund_agent import RefundState


def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    sys.stdout.reconfigure(encoding="utf-8")       # ₹ on Windows consoles
    config.setup_logging()
    bot = Assistant()                              # intent routing: RAG or Refund Agent
    refund_state = RefundState()
    if not bot.retriever.is_ready():
        sys.exit("Knowledge base is empty. Run: python ingest_all.py")

    print("NovaCell Support Assistant - ask a question, or type 'quit' to exit.\n")
    history: list[dict] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break
        try:
            turn, answer = run_turn(bot, question, history, refund_state)
        except Exception as exc:
            print(f"\nBot: Sorry, something went wrong ({type(exc).__name__}). "
                  f"Please try again, or {config.ESCALATION_CONTACT}.\n")
            continue
        print(f"\nBot: {answer}")
        if turn.hits:
            print("Sources: " + ", ".join(dict.fromkeys(h.ref for h in turn.hits)))
        if turn.refund and turn.refund["needs_confirmation"]:
            print("(Reply 'Yes, proceed' to confirm or 'Cancel' to stop.)")
        print()
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": answer, "hits": turn.hits}]
    print("Goodbye.")


if __name__ == "__main__":
    main()
