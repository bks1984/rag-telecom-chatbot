"""Streamlit chat UI.  Run:  .venv\\Scripts\\streamlit run app.py"""
import logging
import os
import time

import streamlit as st

import config
from chain import SupportBot, log_feedback, log_interaction

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "Why is my bill higher than usual?",
    "My eSIM won't activate - what should I do?",
    "How do I set up roaming before I travel?",
    "My calls keep dropping. How do I fix it?",
    "I can't log in to the MyTelecom app",
    "What's my current balance?",
]

SOURCE_BADGE = {"FAQ": "📘 FAQ", "TICKET": "🎫 Ticket", "GUIDE": "📄 Guide"}

st.set_page_config(page_title="NovaCell Support Assistant", page_icon="📶", layout="centered")


@st.cache_resource(show_spinner="Loading knowledge base…")
def load_bot() -> SupportBot:
    return SupportBot()


def render_sources(hits) -> None:
    if not hits:
        return
    with st.expander(f"Sources ({len(hits)})"):
        for h in hits:
            badge = SOURCE_BADGE.get(h.source, h.source)
            st.markdown(f"**{badge} · {h.ref}** — relevance {h.score:.2f}")
            st.caption(h.doc.page_content[:400] + ("…" if len(h.doc.page_content) > 400 else ""))


def render_feedback(msg: dict) -> None:
    turn_id = msg.get("turn_id")
    if not turn_id:
        return
    choice = st.feedback("thumbs", key=f"fb-{turn_id}")
    if choice is not None and msg.get("feedback") is None:
        msg["feedback"] = "up" if choice == 1 else "down"
        log_feedback(turn_id, msg["feedback"])
        st.toast("Thanks for the feedback")


# ---------- state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

# ---------- sidebar ----------
with st.sidebar:
    st.header("📶 NovaCell Support")
    st.caption("Answers come only from NovaCell's FAQ, resolved support cases and user guides. "
               "I can't see your account.")
    st.subheader("Try a question")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown(f"**Need a person?** {config.ESCALATION_CONTACT.capitalize()}.")

# ---------- startup checks ----------
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Copy `.env.example` to `.env` and add your key.")
    st.stop()

bot = load_bot()
if not bot.retriever.is_ready():
    st.error("The knowledge base is empty. Run `python ingest_all.py` first.")
    st.stop()

st.title("How can we help?")
if not st.session_state.messages:
    st.info("Ask about mobile data, bills, SIM/eSIM, roaming, calls or the MyTelecom app.")

# ---------- history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("hits"))
            render_feedback(msg)

# ---------- new turn ----------
question = st.chat_input("Type your question…") or st.session_state.pending
st.session_state.pending = None

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        try:
            turn = bot.ask(question, history)
            answer = st.write_stream(turn.stream)          # FR-05 token streaming
        except Exception as exc:                            # network / API errors
            logging.exception("answer generation failed")
            turn, answer = None, (
                "Sorry, I couldn't reach the answer service just now. "
                f"Please try again, or {config.ESCALATION_CONTACT}."
            )
            st.markdown(answer)
            st.caption(f"Error: {type(exc).__name__}")
        if turn is not None:
            log_interaction(turn, answer, time.perf_counter() - start)
            render_sources(turn.hits)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "hits": turn.hits if turn else [],
        "turn_id": turn.turn_id if turn else None,
        "feedback": None,
    })
    st.rerun()
