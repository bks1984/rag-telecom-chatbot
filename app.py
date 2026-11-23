"""Streamlit chat UI.  Run:  .venv\\Scripts\\streamlit run app.py"""
import logging
import os
import time

import streamlit as st

import config
import refund_data
from assistant import Assistant
from chain import log_feedback, log_interaction
from refund_agent import AWAITING_CONFIRMATION, RefundState, rupees

config.setup_logging()

SAMPLE_QUESTIONS = [
    "Why is my mobile internet so slow?",
    "Why is my bill higher than usual?",
    "My eSIM won't activate - what should I do?",
    "How do I set up roaming before I travel?",
    "My calls keep dropping. How do I fix it?",
    "I can't log in to the MyTelecom app",
    "What's my current balance?",
    "I want a refund of ₹499.",
    "I want a refund of ₹999.",
    "I want a refund.",
]

CONFIRM_TEXT, CANCEL_TEXT = "Yes, proceed", "Cancel"

SOURCE_BADGE = {"FAQ": "📘 FAQ", "TICKET": "🎫 Ticket", "GUIDE": "📄 Guide"}

st.set_page_config(page_title="NovaCell Support Assistant", page_icon="📶", layout="centered")


@st.cache_resource(show_spinner="Loading knowledge base…")
def load_bot() -> Assistant:
    return Assistant()


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


def render_refund_confirmation(msg: dict) -> None:
    """agent_PRD §11: Confirm/Cancel buttons for an eligible refund at or below ₹499."""
    state: RefundState = st.session_state.refund_state
    if not msg.get("needs_confirmation") or state.status != AWAITING_CONFIRMATION:
        return
    left, right = st.columns(2)
    if left.button(f"✅ Confirm Refund ({rupees(state.amount)})", key=f"ok-{msg['turn_id']}",
                   type="primary", use_container_width=True):
        st.session_state.pending = CONFIRM_TEXT
    if right.button("Cancel", key=f"no-{msg['turn_id']}", use_container_width=True):
        st.session_state.pending = CANCEL_TEXT


def reset_conversation() -> None:
    """FR-48 / §11: clears the history and any pending refund."""
    st.session_state.messages = []
    st.session_state.refund_state = RefundState()


# ---------- state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "refund_state" not in st.session_state:
    st.session_state.refund_state = RefundState()
if "user_id" not in st.session_state:
    st.session_state.user_id = config.DEMO_USER_ID   # FR-16: stands in for the signed-in user

# ---------- sidebar ----------
with st.sidebar:
    st.header("📶 NovaCell Support")
    st.caption("Answers come only from NovaCell's FAQ, resolved support cases and user guides. "
               "For refunds, I can look up your recent recharges.")
    st.subheader("Try a question")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        reset_conversation()
        st.rerun()
    with st.expander("Demo account (simulated)"):
        users = list(refund_data.USERS)
        chosen = st.selectbox("Signed-in customer", users, index=users.index(
            st.session_state.user_id) if st.session_state.user_id in users else 0)
        if chosen != st.session_state.user_id:
            st.session_state.user_id = chosen
            reset_conversation()
            st.rerun()
        acct = refund_data.USERS[chosen]
        st.caption(f"{acct['name']} · {acct['plan']} plan")
        st.dataframe(
            [{"recharge": r["recharge_id"], "amount": rupees(r["recharge_amount"]),
              "age (days)": r["recharge_age_days"], "status": r["recharge_status"],
              "refund": r["refund_status"]} for r in acct["recharges"]],
            hide_index=True, use_container_width=True)
        if st.button("Reset demo data", use_container_width=True):
            refund_data.reset_demo_data()
            reset_conversation()
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
            render_refund_confirmation(msg)
            render_feedback(msg)

# ---------- new turn ----------
question = st.chat_input("Type your question…") or st.session_state.pending
st.session_state.pending = None

if question:
    # FR-45/46: session history (with each answer's sources, for follow-up questions).
    history = [{"role": m["role"], "content": m["content"], "hits": m.get("hits")}
               for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        try:
            with st.spinner("Thinking…"):
                turn = bot.respond(question, history, st.session_state.refund_state,
                                   st.session_state.user_id)
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
        "needs_confirmation": bool(turn and turn.refund and turn.refund["needs_confirmation"]),
    })
    st.rerun()
