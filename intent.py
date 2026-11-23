"""Intent detection (agent_PRD §6.1): `requirement_inquiry` or `refund_request`."""
from __future__ import annotations

import logging
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

log = logging.getLogger("novacell.agent")

REQUIREMENT_INQUIRY = "requirement_inquiry"
REFUND_REQUEST = "refund_request"

INTENT_PROMPT = """You route messages for a telecom customer-care chatbot. Classify the \
customer's LATEST message into exactly one intent:

- refund_request: the customer wants money back, or wants a recharge/payment cancelled, \
reversed or refunded now. Also a reply that continues a refund conversation already under way \
(for example giving the amount the assistant asked for).
- requirement_inquiry: everything else - questions about service, network, billing, plans, \
roaming, SIM/eSIM, the app, and questions about HOW refunds or cancellations work in general.

A request to cancel or reverse a recharge is refund_request. A question about how to cancel \
is requirement_inquiry.

Examples:
"Why is my internet slow?" -> requirement_inquiry
"I want a refund of ₹999." -> refund_request
"Please reverse my last recharge" -> refund_request
"How do I cancel a recharge?" -> requirement_inquiry
"What is the refund policy?" -> requirement_inquiry

Use the earlier conversation only to understand the latest message."""

# Fallback if the LLM call fails: an action verb aimed at a refund/recharge, not a "how" question.
_REFUND_FALLBACK = re.compile(
    r"\b(refund|money back|reverse|cancel|undo)\b", re.IGNORECASE)
_HOW_QUESTION = re.compile(r"^\s*(how|what|when|where|can|is|are|do|does|why)\b", re.IGNORECASE)


class Intent(BaseModel):
    intent: Literal["requirement_inquiry", "refund_request"] = Field(
        description="The intent of the customer's latest message")


def _history_messages(history: list[dict]) -> list:
    return [(HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
            for m in history]


def classify_intent(llm, message: str, history: list[dict] | None = None) -> str:
    """FR-01, FR-04, FR-05: structured intent from the existing Groq LLM."""
    try:
        classifier = llm.with_structured_output(Intent)
        result = classifier.invoke([SystemMessage(content=INTENT_PROMPT),
                                    *_history_messages(history or []),
                                    HumanMessage(content=message)])
        intent = result.intent
    except Exception as exc:                                  # API / parsing failure
        log.warning("[Intent] LLM classification failed (%s); using keyword fallback",
                    type(exc).__name__)
        intent = (REFUND_REQUEST if _REFUND_FALLBACK.search(message)
                  and not _HOW_QUESTION.search(message) else REQUIREMENT_INQUIRY)
    log.info("[Intent] %s", intent)
    return intent
