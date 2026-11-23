"""Top-level router (agent_PRD §7): intent detection in front of the existing RAG pipeline.

    requirement_inquiry -> SupportBot.ask (unchanged RAG pipeline)
    refund_request      -> RefundAgent
"""
from __future__ import annotations

import logging

import config
from chain import SupportBot, Turn
from intent import REFUND_REQUEST, classify_intent
from refund_agent import (AWAITING_AMOUNT, AWAITING_CONFIRMATION, RefundAgent, RefundState,
                          parse_confirmation, stated_amounts)

log = logging.getLogger("novacell.agent")


class Assistant:
    def __init__(self, bot: SupportBot | None = None):
        self.bot = bot or SupportBot()
        self.agent = RefundAgent(self.bot.llm)          # FR-04: reuse the existing LLM config

    @property
    def retriever(self):
        return self.bot.retriever

    def respond(self, message: str, history: list[dict] | None, state: RefundState,
                user_id: str = config.DEMO_USER_ID) -> Turn:
        message = message.strip()
        # FR-46, FR-50: a fixed window of recent messages goes to both routes.
        recent = (history or [])[-config.HISTORY_MESSAGES:]
        intent, new_request = self._route(message, recent, state)

        if intent != REFUND_REQUEST:
            if state.status == AWAITING_AMOUNT:
                state.reset()                            # customer moved on; drop the refund
            turn = self.bot.ask(message, recent)         # FR-02: existing pipeline, unchanged
            turn.intent = intent
            return turn

        reply = self.agent.handle(message, recent, state, user_id, new_request)
        return Turn(message, "refund", iter([reply.text]), intent=REFUND_REQUEST,
                    refund={"status": state.status, "amount": reply.amount,
                            "needs_confirmation": reply.needs_confirmation})

    def _route(self, message: str, history: list[dict], state: RefundState) -> tuple[str, bool]:
        """Returns (intent, new_request). A pending refund takes the customer's reply first."""
        if state.status == AWAITING_CONFIRMATION:
            if parse_confirmation(message) != "ambiguous":
                log.info("[Intent] %s (confirmation reply)", REFUND_REQUEST)
                return REFUND_REQUEST, False
            # A different refund request replaces the pending one; anything else is re-asked.
            intent = classify_intent(self.bot.llm, message, history)
            return REFUND_REQUEST, (intent == REFUND_REQUEST and bool(stated_amounts(message)))
        if state.status == AWAITING_AMOUNT and (stated_amounts(message)
                                               or parse_confirmation(message) != "ambiguous"):
            log.info("[Intent] %s (answer to refund question)", REFUND_REQUEST)
            return REFUND_REQUEST, False
        intent = classify_intent(self.bot.llm, message, history)
        return intent, (intent == REFUND_REQUEST and state.status != AWAITING_AMOUNT)
