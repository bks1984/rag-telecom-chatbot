"""Rule-based checks that run before (and around) the LLM.

- Urgent topics (lost/stolen phone, fraud, emergencies) skip troubleshooting and go to a human.
- Explicit requests for a person are honoured immediately.
- Questions that need the customer's own account data get general guidance only;
  the bot never states balances, amounts, dates or plan details.
- Low retrieval confidence means the bot declines instead of guessing.
"""
import re
from dataclasses import dataclass

import config

URGENT = re.compile(
    r"\b(lost|stolen|theft|stole)\b.*\b(phone|sim|device|mobile)\b"
    r"|\b(phone|sim|device|mobile)\b.*\b(lost|stolen)\b"
    r"|\bfraud\w*|\bscam\w*|\bhack(ed|er)?\b|\bsim[- ]?swap\b|\bunauthori[sz]ed\b"
    r"|\bemergency\b|\b911\b|\b112\b",
    re.IGNORECASE,
)

HUMAN_REQUEST = re.compile(
    r"\b(talk|speak|chat|connect)\b.{0,20}\b(human|person|agent|someone|representative|operator)\b"
    r"|^\s*(human|agent|representative|operator)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

ACCOUNT_DATA = re.compile(
    r"\b(my|mine)\b.{0,25}\b(balance|bill|invoice|charge[sd]?|usage|plan|payment|refund|"
    r"account|data left|minutes left|due date|contract|credit)\b"
    r"|\bhow much (do i|did i|have i|i)\b"
    r"|\bwhy (was|am|have|did) i (charged|billed)\b"
    r"|\bwhen (is|was) my\b",
    re.IGNORECASE,
)


@dataclass
class Route:
    kind: str                 # "urgent" | "human" | "answer"
    account_data: bool = False


def classify(question: str) -> Route:
    if URGENT.search(question):
        return Route("urgent")
    if HUMAN_REQUEST.search(question):
        return Route("human")
    return Route("answer", account_data=bool(ACCOUNT_DATA.search(question)))


URGENT_REPLY = (
    "This needs a person right away, so I won't troubleshoot it here.\n\n"
    "- **Lost or stolen phone / suspected fraud:** call **611** now "
    "(from another phone if yours is missing) so the line can be blocked.\n"
    "- **Emergency:** call your local emergency number.\n\n"
    "An agent can verify your identity and act on your account; I can't."
)

HUMAN_REPLY = (
    f"Sure - to talk to a person, {config.ESCALATION_CONTACT}. "
    "Mention what you've already tried so the agent can pick up from there."
)

LOW_CONFIDENCE_REPLY = (
    "I'm not confident I have an answer to that in NovaCell's help material, "
    "and I don't want to guess.\n\n"
    f"For help, {config.ESCALATION_CONTACT}. "
    "You can also try rephrasing your question with more detail (device, what you see on screen)."
)
