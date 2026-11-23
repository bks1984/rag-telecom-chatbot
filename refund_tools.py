"""Refund tools with deterministic business rules (agent_PRD §6.4-6.7).

Only `get_user_account` and `validate_refund` are offered to the LLM. `process_refund` and
`submit_to_support` are called by application code once its own checks pass (FR-23a).
Every function takes the user ID from the session, never from the model or the customer.
"""
from __future__ import annotations

import logging
import uuid

import config
from refund_data import SUPPORT_QUEUE, USERS

log = logging.getLogger("novacell.agent")


def _user(user_id: str) -> dict | None:
    return USERS.get(user_id)


def is_refundable(recharge: dict) -> bool:
    """A recharge that could be refunded in full right now (FR-24, FR-25, FR-28)."""
    return (recharge["recharge_status"] == "completed"
            and recharge["recharge_age_days"] <= config.REFUND_WINDOW_DAYS
            and recharge["refund_status"] != "refunded")


def get_user_account(user_id: str) -> dict:
    """FR-19: the signed-in customer's plan and recharges."""
    log.info("[Agent] get_user_account")
    user = _user(user_id)
    if user is None:
        return {"found": False}
    return {
        "found": True,
        "name": user["name"],
        "plan": user["plan"],
        "plan_amount": user["plan_amount"],
        "recharges": [
            {**r, "refundable": is_refundable(r)}
            for r in sorted(user["recharges"], key=lambda r: r["recharge_age_days"])
        ],
    }


def select_recharge(recharges: list[dict], amount: float) -> dict | None:
    """FR-27a: pick the recharge a refund amount applies to, deterministically.

    1. A recharge whose amount matches exactly (even if already refunded); among several,
       prefer one not yet refunded, then the most recent.
    2. Otherwise the most recent refundable recharge large enough to cover the amount.
    3. Otherwise the largest recharge, so validation can say why it fails.
    """
    if not recharges:
        return None
    by_recency = sorted(recharges, key=lambda r: (r["recharge_age_days"], r["recharge_id"]))
    exact = [r for r in by_recency if r["recharge_amount"] == amount]
    if exact:
        return min(exact, key=lambda r: r["refund_status"] == "refunded")   # stable: keeps recency
    covering = [r for r in by_recency if is_refundable(r) and r["recharge_amount"] >= amount]
    if covering:
        return covering[0]
    return max(by_recency, key=lambda r: r["recharge_amount"])


def validate_refund(user_id: str, amount: float) -> dict:
    """FR-20, FR-24..FR-30: deterministic eligibility check. Decides eligibility only."""
    log.info("[Agent] validate_refund(amount=%s)", _fmt_num(amount))
    result = _validate(user_id, amount)
    log.info("[Validation] eligible=%s%s", result["eligible"],
             "" if result["eligible"] else f" code={result['code']}")
    return result


def _validate(user_id: str, amount: float) -> dict:
    def fail(code: str, reason: str, recharge: dict | None = None) -> dict:
        return {"eligible": False, "code": code, "reason": reason, "requested_amount": amount,
                "recharge_amount": recharge["recharge_amount"] if recharge else None,
                "recharge_id": recharge["recharge_id"] if recharge else None}

    user = _user(user_id)
    if user is None:
        return fail("no_account", "No account was found for this session.")
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return fail("invalid_amount", "The refund amount is not a valid number.")
    if amount == int(amount):
        amount = int(amount)
    if amount <= 0:                                                      # FR-26
        return fail("invalid_amount", "The refund amount must be greater than zero.")

    recharge = select_recharge(user["recharges"], amount)
    if recharge is None:
        return fail("no_recharge", "There is no recharge on this account to refund.")
    if recharge["refund_status"] == "refunded":                          # FR-28
        return fail("already_refunded", "This recharge has already been refunded.", recharge)
    if recharge["recharge_status"] != "completed":                       # FR-24
        return fail("not_completed", "The recharge has not completed, so it cannot be refunded.",
                    recharge)
    if recharge["recharge_age_days"] > config.REFUND_WINDOW_DAYS:        # FR-25
        return fail("outside_window",
                    f"The recharge is outside the {config.REFUND_WINDOW_DAYS}-day refund window.",
                    recharge)
    if amount > recharge["recharge_amount"]:                             # FR-27
        return fail("exceeds_recharge",
                    "The requested amount exceeds the recharge amount.", recharge)
    return {"eligible": True, "code": "eligible", "reason": "Refund is eligible.",
            "requested_amount": amount, "recharge_amount": recharge["recharge_amount"],
            "recharge_id": recharge["recharge_id"]}


def needs_support_review(amount: float) -> bool:
    """FR-36a/FR-37: the approval limit, enforced in code."""
    return amount > config.REFUND_APPROVAL_LIMIT


def process_refund(user_id: str, amount: float) -> dict:
    """FR-21, FR-23, FR-31..FR-35: simulated refund. Re-checks every rule before acting."""
    log.info("[Agent] process_refund(amount=%s)", _fmt_num(amount))
    check = _validate(user_id, amount)
    if not check["eligible"]:
        log.info("[Refund] blocked code=%s", check["code"])
        return {"success": False, "reason": check["reason"], "amount": amount}
    if needs_support_review(check["requested_amount"]):                  # GR-08
        log.info("[Refund] blocked: above approval limit")
        return {"success": False, "amount": amount,
                "reason": "Refunds above the approval limit must be reviewed by the Support Team."}
    recharge = next(r for r in USERS[user_id]["recharges"]
                    if r["recharge_id"] == check["recharge_id"])
    recharge["refund_status"] = "refunded"                               # FR-28a, FR-33
    refund_id = f"ref_{uuid.uuid4().hex[:6]}"
    log.info("[Refund] success refund_id=%s", refund_id)
    return {"success": True, "refund_id": refund_id, "amount": check["requested_amount"],
            "recharge_id": recharge["recharge_id"], "status": "refunded"}


def submit_to_support(user_id: str, amount: float) -> dict:
    """FR-21a, FR-38: hand an above-limit refund request to the Support Team. No outcome is
    simulated (GR-09) and the recharge's refund status is left unchanged."""
    log.info("[Agent] submit_to_support(amount=%s)", _fmt_num(amount))
    check = _validate(user_id, amount)
    if not check["eligible"]:
        return {"submitted": False, "reason": check["reason"]}
    for item in SUPPORT_QUEUE:
        if item["user_id"] == user_id and item["recharge_id"] == check["recharge_id"]:
            log.info("[Support] already submitted ticket=%s", item["ticket_id"])
            return {"submitted": True, "already_submitted": True, **item}
    item = {"ticket_id": f"sup_{uuid.uuid4().hex[:6]}", "user_id": user_id,
            "recharge_id": check["recharge_id"], "amount": check["requested_amount"],
            "status": "submitted_for_review"}
    SUPPORT_QUEUE.append(item)
    log.info("[Support] submitted ticket=%s", item["ticket_id"])
    return {"submitted": True, "already_submitted": False, **item}


def _fmt_num(x) -> str:
    try:
        x = float(x)
        return str(int(x)) if x == int(x) else str(x)
    except (TypeError, ValueError):
        return repr(x)
