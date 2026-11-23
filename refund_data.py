"""Hard-coded customer/account data for the refund demo (agent_PRD FR-14..FR-18a).

No CRM or billing system is involved. `USERS` is mutated in memory by the simulated
refund, and `reset_demo_data()` puts it back to the starting state.
"""
import copy

_SEED = {
    # The demonstration user (FR-18a): two recharges, ₹999 and ₹499.
    "user_123": {
        "name": "Rahul",
        "plan": "Pro",
        "plan_amount": 999,
        "recharges": [
            {"recharge_id": "rch_456", "recharge_amount": 999, "recharge_status": "completed",
             "recharge_age_days": 2, "refund_status": "not_refunded"},
            {"recharge_id": "rch_457", "recharge_amount": 499, "recharge_status": "completed",
             "recharge_age_days": 1, "refund_status": "not_refunded"},
        ],
    },
    # Extra accounts for demonstration scenarios 3 and 4.
    "user_old": {
        "name": "Priya",
        "plan": "Basic",
        "plan_amount": 299,
        "recharges": [
            {"recharge_id": "rch_310", "recharge_amount": 299, "recharge_status": "completed",
             "recharge_age_days": 15, "refund_status": "not_refunded"},
        ],
    },
    "user_refunded": {
        "name": "Arjun",
        "plan": "Basic",
        "plan_amount": 399,
        "recharges": [
            {"recharge_id": "rch_620", "recharge_amount": 399, "recharge_status": "completed",
             "recharge_age_days": 3, "refund_status": "refunded"},
        ],
    },
}

USERS: dict = copy.deepcopy(_SEED)

# Refund requests above the approval limit, handed to the Support Team (FR-38).
SUPPORT_QUEUE: list[dict] = []


def reset_demo_data() -> None:
    USERS.clear()
    USERS.update(copy.deepcopy(_SEED))
    SUPPORT_QUEUE.clear()
