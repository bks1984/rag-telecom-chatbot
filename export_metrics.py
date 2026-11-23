"""Export the assistant's logs as two tidy datasets for the analytics dashboard.

    logs/interactions.jsonl -> metrics/turns.json         one row per answered message
    logs/agent.log          -> metrics/agent_events.json  one row per workflow step

Run:  .venv\\Scripts\\python export_metrics.py
Each file is {"rows": [...], "meta": [...]}; meta carries this script so a reader can see how
the rows were made. Timestamps are UTC. Customer answers are left out; questions are kept.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import config

OUT_DIR = config.ROOT / "metrics"

# agent.log lines: "YYYY-MM-DD HH:MM:SS msg" (current) or "HH:MM:SS msg" (before dates were logged)
_LINE = re.compile(r"^(?:(\d{4}-\d{2}-\d{2}) )?(\d{2}:\d{2}:\d{2}) (.*)$")
# Undated lines start on the day the refund agent was first run; the date rolls over whenever
# the clock goes backwards.
FIRST_UNDATED_DAY = date(2026, 11, 23)

STEPS = [  # (pattern, step name)
    (r"^\[Intent\] refund_request \(confirmation reply\)", "intent_confirmation_reply"),
    (r"^\[Intent\] refund_request \(answer to refund question\)", "intent_amount_reply"),
    (r"^\[Intent\] refund_request$", "intent_refund"),
    (r"^\[Intent\] requirement_inquiry$", "intent_inquiry"),
    (r"^\[Intent\] LLM classification failed", "intent_fallback"),
    (r"^\[Agent\] get_user_account", "account_lookup"),
    (r"^\[Agent\] validate_refund\(amount=(?P<amount>[\d.]+)\)", "validate"),
    (r"^\[Validation\] eligible=True", "eligible"),
    (r"^\[Validation\] eligible=False code=(?P<code>\w+)", "ineligible"),
    (r"^\[Agent\] awaiting_user_confirmation", "confirmation_asked"),
    (r"^\[User\] confirmed", "user_confirmed"),
    (r"^\[User\] rejected", "user_cancelled"),
    (r"^\[User\] ambiguous", "user_ambiguous"),
    (r"^\[User\] accepted suggested amount", "user_accepted_amount"),
    (r"^\[Agent\] process_refund\(amount=(?P<amount>[\d.]+)\)", "process_refund"),
    (r"^\[Refund\] success", "refund_processed"),
    (r"^\[Refund\] blocked", "refund_blocked"),
    (r"^\[Agent\] submit_to_support\(amount=(?P<amount>[\d.]+)\)", "submit_to_support"),
    (r"^\[Support\] submitted", "support_submitted"),
    (r"^\[Support\] already submitted", "support_duplicate"),
    (r"^\[Agent\] discarded unstated amount", "amount_discarded"),
]


def export_turns() -> list[dict]:
    turns, feedback = {}, {}
    for line in config.INTERACTION_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "question" not in r:                 # a later thumbs up/down click
            feedback[r["turn_id"]] = r["feedback"]
            continue
        turns[r["turn_id"]] = {
            "ts": r["ts"],
            "date": r["ts"][:10],
            "turn_id": r["turn_id"],
            "route": r["route"],
            "intent": r.get("intent") or "not_recorded",
            "refund_status": r.get("refund_status") or "",
            "escalated": bool(r.get("escalated")),
            "top_score": r.get("top_score"),
            "latency_s": r.get("latency_s"),
            "sources": "; ".join(dict.fromkeys(r.get("sources") or [])),
            "feedback": r.get("feedback") or "",
            "question": r["question"],
        }
    for turn_id, rating in feedback.items():
        if turn_id in turns:
            turns[turn_id]["feedback"] = rating
    return sorted(turns.values(), key=lambda t: t["ts"])


def export_agent_events() -> list[dict]:
    local_tz = datetime.now().astimezone().tzinfo
    rows, day, last_time = [], FIRST_UNDATED_DAY, None
    for line in config.AGENT_LOG.read_text(encoding="utf-8").splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        logged_day, clock, msg = m.groups()
        t = datetime.strptime(clock, "%H:%M:%S").time()
        if logged_day:
            day = date.fromisoformat(logged_day)
        elif last_time is not None and t < last_time:
            day += timedelta(days=1)
        last_time = t
        ts = datetime.combine(day, t, local_tz).astimezone(timezone.utc)
        for pattern, step in STEPS:
            s = re.match(pattern, msg)
            if s:
                g = s.groupdict()
                rows.append({
                    "ts": ts.isoformat(timespec="seconds"),
                    "date": ts.date().isoformat(),
                    "step": step,
                    "amount": float(g["amount"]) if g.get("amount") else None,
                    "code": g.get("code") or "",
                })
                break
    return rows


def write(name: str, rows: list[dict], source: Path) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    meta = [
        {"label": "export_metrics.py", "value": Path(__file__).read_text(encoding="utf-8"),
         "type": "code", "lang": "python"},
        {"label": "Input", "value": f"logs/{source.name}"},
        {"label": "Exported", "value": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    ]
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps({"rows": rows, "meta": meta}, ensure_ascii=False, indent=0),
                    encoding="utf-8")
    return path


if __name__ == "__main__":
    turns = export_turns()
    events = export_agent_events()
    print(write("turns", turns, config.INTERACTION_LOG), len(turns), "rows")
    print(write("agent_events", events, config.AGENT_LOG), len(events), "rows")
