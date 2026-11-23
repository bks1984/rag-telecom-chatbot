"""Offline tests for the refund capability (no network, scripted fake LLM).

Run:  .venv\\Scripts\\python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage  # noqa: E402

import refund_data  # noqa: E402
import refund_tools as tools  # noqa: E402
from assistant import Assistant  # noqa: E402
from chain import Turn  # noqa: E402
from intent import Intent  # noqa: E402
from refund_agent import (AWAITING_AMOUNT, AWAITING_CONFIRMATION, CANCELLED,  # noqa: E402
                          PROCESSED, REJECTED, SUBMITTED, RefundAgent, RefundState,
                          parse_confirmation, stated_amounts)

U = "user_123"


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c_{name}"}])


class FakeLLM:
    """Replays scripted AIMessages; records which tools were offered."""

    def __init__(self, script=(), intent="refund_request"):
        self.script, self.intent, self.offered = list(script), intent, []

    def bind_tools(self, tools_):
        self.offered = [t.name for t in tools_]
        return self

    def invoke(self, _msgs):
        return self.script.pop(0) if self.script else AIMessage(content="DONE")

    def with_structured_output(self, _schema):
        llm = self

        class _C:
            def invoke(self, _msgs):
                return Intent(intent=llm.intent)
        return _C()


class FakeBot:
    def __init__(self, llm):
        self.llm, self.asked = llm, []

    def ask(self, question, history=None):
        self.asked.append(question)
        return Turn(question, "answer", iter(["rag answer"]))


def account_then_validate(amount):
    return [call("get_user_account"), call("validate_refund", amount=amount)]


class ToolTests(unittest.TestCase):
    def setUp(self):
        refund_data.reset_demo_data()

    def test_eligible_499(self):
        v = tools.validate_refund(U, 499)
        self.assertTrue(v["eligible"])
        self.assertEqual(v["recharge_id"], "rch_457")

    def test_exceeds_recharge(self):
        v = tools.validate_refund(U, 2000)
        self.assertEqual(v["code"], "exceeds_recharge")
        self.assertEqual(v["recharge_amount"], 999)

    def test_zero_and_negative(self):
        self.assertEqual(tools.validate_refund(U, 0)["code"], "invalid_amount")
        self.assertEqual(tools.validate_refund(U, -5)["code"], "invalid_amount")

    def test_outside_window(self):
        self.assertEqual(tools.validate_refund("user_old", 299)["code"], "outside_window")

    def test_already_refunded(self):
        self.assertEqual(tools.validate_refund("user_refunded", 399)["code"], "already_refunded")

    def test_exact_match_selects_refunded_recharge(self):
        self.assertTrue(tools.process_refund(U, 499)["success"])
        # FR-27a: 499 still identifies rch_457, now refunded - not the ₹999 recharge.
        self.assertEqual(tools.validate_refund(U, 499)["code"], "already_refunded")
        self.assertTrue(tools.validate_refund(U, 999)["eligible"])

    def test_partial_amount_uses_covering_recharge(self):
        self.assertEqual(tools.validate_refund(U, 300)["recharge_id"], "rch_457")
        self.assertEqual(tools.validate_refund(U, 700)["recharge_id"], "rch_456")

    def test_process_refund_blocks_above_limit_and_invalid(self):
        self.assertFalse(tools.process_refund(U, 999)["success"])
        self.assertFalse(tools.process_refund(U, 2000)["success"])
        self.assertEqual(refund_data.USERS[U]["recharges"][0]["refund_status"], "not_refunded")

    def test_process_refund_updates_state(self):
        r = tools.process_refund(U, 499)
        self.assertTrue(r["success"])
        self.assertTrue(r["refund_id"].startswith("ref_"))
        self.assertEqual(refund_data.USERS[U]["recharges"][1]["refund_status"], "refunded")

    def test_submit_to_support_leaves_status_unchanged(self):
        r = tools.submit_to_support(U, 999)
        self.assertTrue(r["submitted"])
        self.assertEqual(refund_data.USERS[U]["recharges"][0]["refund_status"], "not_refunded")
        self.assertTrue(tools.submit_to_support(U, 999)["already_submitted"])


class ParsingTests(unittest.TestCase):
    def test_confirmations(self):
        for t in ["Yes, proceed", "yes", "Confirm", "go ahead", "Yes, proceed with the refund"]:
            self.assertEqual(parse_confirmation(t), "confirm", t)
        for t in ["No", "Cancel", "don't do it", "never mind"]:
            self.assertEqual(parse_confirmation(t), "reject", t)
        for t in ["maybe", "ok", "yes but refund 999 instead", "yes?", "I think so",
                  "yes, cancel", "what happens if I say yes"]:
            self.assertEqual(parse_confirmation(t), "ambiguous", t)

    def test_amounts(self):
        self.assertEqual(stated_amounts("refund ₹2,000 please"), {2000.0})
        self.assertEqual(stated_amounts("Rs.499"), {499.0})
        self.assertEqual(stated_amounts("rs 499 and INR 999"), {499.0, 999.0})
        self.assertEqual(stated_amounts("I want a refund"), set())


class AgentTests(unittest.TestCase):
    def setUp(self):
        refund_data.reset_demo_data()
        self.state = RefundState()

    def run_agent(self, script, message, new_request=True, history=()):
        llm = FakeLLM(script)
        reply = RefundAgent(llm).handle(message, list(history), self.state, U, new_request)
        return reply, llm

    def test_only_read_tools_exposed(self):
        _, llm = self.run_agent(account_then_validate(499), "I want a refund of ₹499.")
        self.assertEqual(sorted(llm.offered), ["get_user_account", "validate_refund"])

    def test_scenario1_confirm_then_process(self):
        reply, _ = self.run_agent(account_then_validate(499), "I want a refund of ₹499.")
        self.assertTrue(reply.needs_confirmation)
        self.assertEqual(self.state.status, AWAITING_CONFIRMATION)
        self.assertEqual(refund_data.USERS[U]["recharges"][1]["refund_status"], "not_refunded")
        reply, _ = self.run_agent([], "Yes, proceed", new_request=False)
        self.assertEqual(self.state.status, PROCESSED)
        self.assertIn("ref_", reply.text)
        self.assertEqual(refund_data.USERS[U]["recharges"][1]["refund_status"], "refunded")
        self.assertEqual(refund_data.USERS[U]["recharges"][0]["refund_status"], "not_refunded")

    def test_ambiguous_then_reject(self):
        self.run_agent(account_then_validate(499), "refund 499")
        reply, _ = self.run_agent([], "hmm maybe", new_request=False)
        self.assertEqual(self.state.status, AWAITING_CONFIRMATION)
        self.assertTrue(reply.needs_confirmation)
        self.run_agent([], "No", new_request=False)
        self.assertEqual(self.state.status, CANCELLED)
        self.assertEqual(refund_data.USERS[U]["recharges"][1]["refund_status"], "not_refunded")

    def test_scenario2_exceeds(self):
        reply, _ = self.run_agent(account_then_validate(2000), "Give me a refund of ₹2,000.")
        self.assertEqual(self.state.status, REJECTED)
        self.assertIn("more than your recharge", reply.text)

    def test_scenario6_above_limit_submits(self):
        reply, _ = self.run_agent(account_then_validate(999), "I want a refund of ₹999.")
        self.assertEqual(self.state.status, SUBMITTED)
        self.assertFalse(reply.needs_confirmation)
        self.assertIn("submitted to our Support Team", reply.text)
        for word in ("approved", "processed your", "rejected"):
            self.assertNotIn(word, reply.text)
        self.assertEqual(refund_data.USERS[U]["recharges"][0]["refund_status"], "not_refunded")

    def test_missing_amount_two_recharges_asks(self):
        reply, _ = self.run_agent([call("get_user_account"), AIMessage(content="DONE")],
                                  "I want a refund.")
        self.assertEqual(self.state.status, AWAITING_AMOUNT)
        self.assertIn("₹999", reply.text)
        self.assertIn("₹499", reply.text)

    def test_invented_amount_is_discarded(self):
        reply, _ = self.run_agent(account_then_validate(1234), "I want a refund.")
        self.assertEqual(self.state.status, AWAITING_AMOUNT)
        self.assertNotIn("1,234", reply.text)

    def test_unstated_amount_above_limit_needs_agreement(self):
        tools.process_refund(U, 499)                       # only ₹999 left
        reply, _ = self.run_agent(account_then_validate(999), "I want a refund.")
        self.assertEqual(self.state.status, AWAITING_AMOUNT)
        self.assertEqual(refund_data.SUPPORT_QUEUE, [])
        self.run_agent([], "yes", new_request=False)
        self.assertEqual(self.state.status, SUBMITTED)

    def test_fallback_validates_stated_amount_when_llm_skips_tools(self):
        reply, _ = self.run_agent([AIMessage(content="Sure!")], "refund 499 please")
        self.assertEqual(self.state.status, AWAITING_CONFIRMATION)

    def test_llm_outcome_claim_not_shown(self):
        reply, _ = self.run_agent([AIMessage(content="Your refund has been processed!")],
                                  "I want a refund.")
        self.assertNotIn("processed", reply.text)


class RouterTests(unittest.TestCase):
    def setUp(self):
        refund_data.reset_demo_data()

    def test_inquiry_goes_to_rag(self):
        llm = FakeLLM(intent="requirement_inquiry")
        a = Assistant(bot=FakeBot(llm))
        turn = a.respond("Why is my mobile internet slow?", [], RefundState())
        self.assertEqual(turn.intent, "requirement_inquiry")
        self.assertEqual(a.bot.asked, ["Why is my mobile internet slow?"])

    def test_refund_then_confirm_routes_to_agent(self):
        llm = FakeLLM(account_then_validate(499))
        a, state = Assistant(bot=FakeBot(llm)), RefundState()
        turn = a.respond("I want a refund of ₹499.", [], state)
        self.assertTrue(turn.refund["needs_confirmation"])
        llm.intent = "requirement_inquiry"                 # confirmation must not need the LLM
        a.respond("Yes, proceed", [], state)
        self.assertEqual(state.status, PROCESSED)
        self.assertEqual(a.bot.asked, [])


if __name__ == "__main__":
    unittest.main()
