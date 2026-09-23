import contextlib
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from aithon.agent import ToolAgent
from aithon.capabilities import CapabilityPermissionError
from aithon.cli import run_script
from aithon.models import ProfileConfig, ProviderError, ResolvedConfig


def call(identifier, name, **arguments):
    return {"id": identifier, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments)}}


def response(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


class ExpressionResultTests(unittest.TestCase):
    def run_source(self, source, provider, permissions=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.py"
            path.write_text(source)
            profile = ProfileConfig("default", "fake", "model", max_rounds=2)
            if permissions is not None:
                profile = replace(profile, permissions=permissions)
            config = ResolvedConfig(None, path.parent, "default", {"default": profile})
            return run_script(path, config=config, agent_factory=lambda _: ToolAgent(provider))

    def test_circle_finishes_with_one_call_and_no_handle_round_trip(self):
        provider = Mock()
        provider.complete.return_value = response(
            call("done", "finish", code="__import__('math').pi * float(r)**2"))
        with patch("builtins.input", return_value="22"):
            result = self.run_source('r = input("Enter a number: ")\narea = calculate the circle area from r', provider)
        self.assertAlmostEqual(result["area"], math.pi * 22**2)
        self.assertEqual(provider.complete.call_count, 1)

    def test_blank_unused_finish_fields_do_not_cause_another_model_call(self):
        provider = Mock()
        provider.complete.return_value = response(call(
            "done", "finish", code="'Refunds are available for 30 days.'",
            handle="", result_from=""))

        result = self.run_source("answer: str = answer the refund question", provider)

        self.assertEqual(result["answer"], "Refunds are available for 30 days.")
        self.assertEqual(provider.complete.call_count, 1)

    def test_literal_text_is_returned_without_python_quoting_or_eval(self):
        provider = Mock()
        answer = 'The customer said "hello".\n(policy.md, lines 5–7)'
        provider.complete.return_value = response(call("done", "finish", value=answer))

        result = self.run_source("answer: str = answer the refund question", provider)

        self.assertEqual(result["answer"], answer)
        self.assertEqual(provider.complete.call_count, 1)

    def test_literal_values_distinguish_null_from_missing_result(self):
        for value in (None, False, 0, "", [1, {"ok": True}]):
            with self.subTest(value=value):
                provider = Mock()
                provider.complete.return_value = response(call("done", "finish", value=value))
                result = self.run_source("answer = return a value please", provider)
                self.assertEqual(result["answer"], value)
                self.assertEqual(provider.complete.call_count, 1)

    def test_blank_result_fields_still_require_a_real_expression(self):
        provider = Mock()
        provider.complete.side_effect = [
            response(call("blank", "finish", code=" ", handle="", result_from="")),
            response(call("fixed", "finish", code="''")),
        ]

        result = self.run_source("answer: str = answer the question", provider)

        self.assertEqual(result["answer"], "")
        self.assertEqual(provider.complete.call_count, 2)
        replies = [json.loads(message["content"]) for message in provider.complete.call_args.args[0]
                   if message["role"] == "tool"]
        self.assertIn("requires a result outcome", replies[0]["message"])

    def test_multistep_fibonacci_uses_live_python_in_one_invocation(self):
        provider = Mock()
        provider.complete.return_value = response(
            call("calculate", "execute", code="a, b = 0, 1\nfor _ in range(n):\n    a, b = b, a + b"),
            call("done", "finish", code="a"))
        result = self.run_source("n = 30\nanswer: int = calculate the Fibonacci number at n\n", provider)
        self.assertEqual(result["answer"], 832040)
        self.assertEqual(provider.complete.call_count, 1)

    def test_direct_result_preserves_identity(self):
        provider = Mock()
        provider.complete.return_value = response(call("done", "finish", code="items"))
        result = self.run_source("items = []\nanswer = return these items please", provider)
        self.assertIs(result["answer"], result["items"])

    def test_sphere_identifier_recovery_in_one_tool_call(self):
        provider = Mock()
        provider.complete.return_value = response(call("done", "recover", action="complete",
            code="4 * __import__('math').pi * float(r)**2"))
        with patch("builtins.input", return_value="23"):
            result = self.run_source('r = input("Enter a number: ")\narea = sphere_surface_area', provider)
        self.assertAlmostEqual(result["area"], 4 * math.pi * 23**2)
        self.assertEqual(provider.complete.call_count, 1)
        payload = json.loads(provider.complete.call_args.args[0][1]["content"])
        self.assertEqual(payload["mode"], "recovery")
        self.assertEqual(payload["replacement_target"], "area")

    def test_blank_unused_recovery_fields_do_not_cause_another_model_call(self):
        provider = Mock()
        provider.complete.return_value = response(call(
            "done", "recover", action="complete", code="42", handle="", result_from=""))

        result = self.run_source("answer = undefined_answer", provider)

        self.assertEqual(result["answer"], 42)
        self.assertEqual(provider.complete.call_count, 1)

    def test_recovery_can_return_literal_without_code_execution(self):
        provider = Mock()
        provider.complete.return_value = response(call("done", "recover", action="complete", value="recovered"))

        result = self.run_source("answer = undefined_answer", provider)

        self.assertEqual(result["answer"], "recovered")
        self.assertEqual(provider.complete.call_count, 1)

    def test_direct_result_still_checks_types_and_repairs(self):
        provider = Mock()
        provider.complete.side_effect = [response(call("bad", "finish", code="'42'")),
                                         response(call("fixed", "finish", code="42"))]
        self.assertEqual(self.run_source("answer: int = calculate an answer please", provider)["answer"], 42)
        replies = [json.loads(m["content"]) for m in provider.complete.call_args.args[0] if m["role"] == "tool"]
        self.assertEqual(replies[0]["error"], "TypeViolation")

    def test_literal_result_still_checks_types_and_repairs(self):
        provider = Mock()
        provider.complete.side_effect = [response(call("bad", "finish", value="42")),
                                         response(call("fixed", "finish", value=42))]
        self.assertEqual(self.run_source("answer: int = calculate an answer please", provider)["answer"], 42)
        replies = [json.loads(m["content"]) for m in provider.complete.call_args.args[0] if m["role"] == "tool"]
        self.assertEqual(replies[0]["error"], "TypeViolation")

    def test_direct_result_still_requires_execute_permission(self):
        provider = Mock()
        provider.complete.return_value = response(call("done", "finish", code="42"))
        with self.assertRaises(CapabilityPermissionError):
            self.run_source("answer = calculate an answer please", provider, permissions=("network",))

    def test_ambiguous_terminal_rejected_before_code_executes(self):
        provider = Mock()
        provider.complete.side_effect = [response(call("bad", "finish", code="items.append(1)", handle="bad")),
                                         response(call("fixed", "finish", code="None"))]
        result = self.run_source("items = []\nanswer = do something please", provider)
        self.assertEqual(result["items"], [])
        self.assertIsNone(result["answer"])

    def test_literal_and_code_conflict_rejected_before_side_effects(self):
        provider = Mock()
        provider.complete.side_effect = [response(
            call("change", "execute", code="items.append(1)"),
            call("bad", "finish", value=None, code="items.append(2)")),
            response(call("fixed", "finish", value=None))]

        result = self.run_source("items = []\nanswer = return no value please", provider)

        self.assertEqual(result["items"], [])
        self.assertIsNone(result["answer"])

    def test_terminal_error_stops_without_another_model_call(self):
        provider = Mock()
        provider.complete.return_value = response(call(
            "unavailable", "finish", error="No speech generation route is configured"))

        with self.assertRaisesRegex(ProviderError, "No speech generation route is configured"):
            self.run_source("speech = speak this aloud", provider)

        self.assertEqual(provider.complete.call_count, 1)

    def test_terminal_error_cannot_be_combined_with_a_result(self):
        provider = Mock()
        provider.complete.side_effect = [
            response(call("bad", "finish", error="unavailable", value="pretend success")),
            response(call("fixed", "finish", value="real answer")),
        ]

        result = self.run_source("answer = answer a question please", provider)

        self.assertEqual(result["answer"], "real answer")
        self.assertEqual(provider.complete.call_count, 2)

    def test_circle_input_returns_number_in_one_response(self):
        provider = Mock()
        provider.complete.return_value = response(
            call("calculate", "evaluate", code="__import__('math').pi * float(r) ** 2", result_id="area"),
            call("done", "finish", result_from="area"))
        output = io.StringIO()
        with patch("builtins.input", return_value="22"), contextlib.redirect_stdout(output):
            result = self.run_source('''r = input("Enter a number: ")
area = calculate the circle area from r
print("The area of the circle is:", area)
''', provider)
        self.assertAlmostEqual(result["area"], math.pi * 22 ** 2)
        self.assertIn(str(result["area"]), output.getvalue())
        payload = json.loads(provider.complete.call_args.args[0][1]["content"])
        self.assertTrue(payload["requires_result"])
        self.assertEqual(provider.complete.call_count, 1)

    def test_missing_result_rejects_entire_batch_then_repairs(self):
        provider = Mock()
        provider.complete.side_effect = [response(
            call("side_effect", "execute", code="items.append('must not run'); area = 123"),
            call("missing", "finish")), response(
            call("calculate", "evaluate", code="123", result_id="answer"),
            call("done", "finish", result_from="answer"))]
        result = self.run_source("items = []\narea = calculate a value please\n", provider)
        self.assertEqual(result["area"], 123)
        self.assertEqual(result["items"], [])
        replies = [json.loads(m["content"]) for m in provider.complete.call_args.args[0] if m["role"] == "tool"]
        self.assertEqual([m.get("status") for m in replies[:2]], ["skipped", "error"])
        self.assertIn("requires a result outcome", replies[1]["message"])

    def test_expression_contexts_require_results_without_annotations(self):
        for source in (
            "answer = calculate something please",
            "def work():\n    return calculate something please\nanswer = work()",
            "answer = str(calculate something please)",
            "if determine condition please:\n    answer = 1",
        ):
            with self.subTest(source=source):
                provider = Mock()
                provider.complete.side_effect = [response(call("missing1", "finish")),
                                                 response(call("missing2", "finish"))]
                with self.assertRaisesRegex(ProviderError, "repeated invalid tool batches"):
                    self.run_source(source, provider)

    def test_explicit_none_is_a_valid_expression_value(self):
        provider = Mock()
        provider.complete.return_value = response(
            call("none", "evaluate", code="None"), call("done", "finish", result_from="none"))
        self.assertIsNone(self.run_source("answer = return no value please", provider)["answer"])

    def test_standalone_side_effect_needs_no_result(self):
        provider = Mock()
        provider.complete.return_value = response(
            call("change", "execute", code="items.append(1)"), call("done", "finish"))
        self.assertEqual(self.run_source("items = []\nappend one to items please", provider)["items"], [1])
        payload = json.loads(provider.complete.call_args.args[0][1]["content"])
        self.assertFalse(payload["requires_result"])
