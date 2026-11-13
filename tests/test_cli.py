"""Check the actual command-line entry point and its JSON input boundary."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(PROJECT / "cli.py"), *args],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=PROJECT,
        )

    def run_json(self, text, *args):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "instance.json"
            path.write_text(text, encoding="utf-8")
            return self.run_cli("--input", str(path), *args)

    def assert_input_error(self, result):
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_inline_both_methods(self):
        for method in ("dp", "bitset"):
            result = self.run_cli("--values", "6", "10", "15", "--budget", "17", "--method", method)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, f"method: {method}\nbudget: 17\nbest_sum: 16\n")

    def test_json_example(self):
        result = self.run_cli("--input", str(PROJECT / "examples" / "instance.json"), "--method", "bitset")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("best_sum: 16", result.stdout)

    def test_empty_inline_list(self):
        result = self.run_cli("--values", "--budget", "10")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("best_sum: 0", result.stdout)

    def test_invalid_inline_inputs(self):
        for args in (
            ("--values", "1"),
            ("--values", "1", "--budget", "-1"),
            ("--values", "-2", "--budget", "10"),
            ("--values", "1.5", "--budget", "10"),
            ("--values", "1", "--budget", "10", "--method", "unknown"),
            ("--budget", "10"),
        ):
            with self.subTest(args=args):
                self.assert_input_error(self.run_cli(*args))

    def test_invalid_json_documents(self):
        for text in ("{", "[]", "null", '{}', '{"values": []}', '{"budget": 1}'):
            with self.subTest(text=text):
                self.assert_input_error(self.run_json(text))

    def test_invalid_json_values(self):
        for values, budget in (([True], 10), ([1.0], 10), ([-1], 0), ([1], True), ("1", 10)):
            with self.subTest(values=values, budget=budget):
                self.assert_input_error(self.run_json(json.dumps({"values": values, "budget": budget})))

    def test_conflicting_sources_and_budget(self):
        path = str(PROJECT / "examples" / "instance.json")
        self.assert_input_error(self.run_cli("--input", path, "--values", "1", "--budget", "10"))
        self.assert_input_error(self.run_cli("--input", path, "--budget", "10"))

    def test_missing_input_file(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assert_input_error(self.run_cli("--input", str(Path(directory) / "missing.json")))

    def test_approximation_and_scaling_examples(self):
        result = self.run_cli("--values", "104", "102", "201", "101", "--budget", "308",
                              "--method", "approx", "--epsilon", "0.40")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("best_sum: 302", result.stdout)
        self.assertIn("epsilon: 0.40", result.stdout)
        for method in ("dp", "bitset"):
            result = self.run_cli("--values", "60", "100", "150", "--budget", "179",
                                  "--method", method, "--gcd-scale")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"method: {method}_gcd", result.stdout)
            self.assertIn("best_sum: 160", result.stdout)

    def test_invalid_approximation_options(self):
        for options in (("--epsilon", "0"), ("--epsilon", "1/10"), ("--gcd-scale",)):
            self.assert_input_error(self.run_cli("--values", "--budget", "0", "--method", "approx", *options))


if __name__ == "__main__":
    unittest.main()
