"""Generator, worker isolation, fault retention, saved records and replay checks."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import benchmark


PROJECT = Path(__file__).resolve().parents[1]
SETTINGS = dict(repeats=2, max_dp_work=20_000_000, max_bitset_states=10_000_000,
                timeout_seconds=10, memory_timeout_seconds=10, target_batch_ns=100_000,
                max_batch_count=32)


def tiny_manifest():
    cases = []
    for factor in (1, 10):
        cases.append(dict(case_id=f"tiny-g{factor}", experiment="D", family="gcd", seed=319,
                          values=[factor * v for v in (6, 10, 15)], budget=17 * factor + factor - 1,
                          scale_factor=factor, base_id="tiny",
                          configurations=[benchmark.configuration(m) for m in
                                          ("bitset", "dp", "bitset_gcd", "dp_gcd")]))
    cases.append(dict(case_id="literature", experiment="example", family="literature", seed=None,
                      values=[104, 102, 201, 101], budget=308,
                      configurations=[benchmark.configuration("bitset"), benchmark.configuration("approx", "0.40")]))
    return dict(schema_version=1, profile="quick", seeds=[319], instances=cases)


class GeneratorTests(unittest.TestCase):
    def test_seeded_grid_and_quick_full_overlap(self):
        quick = benchmark.generate_instances("quick")
        full = benchmark.generate_instances("full")
        self.assertEqual(quick, benchmark.generate_instances("quick"))
        self.assertEqual(len(quick["instances"]), 9)
        self.assertEqual(len(full["instances"]), 40)
        self.assertEqual(sum(len(c["configurations"]) for c in quick["instances"]), 28)
        self.assertEqual(sum(len(c["configurations"]) for c in full["instances"]), 140)
        full_by_id = {c["case_id"]: c for c in full["instances"]}
        for case in quick["instances"]:
            other = full_by_id[case["case_id"]]
            self.assertEqual(case["values"], other["values"])
            self.assertEqual(case["budget"], other["budget"])
            self.assertGreater(sum(case["values"]), case["budget"])
            self.assertTrue(all(0 < v <= case["budget"] for v in case["values"]))
        benchmark.validate_manifest(full)

    def test_gcd_families_preserve_the_base_instance(self):
        cases = [c for c in benchmark.generate_instances("full")["instances"] if c["experiment"] == "D"]
        for seed in (319, 320, 321):
            family = [c for c in cases if c["seed"] == seed]
            base = family[0]["values"]
            self.assertEqual(base[:2], [20, 21])
            for case in family:
                factor = case["scale_factor"]
                self.assertEqual(case["values"], [v * factor for v in base])
                self.assertEqual(case["budget"], factor * 1500 + factor - 1)
                self.assertEqual(benchmark.instance_details(case, "dp_gcd")["solver_budget"], 1500)

    def test_eligibility_caps_use_the_scaled_budget(self):
        case = dict(values=[60, 100, 150], budget=179)
        plain = benchmark.instance_details(case, "dp")
        scaled = benchmark.instance_details(case, "dp_gcd")
        self.assertEqual(scaled["gcd"], 10)
        self.assertEqual(scaled["solver_budget"], 17)
        self.assertIn("DP work", benchmark.eligibility_reason(plain, "dp", 100, 100))
        self.assertEqual(benchmark.eligibility_reason(scaled, "dp_gcd", 100, 100), "")
        self.assertIn("bitset states", benchmark.eligibility_reason(plain, "bitset", 100, 18))
        self.assertEqual(benchmark.eligibility_reason(scaled, "bitset_gcd", 100, 18), "")
        trivial = benchmark.instance_details(dict(values=[1, 2], budget=10**100), "dp")
        self.assertEqual(benchmark.eligibility_reason(trivial, "dp", 0, 0), "")

    def test_manifest_errors_are_rejected_before_measurement(self):
        for alteration in ("duplicate", "bad_epsilon", "bad_values", "bad_budget", "bad_method"):
            manifest = tiny_manifest()
            if alteration == "duplicate":
                manifest["instances"].append(manifest["instances"][0])
            elif alteration == "bad_epsilon":
                manifest["instances"][-1]["configurations"][-1]["epsilon"] = "1/10"
            elif alteration == "bad_values":
                manifest["instances"][0]["values"] = [True]
            elif alteration == "bad_budget":
                manifest["instances"][0]["budget"] = -1
            else:
                manifest["instances"][0]["configurations"][0]["method"] = "unknown"
            with self.subTest(alteration=alteration):
                with self.assertRaises((TypeError, ValueError)):
                    benchmark.validate_manifest(manifest)


class WorkerTests(unittest.TestCase):
    def payload(self):
        return dict(values=[6, 10, 15], budget=17, method="bitset", epsilon="",
                    repeats=2, target_batch_ns=100_000, max_batch_count=32)

    def test_real_timing_and_memory_workers(self):
        timing = benchmark.run_worker("timing", self.payload(), 10)
        self.assertEqual(timing["status"], "ok", timing)
        samples = [event for event in timing["events"] if event["type"] == "sample"]
        self.assertEqual(len(samples), 2)
        counts = {sample["batch_count"] for sample in samples}
        self.assertEqual(len(counts), 1)
        for sample in samples:
            self.assertEqual(sample["answer"], 16)
            self.assertGreater(sample["batch_duration_ns"], 0)
            self.assertEqual(sample["duration_per_call_ns"], sample["batch_duration_ns"] / sample["batch_count"])
        memory = benchmark.run_worker("memory", self.payload(), 10)
        self.assertEqual(memory["status"], "ok", memory)
        measurement = memory["events"][0]
        self.assertEqual(measurement["answer"], 16)
        self.assertGreater(measurement["peak_traced_bytes"], 0)

    def test_real_worker_errors_and_timeout(self):
        payload = self.payload()
        payload["method"] = "unknown"
        error = benchmark.run_worker("timing", payload, 10)
        self.assertEqual(error["status"], "error")
        self.assertIn("method", error["reason"])
        timeout = benchmark.run_worker("timing", self.payload(), 0.000001)
        self.assertEqual(timeout["status"], "timeout")
        self.assertFalse(any(event["type"] == "complete" for event in timeout["events"]))

    def test_memory_timeout_preserves_valid_timing(self):
        samples = [dict(type="sample", repeat=i, batch_count=1, batch_duration_ns=100,
                        duration_per_call_ns=100, answer=16) for i in (1, 2)]
        responses = [dict(status="ok", reason="", events=samples),
                     dict(status="timeout", reason="injected memory timeout", events=[])]
        with patch("benchmark.run_worker", side_effect=responses):
            row, raw, _ = benchmark.measure_configuration(tiny_manifest()["instances"][0],
                                                         benchmark.configuration("bitset"), SETTINGS)
        self.assertEqual(row["median_runtime_ns"], 100)
        self.assertEqual(row["answer"], 16)
        self.assertEqual(row["memory_status"], "timeout")
        self.assertEqual(row["peak_traced_bytes"], "")
        self.assertEqual(len(raw), 2)

    def test_partial_timing_timeout_keeps_raw_sample_without_median(self):
        sample = dict(type="sample", repeat=1, batch_count=1, batch_duration_ns=100,
                      duration_per_call_ns=100, answer=16)
        with patch("benchmark.run_worker", return_value=dict(status="timeout", reason="timeout", events=[sample])) as worker:
            row, raw, _ = benchmark.measure_configuration(tiny_manifest()["instances"][0],
                                                         benchmark.configuration("bitset"), SETTINGS)
        self.assertEqual(worker.call_count, 1)
        self.assertEqual(row["timing_status"], "timeout")
        self.assertEqual(row["median_runtime_ns"], "")
        self.assertEqual(row["memory_status"], "not_run")
        self.assertEqual([record["status"] for record in raw], ["ok", "timeout"])

    def test_failed_configuration_does_not_lose_later_results(self):
        manifest = tiny_manifest()
        manifest["instances"] = manifest["instances"][:1]
        case = manifest["instances"][0]
        case["configurations"] = [benchmark.configuration("bitset"), benchmark.configuration("dp")]
        original = benchmark.run_worker

        def fail_bitset(mode, payload, timeout):
            if payload["method"] == "bitset":
                return dict(status="error", reason="injected failure", events=[])
            return original(mode, payload, timeout)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            with patch("benchmark.run_worker", side_effect=fail_bitset), redirect_stdout(io.StringIO()):
                code = benchmark.run_benchmarks(manifest, out, SETTINGS, ["benchmark.py", "test"])
            self.assertEqual(code, 1)
            with (out / "summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["timing_status"] for row in rows], ["error", "ok"])
            self.assertEqual(rows[1]["exact_optimum"], "16")
            self.assertEqual(json.loads((out / "environment.json").read_text())["run_status"], "completed_with_failures")


class VerificationTests(unittest.TestCase):
    def row(self, method, answer, epsilon=""):
        row = {key: "" for key in benchmark.SUMMARY_FIELDS}
        row.update(case_id="test", method=method, answer=answer, epsilon=epsilon,
                   original_budget=308, timing_status="ok", memory_status="ok", median_runtime_ns=100)
        return row

    def test_missing_reference_zero_optimum_and_error_use_optimum(self):
        cases = {"test": dict(experiment="example")}
        approximation = self.row("approx", 302, "0.40")
        benchmark.verify_rows([approximation], cases)
        self.assertEqual(approximation["verification_status"], "no_reference")
        self.assertEqual(approximation["relative_error_pct"], "")
        exact = self.row("bitset", 307)
        benchmark.verify_rows([approximation, exact], cases)
        self.assertAlmostEqual(approximation["relative_error_pct"], 100 * 5 / 307)
        self.assertEqual(approximation["verification_status"], "ok")
        zero = self.row("bitset", 0)
        benchmark.verify_rows([zero], cases)
        self.assertEqual(zero["relative_error_pct"], 0)

    def test_disagreement_guarantee_and_memory_failures_stay_visible(self):
        cases = {"test": dict(experiment="example")}
        exact = self.row("bitset", 307)
        wrong = self.row("dp", 306)
        approximate = self.row("approx", 100, "0.10")
        wrong["verification_reason"] = "memory and timing answers differ"
        for _ in range(2):
            benchmark.verify_rows([exact, wrong, approximate], cases)
            self.assertEqual(wrong["verification_status"], "failed")
            self.assertIn("memory and timing", wrong["verification_reason"])
            self.assertIn("exact methods disagree", wrong["verification_reason"])
            self.assertEqual(approximate["verification_status"], "failed")


class BenchmarkCliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(PROJECT / "benchmark.py"), *map(str, args)],
                              capture_output=True, text=True, timeout=30)

    def test_saved_outputs_replay_and_existing_directory_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps(tiny_manifest()))
            out = root / "first"
            args = ["--instances", source, "--out", out, "--repeats", "2",
                    "--batch-target-ms", "0.1", "--max-batch-count", "32"]
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            with (out / "summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            with (out / "timings.csv").open() as stream:
                timings = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 10)
            self.assertEqual(len(timings), 20)
            self.assertTrue(all(row["verification_status"] == "ok" for row in rows))
            self.assertTrue(all(row["memory_status"] == "ok" for row in rows))
            approx = next(row for row in rows if row["method"] == "approx")
            self.assertEqual(approx["answer"], "302")
            self.assertEqual(approx["exact_optimum"], "307")
            for row in rows:
                if row["method"].endswith("_gcd"):
                    self.assertGreater(float(row["preprocessing_speedup"]), 0)
            metadata = json.loads((out / "environment.json").read_text())
            self.assertEqual(metadata["run_status"], "completed")
            self.assertIn("solvers.py", metadata["source_hashes"])
            self.assertEqual(len(metadata["calibration"]), 10)

            before = (out / "summary.csv").read_bytes()
            rejected = self.run_cli(*args)
            self.assertEqual(rejected.returncode, 2)
            self.assertEqual(before, (out / "summary.csv").read_bytes())
            replay = root / "replay"
            result = self.run_cli("--instances", out / "instances.json", "--out", replay,
                                  "--max-dp-work", "0", "--max-bitset-states", "0",
                                  "--batch-target-ms", "0.1", "--max-batch-count", "32")
            self.assertEqual(result.returncode, 0, result.stderr)
            with (replay / "summary.csv").open() as stream:
                skipped = list(csv.DictReader(stream))
            self.assertEqual(sum(row["timing_status"] == "skipped" for row in skipped), 9)
            remaining = next(row for row in skipped if row["method"] == "approx")
            self.assertEqual(remaining["verification_status"], "no_reference")
            self.assertEqual(remaining["exact_optimum"], "")
            self.assertEqual(remaining["relative_error_pct"], "")
            self.assertEqual(json.loads((replay / "instances.json").read_text()), tiny_manifest())

    def test_invalid_options(self):
        for option, value in (("--repeats", "0"), ("--timeout", "nan"), ("--memory-timeout", "-1"),
                              ("--max-dp-work", "-1"), ("--max-batch-count", "0")):
            result = self.run_cli("--out", "unused", option, value)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
