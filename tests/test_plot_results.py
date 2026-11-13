"""Check saved-data audits, missing values, aggregation and plot regeneration."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from benchmark import SUMMARY_FIELDS, TIMING_FIELDS
import plot_results as plots


def fixture(directory):
    """Small artificial measurements for tests only; answers have known subsets."""
    cases = []
    for experiment, family, seed, budget in (("A", "count", 319, 17), ("A", "count", 320, 17),
                                            ("B", "budget", 319, 17), ("B", "budget", 319, 18),
                                            ("C", "broad", 319, 17), ("C", "clustered", 319, 17)):
        methods = [("bitset", ""), ("dp", ""), ("approx", "0.10")]
        if experiment == "C":
            methods = [("bitset", ""), ("approx", "0.10"), ("approx", "0.40")]
        cases.append(dict(case_id=f"{experiment}-{family}-{seed}-{budget}", experiment=experiment,
                          family=family, seed=seed, budget=budget, values=[6, 10, 15],
                          configurations=[dict(method=m, epsilon=e) for m, e in methods]))
    manifest = dict(schema_version=1, profile="quick", seeds=[319, 320], instances=cases)
    (directory / "instances.json").write_text(json.dumps(manifest))
    environment = dict(run_status="completed", profile="quick", seeds=[319, 320],
                       instances_sha256=plots.digest(directory / "instances.json"),
                       settings=dict(repeats=1, max_dp_work=100, max_bitset_states=1000))
    (directory / "environment.json").write_text(json.dumps(environment))
    rows, raw = [], []
    for case in cases:
        for config in case["configurations"]:
            row = {key: "" for key in SUMMARY_FIELDS}
            row.update({key: case[key] for key in ("case_id", "experiment", "family", "seed")})
            row.update(config)
            row.update(original_n=3, original_budget=case["budget"], effective_n=3,
                       effective_budget=case["budget"], solver_budget=case["budget"], gcd=1,
                       timing_status="ok", memory_status="ok", verification_status="ok", scaling_check="not_applicable",
                       answer=16, exact_optimum=16, relative_error_pct=0, reference_source="bitset",
                       median_runtime_ns=1_000_000 if case["seed"] == 319 else 3_000_000,
                       peak_traced_bytes=1024)
            sample = {key: "" for key in TIMING_FIELDS}
            sample.update({key: row[key] for key in ("case_id", "experiment", "family", "seed", "method", "epsilon")})
            if case["experiment"] == "B" and case["budget"] == 18 and config["method"] == "dp":
                row.update(timing_status="skipped", memory_status="skipped", verification_status="not_checked",
                           answer="", median_runtime_ns="", peak_traced_bytes="", exact_optimum="", relative_error_pct="",
                           timing_reason="test cap", memory_reason="test cap")
                sample.update(status="skipped", reason="test cap")
            else:
                sample.update(repeat=1, batch_count=2, batch_duration_ns=2 * row["median_runtime_ns"],
                              duration_per_call_ns=row["median_runtime_ns"], answer=16, status="ok")
            rows.append(row)
            raw.append(sample)
    plots.write_csv(directory / "summary.csv", rows, SUMMARY_FIELDS)
    plots.write_csv(directory / "timings.csv", raw, TIMING_FIELDS)


class SavedRecordTests(unittest.TestCase):
    def test_seed_aggregation_and_missing_points(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            fixture(path)
            rows, cases, _, audit = plots.read_run(path)
            self.assertEqual(audit["answers_checked"], 17)
            aggregates = plots.aggregate_rows(rows, cases)
            record = next(r for r in aggregates if r["experiment"] == "A" and r["method"] == "bitset")
            self.assertEqual((record["runtime_ms_median"], record["runtime_ms_min"], record["runtime_ms_max"]), (2, 1, 3))
            missing = next(r for r in aggregates if r["experiment"] == "B" and r["method"] == "dp" and r["parameter"] == 18)
            self.assertIsNone(missing["runtime_ms_median"])
            self.assertEqual(missing["timing_completed"], 0)
            self.assertEqual(missing["expected_configurations"], 1)
            common = plots.common_budget_table(rows)
            self.assertEqual({r["budget"] for r in common}, {17})
            self.assertEqual({r["method"] for r in common}, {"dp", "bitset", "approx"})
            allocation = plots.allocation_table(rows)
            budgets = {r["method"]: r["budget"] for r in allocation if r["experiment"] == "B"}
            self.assertEqual(budgets, {"dp": 17, "bitset": 18, "approx": 18})

    def test_tampered_batch_and_summary_median_are_rejected(self):
        for name, field in (("timings.csv", "duration_per_call_ns"), ("summary.csv", "median_runtime_ns")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                fixture(path)
                rows = plots.read_csv(path / name)
                rows[0][field] = str(float(rows[0][field]) * 2)
                plots.write_csv(path / name, rows, TIMING_FIELDS if name == "timings.csv" else SUMMARY_FIELDS)
                with self.assertRaisesRegex(ValueError, "normalization|median"):
                    plots.read_run(path)

    def test_unreachable_approximate_answer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            fixture(path)
            rows = plots.read_csv(path / "summary.csv")
            row = next(row for row in rows if row["method"] == "approx")
            row["answer"] = "14"  # Within budget but not a subset of [6, 10, 15].
            plots.write_csv(path / "summary.csv", rows, SUMMARY_FIELDS)
            raw = plots.read_csv(path / "timings.csv")
            for sample in raw:
                if plots.identity(sample) == plots.identity(row):
                    sample["answer"] = "14"
            plots.write_csv(path / "timings.csv", raw, TIMING_FIELDS)
            with self.assertRaisesRegex(ValueError, "achievable"):
                plots.read_run(path)

    def test_input_hash_failed_checks_and_resource_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            fixture(path)
            with self.assertRaisesRegex(ValueError, "verification-states"):
                plots.read_run(path, verification_states=10)
            rows = plots.read_csv(path / "summary.csv")
            rows[0]["verification_status"] = "failed"
            plots.write_csv(path / "summary.csv", rows, SUMMARY_FIELDS)
            with self.assertRaisesRegex(ValueError, "correctness"):
                plots.read_run(path)
            with (path / "instances.json").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                plots.read_run(path)

    def test_paired_speedups_are_aggregated_per_seed(self):
        rows = [dict(experiment="D", gcd=10, method="dp_gcd", preprocessing_speedup=value) for value in (1, 3, 20)]
        record = plots.speedup_table(rows)[0]
        self.assertEqual(record["speedup_median"], 3)
        self.assertEqual(record["speedup_min"], 1)
        self.assertEqual(record["speedup_max"], 20)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "install requirements.txt for rendering checks")
    def test_export_regenerates_without_touching_raw_data(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            fixture(source)
            before = {path.name: plots.digest(path) for path in source.iterdir()}
            output = Path(directory) / "figures"
            audit = plots.generate_outputs(source, output)
            self.assertEqual(audit["configurations"], 18)
            first = {path.name: plots.digest(path) for path in output.glob("*.png")}
            self.assertEqual(len(first), 3)
            for path in output.glob("*.png"):
                self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))
            plots.generate_outputs(source, output)
            self.assertEqual(first, {path.name: plots.digest(path) for path in output.glob("*.png")})
            self.assertEqual(before, {path.name: plots.digest(path) for path in source.iterdir()})
            self.assertEqual(len(plots.read_csv(output / "omitted_measurements.csv")), 1)


if __name__ == "__main__":
    unittest.main()
