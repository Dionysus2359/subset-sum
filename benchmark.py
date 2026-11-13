"""Reproducible subset-sum experiments with isolated timing and memory workers."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from math import gcd, isfinite
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc

from solvers import parse_epsilon, solve


METHODS = ("dp", "bitset", "approx", "dp_gcd", "bitset_gcd")
TIMING_FIELDS = [
    "case_id", "experiment", "family", "seed", "method", "epsilon", "repeat",
    "batch_count", "batch_duration_ns", "duration_per_call_ns", "answer", "status", "reason",
]
SUMMARY_FIELDS = [
    "case_id", "experiment", "family", "seed", "method", "epsilon", "answer",
    "median_runtime_ns", "peak_traced_bytes", "exact_optimum", "relative_error_pct",
    "original_n", "original_budget", "effective_n", "effective_budget", "solver_budget",
    "gcd", "timing_status", "memory_status", "verification_status", "verification_reason",
    "reference_source", "scaling_check", "timing_reason", "memory_reason", "preprocessing_speedup",
]


def configuration(method, epsilon=""):
    return {"method": method, "epsilon": epsilon}


def generate_instances(profile):
    """Generate the full grid before selecting quick points, keeping overlaps equal."""
    if profile not in ("quick", "full"):
        raise ValueError("profile must be quick or full")
    seeds = [319] if profile == "quick" else [319, 320, 321]
    cases = []
    common = [configuration("bitset"), configuration("dp"), configuration("approx", "0.10")]

    def draw(rng, n, low, high, budget, prefix=()):
        for _ in range(1000):
            values = list(prefix) + [rng.randint(low, high) for _ in range(n - len(prefix))]
            if sum(values) > budget:
                return values
        raise ValueError("could not generate a nontrivial instance")

    for seed in seeds:
        rng = random.Random(seed)
        for n in (25, 50, 100, 200):
            budget = 20_000
            values = draw(rng, n, max(1, budget // n), max(2, 3 * budget // n), budget)
            if profile == "full" or n in (25, 100):
                cases.append(dict(case_id=f"A-s{seed}-n{n}", experiment="A", family="count",
                                  seed=seed, values=values, budget=budget, configurations=common))
        for budget in (5_000, 20_000, 80_000, 320_000):
            n = 80
            values = draw(rng, n, max(1, budget // n), max(2, 3 * budget // n), budget)
            if profile == "full" or budget in (5_000, 20_000):
                cases.append(dict(case_id=f"B-s{seed}-b{budget}", experiment="B", family="budget",
                                  seed=seed, values=values, budget=budget, configurations=common))
        epsilons = ("0.10", "0.40") if profile == "quick" else ("0.05", "0.10", "0.20", "0.40")
        for family, low, high in (("broad", 1, 100_000), ("clustered", 49_000, 51_000)):
            values = draw(rng, 60, low, high, 1_000_000)
            configs = [configuration("bitset")] + [configuration("approx", e) for e in epsilons]
            cases.append(dict(case_id=f"C-s{seed}-{family}", experiment="C", family=family,
                              seed=seed, values=values, budget=1_000_000, configurations=configs))
        base = draw(rng, 60, 20, 100, 1500, prefix=(20, 21))
        for factor in ((1, 10) if profile == "quick" else (1, 10, 100)):
            cases.append(dict(
                case_id=f"D-s{seed}-g{factor}", experiment="D", family="gcd", seed=seed,
                values=[factor * value for value in base], budget=factor * 1500 + factor - 1,
                scale_factor=factor, base_id=f"D-s{seed}",
                configurations=[configuration(m) for m in ("bitset", "bitset_gcd", "dp", "dp_gcd")],
            ))
    cases.append(dict(
        case_id="literature-308", experiment="example", family="literature", seed=None,
        values=[104, 102, 201, 101], budget=308,
        configurations=[configuration("bitset"), configuration("approx", "0.40")],
    ))
    return {"schema_version": 1, "profile": profile, "seeds": seeds, "instances": cases}


def validate_manifest(manifest):
    """Validate saved inputs without running a nontrivial solver in the parent."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("instances must use schema_version 1")
    if manifest.get("profile") not in ("quick", "full"):
        raise ValueError("saved profile must be quick or full")
    cases = manifest.get("instances")
    if not isinstance(cases, list) or not cases:
        raise ValueError("instances must be a nonempty list")
    identifiers = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each instance must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in identifiers:
            raise ValueError("case_id must be a unique nonempty string")
        identifiers.add(case_id)
        if not isinstance(case.get("values"), list):
            raise ValueError(f"{case_id}: values must be a JSON list")
        # These trivial calls validate both original fields without allocating DP.
        solve(case["values"], 0)
        solve([], case.get("budget"))
        for key in ("experiment", "family"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise ValueError(f"{case_id}: {key} must be nonempty text")
        seed = case.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError(f"{case_id}: seed must be an integer or null")
        if case["experiment"] == "D":
            factor = case.get("scale_factor")
            if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
                raise ValueError(f"{case_id}: scale_factor must be positive")
            if not isinstance(case.get("base_id"), str) or not case["base_id"]:
                raise ValueError(f"{case_id}: base_id is required for experiment D")
        configs = case.get("configurations")
        if not isinstance(configs, list) or not configs:
            raise ValueError(f"{case_id}: configurations must be a nonempty list")
        seen = set()
        for config in configs:
            if not isinstance(config, dict) or config.get("method") not in METHODS:
                raise ValueError(f"{case_id}: unknown method")
            method = config["method"]
            epsilon = config.get("epsilon", "")
            if method == "approx":
                parse_epsilon(epsilon)
            elif epsilon != "":
                raise ValueError(f"{case_id}: exact methods must have empty epsilon")
            pair = (method, epsilon)
            if pair in seen:
                raise ValueError(f"{case_id}: duplicate configuration")
            seen.add(pair)
    return manifest


def instance_details(case, method):
    """Describe actual prepared/scaled work for runner eligibility and output."""
    budget = case["budget"]
    usable = [value for value in case["values"] if 0 < value <= budget]
    total = sum(usable)
    divisor = 0
    for value in usable:
        divisor = gcd(divisor, value)
    effective_budget = min(budget, total)
    trivial = not usable or budget == 0 or total <= budget
    solver_budget = 0 if trivial else effective_budget
    if method.endswith("_gcd") and divisor > 1:
        solver_budget //= divisor
    return dict(original_n=len(case["values"]), original_budget=budget,
                effective_n=len(usable), effective_budget=effective_budget,
                solver_budget=solver_budget, gcd=divisor, trivial=trivial)


def eligibility_reason(details, method, max_dp_work, max_bitset_states):
    if details["trivial"]:
        return ""
    if method in ("dp", "dp_gcd"):
        work = details["effective_n"] * details["solver_budget"]
        if work > max_dp_work:
            return f"DP work {work} exceeds cap {max_dp_work}"
    if method in ("bitset", "bitset_gcd"):
        states = details["solver_budget"] + 1
        if states > max_bitset_states:
            return f"bitset states {states} exceed cap {max_bitset_states}"
    return ""


def emit_event(event):
    print(json.dumps(event), flush=True)


def worker_main(mode):
    """Load first, then measure only public calls; serialize between measurements."""
    try:
        payload = json.load(sys.stdin)
        values, budget = payload["values"], payload["budget"]
        method = payload["method"]
        kwargs = dict(method=method.removesuffix("_gcd"), gcd_scale=method.endswith("_gcd"),
                      epsilon=payload.get("epsilon", "0.10"))
        # A tracing environment variable must not silently contaminate timings.
        tracemalloc.stop()
        if mode == "memory":
            tracemalloc.start()
            try:
                answer = solve(values, budget, **kwargs)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            emit_event(dict(type="memory", answer=answer, peak_traced_bytes=peak))
        else:
            expected = solve(values, budget, **kwargs)  # One untimed warm-up.

            def timed_batch(count):
                start = time.perf_counter_ns()
                for _ in range(count):
                    answer = solve(values, budget, **kwargs)
                duration = time.perf_counter_ns() - start
                if answer != expected:
                    raise ValueError("answer changed between measured calls")
                return duration, answer

            count = 1
            calibration_calls = 0
            while True:
                elapsed, _ = timed_batch(count)
                calibration_calls += count
                if elapsed >= payload["target_batch_ns"] or count >= payload["max_batch_count"]:
                    break
                count = min(count * 2, payload["max_batch_count"])
            emit_event(dict(type="calibration", batch_count=count,
                            calibration_calls=calibration_calls,
                            reached_target=elapsed >= payload["target_batch_ns"]))
            for repeat in range(1, payload["repeats"] + 1):
                duration, answer = timed_batch(count)
                emit_event(dict(type="sample", repeat=repeat, batch_count=count,
                                batch_duration_ns=duration, duration_per_call_ns=duration / count,
                                answer=answer))
        emit_event(dict(type="complete"))
        return 0
    except Exception as error:
        emit_event(dict(type="error", reason=f"{type(error).__name__}: {error}"))
        return 1


def run_worker(mode, payload, timeout):
    """Return parsed events even when a later repeat times out or fails."""
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", mode]
    try:
        process = subprocess.run(command, input=json.dumps(payload), capture_output=True,
                                 text=True, timeout=timeout)
        output = process.stdout
        status = "ok" if process.returncode == 0 else "error"
        reason = "" if status == "ok" else (process.stderr.strip()[-1000:] or f"worker exit {process.returncode}")
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        status, reason = "timeout", f"{mode} worker exceeded {timeout:g} seconds"
    except OSError as error:
        return dict(status="error", reason=f"{type(error).__name__}: {error}", events=[])

    events = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
            if not isinstance(event, dict) or "type" not in event:
                raise ValueError("invalid event")
            events.append(event)
        except (ValueError, TypeError):
            if status != "timeout":
                status, reason = "error", "invalid worker output"
    errors = [event["reason"] for event in events if event["type"] == "error"]
    if errors and status != "timeout":
        status, reason = "error", errors[-1]
    if status == "ok" and (not events or events[-1]["type"] != "complete"):
        status, reason = "error", "worker did not complete"
    return dict(status=status, reason=reason, events=events)


def measure_configuration(case, config, settings):
    details = instance_details(case, config["method"])
    base = {key: case.get(key) for key in ("case_id", "experiment", "family", "seed")}
    base.update(method=config["method"], epsilon=config.get("epsilon", ""))
    row = {key: "" for key in SUMMARY_FIELDS}
    row.update(base)
    row.update({key: value for key, value in details.items() if key != "trivial"})
    row.update(timing_status="skipped", memory_status="skipped", verification_status="not_checked")
    reason = eligibility_reason(details, config["method"], settings["max_dp_work"], settings["max_bitset_states"])
    if reason:
        row.update(timing_reason=reason, memory_reason=reason)
        return row, [dict(base, status="skipped", reason=reason)], []

    payload = dict(base, values=case["values"], budget=case["budget"],
                   repeats=settings["repeats"], target_batch_ns=settings["target_batch_ns"],
                   max_batch_count=settings["max_batch_count"])
    timing = run_worker("timing", payload, settings["timeout_seconds"])
    samples = [event for event in timing["events"] if event["type"] == "sample"]
    if timing["status"] == "ok" and len(samples) != settings["repeats"]:
        timing.update(status="error", reason="incomplete repeat records")
    row.update(timing_status=timing["status"], timing_reason=timing["reason"], memory_status="not_run")
    raw = []
    for sample in samples:
        raw.append(dict(base, **{key: value for key, value in sample.items() if key != "type"},
                        status="ok", reason=""))
    if timing["status"] != "ok":
        raw.append(dict(base, status=timing["status"], reason=timing["reason"]))
        row["memory_reason"] = "timing configuration did not complete"
    else:
        answers = {sample["answer"] for sample in samples}
        if len(answers) != 1:
            row.update(timing_status="error", timing_reason="answers differ across repeats",
                       memory_reason="timing configuration did not complete")
            raw.append(dict(base, status="error", reason=row["timing_reason"]))
        else:
            row["answer"] = samples[0]["answer"]
            row["median_runtime_ns"] = statistics.median(sample["duration_per_call_ns"] for sample in samples)
            memory = run_worker("memory", payload, settings["memory_timeout_seconds"])
            measurements = [event for event in memory["events"] if event["type"] == "memory"]
            if memory["status"] == "ok" and len(measurements) != 1:
                memory.update(status="error", reason="missing memory measurement")
            row.update(memory_status=memory["status"], memory_reason=memory["reason"])
            if memory["status"] == "ok":
                row["peak_traced_bytes"] = measurements[0]["peak_traced_bytes"]
                if measurements[0]["answer"] != row["answer"]:
                    row["verification_reason"] = "memory and timing answers differ"
    calibration = [dict(base, **{key: value for key, value in event.items() if key != "type"})
                   for event in timing["events"] if event["type"] == "calibration"]
    return row, raw, calibration


def verify_rows(rows, cases):
    """Refresh references and paired checks, including rows saved before a reference."""
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    references = {}
    for case_id, group in by_case.items():
        exact = [row for row in group if row["method"] != "approx" and row["answer"] != ""]
        if exact:
            reference = next((row for row in exact if row["method"] == "bitset"), exact[0])
            references[case_id] = (reference["answer"], reference["method"])
    base_optima = {}
    for case_id, (optimum, _) in references.items():
        case = cases[case_id]
        if case["experiment"] == "D" and case["scale_factor"] == 1:
            base_optima[case["base_id"]] = optimum

    for row in rows:
        row.update(exact_optimum="", relative_error_pct="", reference_source="", preprocessing_speedup="",
                   scaling_check="not_applicable")
        if row["answer"] == "":
            row["verification_status"] = "not_checked"
            continue
        memory_mismatch = "memory and timing answers differ"
        reasons = [memory_mismatch] if memory_mismatch in row["verification_reason"].split("; ") else []
        answer = row["answer"]
        if not isinstance(answer, int) or isinstance(answer, bool) or not 0 <= answer <= row["original_budget"]:
            reasons.append("answer is outside the feasible numeric range")
        reference = references.get(row["case_id"])
        if reference is not None:
            optimum, source = reference
            row.update(exact_optimum=optimum, reference_source=source,
                       relative_error_pct=0.0 if optimum == 0 else 100 * (optimum - answer) / optimum)
            if row["method"] == "approx":
                ratio = parse_epsilon(row["epsilon"])
                if answer > optimum or ratio.denominator * answer < (ratio.denominator - ratio.numerator) * optimum:
                    reasons.append("approximation guarantee violated")
            elif answer != optimum:
                reasons.append("exact methods disagree")
        case = cases[row["case_id"]]
        if case["experiment"] == "D":
            base_optimum = base_optima.get(case["base_id"])
            row["scaling_check"] = "no_base_reference" if base_optimum is None else "ok"
            if base_optimum is not None and answer != case["scale_factor"] * base_optimum:
                row["scaling_check"] = "failed"
                reasons.append("GCD family differs from factor times base optimum")
        row["verification_status"] = "failed" if reasons else ("ok" if reference is not None else "no_reference")
        row["verification_reason"] = "; ".join(reasons)

    for group in by_case.values():
        for row in group:
            if row["method"].endswith("_gcd") and row["verification_status"] == "ok":
                original = next((other for other in group if other["method"] == row["method"].removesuffix("_gcd")), None)
                if (original and original["verification_status"] == "ok" and
                        row["timing_status"] == original["timing_status"] == "ok" and
                        row["median_runtime_ns"] > 0 and original["median_runtime_ns"] > 0):
                    row["preprocessing_speedup"] = original["median_runtime_ns"] / row["median_runtime_ns"]


def write_summary(path, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
    temporary.replace(path)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def environment_metadata(manifest, settings, argv):
    root = Path(__file__).resolve().parent
    files = sorted([*root.glob("*.py"), *root.glob("tests/*.py")])
    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    clock = time.get_clock_info("perf_counter")
    return dict(
        schema_version=1, started_at=datetime.now(timezone.utc).isoformat(), run_status="running",
        python_implementation=platform.python_implementation(), python_version=platform.python_version(),
        python_executable=sys.executable, operating_system=platform.platform(), architecture=platform.machine(),
        cpu=cpu, bits_per_digit=sys.int_info.bits_per_digit,
        clock=dict(implementation=clock.implementation, resolution_seconds=clock.resolution, monotonic=clock.monotonic),
        command=[sys.executable, *argv], profile=manifest["profile"],
        seeds=sorted({case["seed"] for case in manifest["instances"] if case.get("seed") is not None}),
        settings=settings, source_hashes={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        memory_metric="peak traced Python allocation during solve, bytes; separate un-warmed process",
        timing_metric="median ns per public solve; batch loop overhead included; tracing disabled",
        calibration=[],
    )


def run_benchmarks(manifest, out, settings, argv):
    validate_manifest(manifest)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "instances.json", manifest)
    metadata = environment_metadata(manifest, settings, argv)
    metadata["instances_sha256"] = hashlib.sha256((out / "instances.json").read_bytes()).hexdigest()
    write_json(out / "environment.json", metadata)
    rows = []
    write_summary(out / "summary.csv", rows)
    cases = {case["case_id"]: case for case in manifest["instances"]}
    total = sum(len(case["configurations"]) for case in cases.values())
    try:
        with (out / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=TIMING_FIELDS)
            writer.writeheader()
            stream.flush()
            for case in cases.values():
                for config in case["configurations"]:
                    row, raw, calibration = measure_configuration(case, config, settings)
                    writer.writerows(raw)
                    stream.flush()
                    rows.append(row)
                    verify_rows(rows, cases)
                    write_summary(out / "summary.csv", rows)
                    metadata["calibration"].extend(calibration)
                    metadata["completed_configurations"] = len(rows)
                    write_json(out / "environment.json", metadata)
                    print(f"[{len(rows)}/{total}] {row['case_id']} {row['method']} {row['epsilon']}: "
                          f"timing={row['timing_status']} memory={row['memory_status']} "
                          f"check={row['verification_status']}", flush=True)
    except BaseException:
        metadata.update(run_status="interrupted", finished_at=datetime.now(timezone.utc).isoformat())
        write_json(out / "environment.json", metadata)
        raise
    failures = any(row["timing_status"] in ("error", "timeout") or
                   row["memory_status"] in ("error", "timeout") or row["verification_status"] == "failed"
                   for row in rows)
    metadata.update(
        run_status="completed_with_failures" if failures else "completed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        timing_status_counts=dict(Counter(row["timing_status"] for row in rows)),
        memory_status_counts=dict(Counter(row["memory_status"] for row in rows)),
        verification_status_counts=dict(Counter(row["verification_status"] for row in rows)),
    )
    write_json(out / "environment.json", metadata)
    return 1 if failures else 0


def nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def positive_int(text):
    value = nonnegative_int(text)
    if value == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def positive_float(text):
    value = float(text)
    if not isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), help="default quick; inherited on replay")
    parser.add_argument("--out", type=Path, help="new output directory; existing directories are refused")
    parser.add_argument("--instances", type=Path, help="replay a saved instances.json with its configurations")
    parser.add_argument("--repeats", type=positive_int, help="default 2 for quick, 3 for full")
    parser.add_argument("--max-dp-work", type=nonnegative_int, default=20_000_000)
    parser.add_argument("--max-bitset-states", type=nonnegative_int, default=10_000_000)
    parser.add_argument("--timeout", type=positive_float, default=30.0, help="seconds per timing worker")
    parser.add_argument("--memory-timeout", type=positive_float, default=30.0, help="seconds per memory worker")
    parser.add_argument("--batch-target-ms", type=positive_float, default=20.0)
    parser.add_argument("--max-batch-count", type=positive_int, default=65_536)
    parser.add_argument("--worker", choices=("timing", "memory"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker_main(args.worker)
    if args.out is None:
        parser.error("--out is required")
    try:
        if args.instances:
            manifest = validate_manifest(json.loads(args.instances.read_text(encoding="utf-8")))
            if args.profile and args.profile != manifest["profile"]:
                parser.error("--profile conflicts with the saved instance profile")
        else:
            manifest = generate_instances(args.profile or "quick")
        settings = dict(
            repeats=args.repeats or (2 if manifest["profile"] == "quick" else 3),
            max_dp_work=args.max_dp_work, max_bitset_states=args.max_bitset_states,
            timeout_seconds=args.timeout, memory_timeout_seconds=args.memory_timeout,
            target_batch_ns=max(1, int(args.batch_target_ms * 1_000_000)),
            max_batch_count=args.max_batch_count,
        )
        command = sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]
        return run_benchmarks(manifest, args.out, settings, command)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("Benchmark interrupted; completed records were retained.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
