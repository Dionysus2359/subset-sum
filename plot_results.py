"""Validate saved benchmark records and regenerate standalone figures and tables."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import textwrap


LABELS = {"dp": "Standard DP", "bitset": "Bitset DP", "approx": "Trimming approximation",
          "dp_gcd": "DP + GCD", "bitset_gcd": "Bitset + GCD"}
COLORS = {"dp": "#ad493b", "bitset": "#007d8a", "approx": "#b47800"}
AGGREGATE_FIELDS = [
    "experiment", "family", "method", "epsilon", "parameter", "expected_configurations",
    "timing_completed", "memory_completed", "runtime_ms_median", "runtime_ms_min", "runtime_ms_max",
    "peak_kib_median", "peak_kib_min", "peak_kib_max", "error_pct_median", "error_pct_max",
]
ALLOCATION_FIELDS = ["experiment", "family", "method", "epsilon", "n", "budget", "seeds",
                     "runtime_ms_median", "peak_kib_median", "peak_kib_min", "peak_kib_max", "error_pct_max"]
SPEEDUP_FIELDS = ["gcd", "method", "paired_seeds", "speedup_median", "speedup_min", "speedup_max"]


def identity(row):
    return row["case_id"], row["method"], row.get("epsilon", "")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(directory, verification_states=10_000_000):
    """Audit raw durations and independently check saved subset answers.

    Verification is outside the benchmark: a fresh reachability mask checks both
    exact optima and membership of approximate answers on each saved instance.
    No solver is timed again and no missing measurement is filled in.
    """
    manifest = json.loads((directory / "instances.json").read_text())
    environment = json.loads((directory / "environment.json").read_text())
    if environment.get("run_status") not in ("completed", "completed_with_failures"):
        raise ValueError("only completed benchmark runs can be plotted")
    if digest(directory / "instances.json") != environment["instances_sha256"]:
        raise ValueError("saved instances hash does not match environment metadata")
    cases = {case["case_id"]: case for case in manifest["instances"]}
    if len(cases) != len(manifest["instances"]):
        raise ValueError("duplicate saved case IDs")
    expected = {(case["case_id"], config["method"], config.get("epsilon", ""))
                for case in cases.values() for config in case["configurations"]}
    rows = read_csv(directory / "summary.csv")
    if len(rows) != len(expected) or {identity(row) for row in rows} != expected:
        raise ValueError("summary does not match the saved configurations")
    samples = defaultdict(list)
    for sample in read_csv(directory / "timings.csv"):
        if identity(sample) not in expected:
            raise ValueError("raw timing has an unknown configuration")
        if sample["status"] == "ok":
            duration = int(sample["batch_duration_ns"])
            count = int(sample["batch_count"])
            normalized = float(sample["duration_per_call_ns"])
            if duration <= 0 or count <= 0 or not math.isclose(duration / count, normalized, rel_tol=1e-12):
                raise ValueError("raw batch duration normalization is inconsistent")
            samples[identity(sample)].append(sample)
        elif sample["batch_duration_ns"] or sample["duration_per_call_ns"]:
            raise ValueError("an unsuccessful timing row contains a duration")

    reachable = {}
    for case_id, case in cases.items():
        budget = case["budget"]
        if budget + 1 > verification_states:
            raise ValueError("independent verification exceeds --verification-states")
        mask = (1 << (budget + 1)) - 1
        bits = 1
        for value in case["values"]:
            if 0 < value <= budget:
                bits = (bits | (bits << value)) & mask
        reachable[case_id] = bits

    checked_answers = checked_approx = checked_raw = 0
    for row in rows:
        case = cases[row["case_id"]]
        if row["verification_status"] == "failed":
            raise ValueError(f"saved correctness check failed: {identity(row)}")
        for key in ("original_n", "original_budget", "effective_n", "effective_budget", "solver_budget", "gcd"):
            row[key] = int(row[key])
        if row["original_n"] != len(case["values"]) or row["original_budget"] != case["budget"]:
            raise ValueError("summary size/budget differs from saved input")
        for key in ("answer", "peak_traced_bytes", "exact_optimum"):
            row[key] = None if row[key] == "" else int(row[key])
        for key in ("median_runtime_ns", "relative_error_pct", "preprocessing_speedup"):
            row[key] = None if row[key] == "" else float(row[key])
            if row[key] is not None and not math.isfinite(row[key]):
                raise ValueError("nonfinite numeric measurement")
        if row["memory_status"] == "ok":
            if row["peak_traced_bytes"] is None or row["peak_traced_bytes"] < 0:
                raise ValueError("successful memory row lacks a valid allocation")
        elif row["peak_traced_bytes"] is not None:
            raise ValueError("unsuccessful memory row contains an allocation")
        if row["timing_status"] != "ok":
            if row["median_runtime_ns"] is not None or row["answer"] is not None:
                raise ValueError("unsuccessful timing row contains a summary result")
            continue
        raw = samples[identity(row)]
        if len(raw) != environment["settings"]["repeats"]:
            raise ValueError("successful configuration has an incomplete repeat count")
        if {int(sample["repeat"]) for sample in raw} != set(range(1, len(raw) + 1)):
            raise ValueError("duplicate or missing repeat indices")
        if len({sample["batch_count"] for sample in raw}) != 1:
            raise ValueError("recorded repeats use different batch counts")
        median = statistics.median(float(sample["duration_per_call_ns"]) for sample in raw)
        if row["median_runtime_ns"] is None or not math.isclose(median, row["median_runtime_ns"], rel_tol=1e-12):
            raise ValueError("summary median differs from raw repeats")
        if any(int(sample["answer"]) != row["answer"] for sample in raw):
            raise ValueError("raw and summary answers differ")
        checked_raw += len(raw)
        bits = reachable[row["case_id"]]
        optimum = bits.bit_length() - 1
        answer = row["answer"]
        if answer is None or answer < 0 or answer > case["budget"] or not (bits >> answer) & 1:
            raise ValueError("returned answer is not an achievable original subset sum")
        if row["exact_optimum"] is not None:
            if row["exact_optimum"] != optimum:
                raise ValueError("saved reference differs from independently checked optimum")
            error = 0 if optimum == 0 else 100 * (optimum - answer) / optimum
            if row["relative_error_pct"] is None or not math.isclose(error, row["relative_error_pct"], abs_tol=1e-12):
                raise ValueError("saved error is not relative to the exact optimum")
        elif row["relative_error_pct"] is not None:
            raise ValueError("relative error exists without a saved exact reference")
        if row["method"] == "approx":
            ratio = Fraction(row["epsilon"])
            if not 0 < ratio < 1 or ratio.denominator * answer < (ratio.denominator - ratio.numerator) * optimum:
                raise ValueError("approximation guarantee violated")
            checked_approx += 1
        elif answer != optimum:
            raise ValueError("exact answer disagrees with independent verification")
        checked_answers += 1

    for row in rows:
        if row["preprocessing_speedup"] is not None:
            original = next((other for other in rows if other["case_id"] == row["case_id"] and
                             other["method"] == row["method"].removesuffix("_gcd")), None)
            if (not row["method"].endswith("_gcd") or not original or
                    row["timing_status"] != "ok" or original["timing_status"] != "ok" or
                    not math.isclose(row["preprocessing_speedup"], original["median_runtime_ns"] / row["median_runtime_ns"], rel_tol=1e-12)):
                raise ValueError("paired preprocessing speedup is inconsistent")
        case = cases[row["case_id"]]
        if case["experiment"] == "D" and row["answer"] is not None:
            bases = [other for other in cases.values() if other["experiment"] == "D" and
                     other["base_id"] == case["base_id"] and other["scale_factor"] == 1]
            if bases and row["answer"] != case["scale_factor"] * (reachable[bases[0]["case_id"]].bit_length() - 1):
                raise ValueError("GCD family answer differs from the scaled base optimum")
    audit = dict(
        cases=len(cases), configurations=len(rows), answers_checked=checked_answers,
        approximation_feasibility_and_bound_checks=checked_approx, timed_repeats_checked=checked_raw,
        timing_status_counts=dict(Counter(row["timing_status"] for row in rows)),
        memory_status_counts=dict(Counter(row["memory_status"] for row in rows)),
        independent_verification="fresh reachability mask on original saved inputs; no benchmark reruns",
        verification_states_limit=verification_states,
    )
    return rows, cases, environment, audit


def parameter(row, cases):
    if row["experiment"] == "A":
        return row["original_n"]
    if row["experiment"] == "C":
        return float(row["epsilon"]) if row["method"] == "approx" else 0
    if row["experiment"] == "D":
        return cases[row["case_id"]]["scale_factor"]
    return row["original_budget"]


def summarize(values, operation=statistics.median):
    return operation(values) if values else None


def aggregate_rows(rows, cases):
    """Use medians/ranges across seed-level medians, never pool raw repeats."""
    groups = defaultdict(list)
    for row in rows:
        key = (row["experiment"], row["family"], row["method"], row["epsilon"], parameter(row, cases))
        groups[key].append(row)
    aggregates = []
    for key, group in sorted(groups.items()):
        record = dict(zip(AGGREGATE_FIELDS[:5], key))
        runtime = [row["median_runtime_ns"] / 1_000_000 for row in group if row["timing_status"] == "ok"]
        memory = [row["peak_traced_bytes"] / 1024 for row in group if row["memory_status"] == "ok"]
        error = [row["relative_error_pct"] for row in group if row["relative_error_pct"] is not None]
        record.update(expected_configurations=len(group), timing_completed=len(runtime), memory_completed=len(memory))
        for name, values in (("runtime_ms", runtime), ("peak_kib", memory)):
            record.update({f"{name}_median": summarize(values), f"{name}_min": summarize(values, min),
                           f"{name}_max": summarize(values, max)})
        record.update(error_pct_median=summarize(error), error_pct_max=summarize(error, max))
        aggregates.append(record)
    return aggregates


def allocation_table(rows):
    """Per family/method, use epsilon .10 and largest completed budget, then n."""
    groups = defaultdict(list)
    for row in rows:
        if row["experiment"] not in ("A", "B", "C", "D") or row["memory_status"] != "ok":
            continue
        if row["method"] == "approx" and Fraction(row["epsilon"]) != Fraction(1, 10):
            continue
        groups[row["experiment"], row["family"], row["method"]].append(row)
    output = []
    for (experiment, family, method), group in sorted(groups.items()):
        chosen = max((row["original_budget"], row["original_n"]) for row in group)
        selected = [row for row in group if (row["original_budget"], row["original_n"]) == chosen]
        memory = [row["peak_traced_bytes"] / 1024 for row in selected]
        errors = [row["relative_error_pct"] for row in selected if row["relative_error_pct"] is not None]
        output.append(dict(experiment=experiment, family=family, method=method, epsilon=selected[0]["epsilon"],
                           n=chosen[1], budget=chosen[0], seeds=len(selected),
                           runtime_ms_median=statistics.median(row["median_runtime_ns"] / 1e6 for row in selected),
                           peak_kib_median=statistics.median(memory), peak_kib_min=min(memory), peak_kib_max=max(memory),
                           error_pct_max=summarize(errors, max)))
    return output


def common_budget_table(rows):
    """Small report candidate: largest B budget completed by all methods/seeds."""
    group = [row for row in rows if row["experiment"] == "B" and row["method"] in ("dp", "bitset", "approx")]
    seeds = {row["seed"] for row in group}
    eligible = []
    for budget in sorted({row["original_budget"] for row in group}):
        if all({row["seed"] for row in group if row["original_budget"] == budget and row["method"] == method and
                row["timing_status"] == row["memory_status"] == "ok"} == seeds for method in ("dp", "bitset", "approx")):
            eligible.append(budget)
    if not eligible:
        return []
    return allocation_table([row for row in group if row["original_budget"] == max(eligible)])


def speedup_table(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["experiment"] == "D" and row["preprocessing_speedup"] is not None:
            groups[row["gcd"], row["method"].removesuffix("_gcd")].append(row["preprocessing_speedup"])
    return [dict(gcd=factor, method=method, paired_seeds=len(values), speedup_median=statistics.median(values),
                 speedup_min=min(values), speedup_max=max(values))
            for (factor, method), values in sorted(groups.items())]


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, fields):
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        def fmt(value):
            if value is None:
                return "Missing"
            return f"{value:.4g}" if isinstance(value, float) else str(value)
        lines.append("| " + " | ".join(fmt(row[field]) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def draw_curve(ax, records, color, label, metric="runtime_ms", linestyle="-"):
    records = sorted(records, key=lambda record: record["parameter"])
    if not records:
        return
    xs = [record["parameter"] for record in records]
    ys = [record[f"{metric}_median"] if record[f"{metric}_median"] is not None else math.nan for record in records]
    ax.plot(xs, ys, marker="o", markersize=4, color=color, label=label, linewidth=1.8, linestyle=linestyle)
    present = [record for record in records if record[f"{metric}_median"] is not None]
    if present:
        ax.errorbar([r["parameter"] for r in present], [r[f"{metric}_median"] for r in present],
                    yerr=[[r[f"{metric}_median"] - r[f"{metric}_min"] for r in present],
                          [r[f"{metric}_max"] - r[f"{metric}_median"] for r in present]],
                    fmt="none", ecolor=color, capsize=3, alpha=0.7, linewidth=1)


def save_figure(fig, path, title, subtitle, caption, plt):
    fig.suptitle(title, x=0.09, y=0.98, ha="left", fontsize=16, fontweight="bold", color="#172b3a")
    fig.text(0.09, 0.918, subtitle, fontsize=10, color="#425564")
    fig.text(0.09, 0.025, textwrap.fill(caption, 128), fontsize=9, color="#425564", va="bottom", linespacing=1.45)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(rows, cases, aggregates, environment, out):
    # Imports are lazy so record verification/tests need no plotting dependency.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#243746", "axes.titleweight": "bold",
                         "axes.edgecolor": "#b8c3ca", "xtick.color": "#425564", "ytick.color": "#425564",
                         "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.2, "grid.color": "#8798a4"})
    settings = f"{environment['profile'].capitalize()} profile | seeds {', '.join(map(str, environment['seeds']))} | "
    settings += f"{environment['settings']['repeats']} timed repeats/configuration"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.77, bottom=0.23, wspace=0.28)
    for ax, experiment, title, xlabel in zip(axes, ("A", "B"),
        ("A  Item count; budget = 20,000", "B  Budget; item count = 80"),
        ("Number of items", "Budget")):
        selected = [record for record in aggregates if record["experiment"] == experiment]
        for method in ("dp", "bitset", "approx"):
            draw_curve(ax, [r for r in selected if r["method"] == method], COLORS[method],
                       LABELS[method] + (" (epsilon = 0.10)" if method == "approx" else ""))
        ticks = sorted({r["parameter"] for r in selected})
        if experiment == "B":
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
        ax.set_xticks(ticks)
        ax.set_yscale("log")
        ax.set(title=title, xlabel=xlabel, ylabel="Median runtime per solve (ms)")
        skipped = [row for row in rows if row["experiment"] == experiment and row["timing_status"] != "ok"]
        if skipped:
            note = f"{len(skipped)} unavailable timing points\nRecorded as missing"
            if all(row["method"] == "dp" and row["timing_status"] == "skipped" for row in skipped):
                budgets = ", ".join(f"{budget / 1000:g}k" for budget in sorted({row["original_budget"] for row in skipped}))
                note = f"Standard DP skipped at budget {budgets}\n{len(skipped)} seeds; work cap exceeded"
            ax.text(0.98, 0.04, note,
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=COLORS["dp"],
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.89), ncol=3, fontsize=9)
    save_figure(fig, out / "runtime_scaling.png", "Exact and approximate subset sum: scaling", settings,
                "Lines show medians across seeds; whiskers span the minimum and maximum seed-level medians. "
                "Runtime axes are logarithmic. Missing runs have no plotted value. "
                f"DP cap: n x solver budget <= {environment['settings']['max_dp_work']:,}; "
                f"bitset cap: {environment['settings']['max_bitset_states']:,} states.", plt)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.14, wspace=0.29, hspace=0.56)
    for index, family in enumerate(("broad", "clustered")):
        runtime_ax, error_ax = axes[index]
        records = [r for r in aggregates if r["experiment"] == "C" and r["family"] == family and r["method"] == "approx"]
        epsilons = sorted({r["parameter"] for r in records}) or [0.05, 0.10, 0.20, 0.40]
        draw_curve(runtime_ax, records, COLORS["approx"], "Trimming approximation")
        reference = [r for r in aggregates if r["experiment"] == "C" and r["family"] == family and r["method"] == "bitset"]
        if reference and reference[0]["runtime_ms_median"] is not None:
            ref = reference[0]
            runtime_ax.axhline(ref["runtime_ms_median"], color=COLORS["bitset"], linestyle="--", label="Exact bitset reference")
            runtime_ax.axhspan(ref["runtime_ms_min"], ref["runtime_ms_max"], color=COLORS["bitset"], alpha=0.10)
        runtime_ax.set_yscale("log")
        runtime_ax.set(title=f"{family.capitalize()} values: runtime", ylabel="Runtime per solve (ms)")
        if records or reference:
            runtime_ax.legend(fontsize=8.5, loc="best")
        error_records = [dict(r, error_pct_min=0) for r in records]
        for record in error_records:
            actual = [row["relative_error_pct"] for row in rows if row["experiment"] == "C" and row["family"] == family and
                      row["epsilon"] == record["epsilon"] and row["relative_error_pct"] is not None]
            record["error_pct_min"] = min(actual) if actual else None
        draw_curve(error_ax, error_records, COLORS["approx"], "Observed error", metric="error_pct")
        observed_max = max((r["error_pct_max"] for r in records if r["error_pct_max"] is not None), default=0)
        error_ax.set_ylim(0, max(0.005, observed_max * 1.25))
        error_ax.set(title=f"{family.capitalize()} values: observed error", ylabel="Error relative to OPT (%)")
        inset = error_ax.inset_axes([0.10, 0.52, 0.42, 0.40])
        inset.plot(epsilons, [100 * e for e in epsilons], color="#6c7580", linestyle="--", linewidth=1.4)
        inset.set_ylim(0, max(epsilons) * 110)
        inset.set_title("Guarantee ceiling (%)", fontsize=7.5, pad=3)
        inset.tick_params(labelsize=7)
        inset.set_xticks([min(epsilons), max(epsilons)])
        inset.set_facecolor("#f3f5f7")
        for ax in (runtime_ax, error_ax):
            ax.set_xscale("log", base=2)
            ax.set_xticks(epsilons, [f"{e:g}" for e in epsilons])
            ax.set_xlabel("Epsilon")
            ax.set_xlim(min(epsilons) * 0.84, max(epsilons) * 1.18)
    save_figure(fig, out / "epsilon_tradeoff.png", "Approximation: accuracy and computation", settings + " | n = 60; budget = 1,000,000",
                "Lines are medians and whiskers are seed ranges. Exact bitset references are reused across epsilon settings. "
                "Error panels zoom in on observed error; each inset uses its own scale for the theoretical 100 x epsilon ceiling.", plt)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.23, wspace=0.28)
    for ax, method in zip(axes, ("dp", "bitset")):
        for scaled in (False, True):
            label = method + ("_gcd" if scaled else "")
            records = [r for r in aggregates if r["experiment"] == "D" and r["method"] == label]
            draw_curve(ax, records, COLORS["bitset"] if scaled else COLORS["dp"],
                       "With GCD scaling" if scaled else "Without scaling", linestyle="--" if scaled else "-")
        factors = sorted({r["parameter"] for r in aggregates if r["experiment"] == "D"})
        ax.set_xscale("log", base=10)
        ax.set_xticks(factors, [str(g) for g in factors])
        ax.set_yscale("log")
        ax.set(title=LABELS[method], xlabel="Common factor g", ylabel="Runtime per solve (ms)")
        if factors:
            ax.legend(fontsize=9)
    save_figure(fig, out / "gcd_scaling.png", "GCD preprocessing: benefit and overhead", settings,
                "Each seed shares one 60-item base with GCD 1 and budget 1,500. Inputs are multiplied by g; "
                "budget = 1,500 x g + (g - 1). Medians and ranges include preprocessing. The g = 1 control measures overhead.", plt)
    return matplotlib.__version__


def generate_outputs(directory, out, verification_states=10_000_000):
    if directory.resolve() == out.resolve():
        raise ValueError("figure output must be separate from raw input")
    rows, cases, environment, audit = read_run(directory, verification_states)
    aggregates = aggregate_rows(rows, cases)
    allocation = allocation_table(rows)
    common = common_budget_table(rows)
    speedups = speedup_table(rows)
    out.mkdir(parents=True, exist_ok=True)
    version = make_figures(rows, cases, aggregates, environment, out)
    write_csv(out / "aggregates.csv", aggregates, AGGREGATE_FIELDS)
    write_csv(out / "allocation_table.csv", allocation, ALLOCATION_FIELDS)
    write_csv(out / "report_table.csv", common, ALLOCATION_FIELDS)
    write_csv(out / "gcd_speedups.csv", speedups, SPEEDUP_FIELDS)
    omitted = [dict(case_id=r["case_id"], method=r["method"], epsilon=r["epsilon"], timing_status=r["timing_status"],
                    memory_status=r["memory_status"], timing_reason=r["timing_reason"], memory_reason=r["memory_reason"])
               for r in rows if r["timing_status"] != "ok" or r["memory_status"] != "ok"]
    write_csv(out / "omitted_measurements.csv", omitted, ["case_id", "method", "epsilon", "timing_status", "memory_status", "timing_reason", "memory_reason"])
    (out / "allocation_table.md").write_text(
        "# Representative traced allocation\n\nSelection: per experiment/family/method, use epsilon 0.10 for approximation; "
        "choose largest completed budget, then largest item count. Aggregate the selected seeds. Budgets can differ by method "
        "when larger runs were skipped. Peak values show median [minimum, maximum] across seeds; KiB = bytes / 1024.\n\n" +
        markdown_table([{"Experiment / family": f"{r['experiment']} / {r['family']}", "Method": LABELS[r["method"]],
                         "n": r["n"], "Budget": f"{r['budget']:,}", "Seeds": r["seeds"],
                         "Runtime (ms)": r["runtime_ms_median"],
                         "Peak traced (KiB)": f"{r['peak_kib_median']:.2f} [{r['peak_kib_min']:.2f}, {r['peak_kib_max']:.2f}]",
                         "Max error (%)": r["error_pct_max"]} for r in allocation],
                       ["Experiment / family", "Method", "n", "Budget", "Seeds", "Runtime (ms)", "Peak traced (KiB)", "Max error (%)"]), encoding="utf-8")
    chosen_setting = (f" Selected n = {common[0]['n']}, budget = {common[0]['budget']:,}, seeds = {common[0]['seeds']}."
                      if common else " No common completed budget is available.")
    (out / "report_table.md").write_text(
        "# Compact report table candidate\n\nExperiment B: largest budget with timing and memory completed for all "
        "three methods on all saved seeds." + chosen_setting + " Values are medians across seeds; "
        "error is the maximum observed. Approximation uses epsilon 0.10.\n\n" +
        markdown_table([{"Method": LABELS[r["method"]], "Runtime (ms)": r["runtime_ms_median"],
                         "Peak traced (KiB)": r["peak_kib_median"], "Max error (%)": r["error_pct_max"]} for r in common],
                       ["Method", "Runtime (ms)", "Peak traced (KiB)", "Max error (%)"]), encoding="utf-8")
    (out / "gcd_speedups.md").write_text(
        "# Paired preprocessing speedups\n\nCompute unscaled/scaled runtime per seed before aggregating. "
        "Values below 1 mean overhead. Ranges are seed minima/maxima.\n\n" +
        markdown_table([{"GCD": r["gcd"], "Method": LABELS[r["method"]], "Seeds": r["paired_seeds"],
                         "Speedup median [min, max]": f"{r['speedup_median']:.3f} [{r['speedup_min']:.3f}, {r['speedup_max']:.3f}]"}
                        for r in speedups], ["GCD", "Method", "Seeds", "Speedup median [min, max]"]), encoding="utf-8")
    captions = (
        "# Phase 4 figures\n\nGenerated from saved records; no benchmarks are rerun by this command.\n\n"
        "- [Runtime scaling](runtime_scaling.png): A/B panels; log runtime axes, seed medians/ranges, DP omissions annotated.\n"
        "- [Epsilon tradeoff](epsilon_tradeoff.png): runtime and error for both C families; theory insets have separate scales.\n"
        "- [GCD scaling](gcd_scaling.png): paired exact methods including the GCD-one control.\n"
        "- [Allocation table](allocation_table.md): representative largest completed instances.\n"
        "- [Compact table](report_table.md): largest common completed B budget.\n"
        "- [GCD speedups](gcd_speedups.md): ratios computed within each paired seed.\n\n"
        "PNG files are 220 dpi. Lines show medians across seed-level medians; whiskers span seeds, not confidence intervals. "
        "The source run and export hashes are recorded in plot_environment.json; validation.json records the independent audit.\n"
    )
    (out / "README.md").write_text(captions, encoding="utf-8")
    (out / "validation.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    source_files = ("instances.json", "summary.csv", "timings.csv", "environment.json")
    metadata = dict(
        generated_at=datetime.now(timezone.utc).isoformat(), input_directory=str(directory.resolve()),
        python_version=platform.python_version(), matplotlib_version=version,
        numpy_version=importlib.metadata.version("numpy"),
        plotting_source_sha256=digest(Path(__file__)), input_hashes={name: digest(directory / name) for name in source_files},
        figure_dpi=220, aggregation="median/min/max of seed-level per-configuration medians; no pooling of raw repeats",
        export_hashes={path.name: digest(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != "plot_environment.json"},
    )
    (out / "plot_environment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verification-states", type=int, default=10_000_000)
    args = parser.parse_args(argv)
    if args.verification_states < 1:
        parser.error("--verification-states must be positive")
    try:
        audit = generate_outputs(args.input, args.out, args.verification_states)
    except ImportError as error:
        parser.error(f"plot dependency unavailable: {error}; install requirements.txt")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
