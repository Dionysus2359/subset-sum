# Subset Sum Under a Budget

CSD319 Design and Analysis of Algorithms: **Comparing Exact and Approximation
Algorithms**. All five phases are complete: three algorithms, exact GCD
preprocessing, an instance CLI, independent checks, a benchmark runner, full
experiment results, plots/tables and the final five-page report.

See [the roadmap](PHASES.md), [plan review](docs/plan_review.md), and
[phase-2/3 verification record](docs/phase2_3_verification.md), and
[full results and analysis](docs/phase4_results.md).

## Setup and solve

Use Python 3.11 or later. Algorithms, tests and benchmarks need only the standard
library; no package installation is needed. Run from this directory:

```bash
python3 -m unittest discover -s tests -v
python3 cli.py --values 6 10 15 --budget 17 --method dp
python3 cli.py --input examples/instance.json --method bitset
python3 cli.py --values 104 102 201 101 --budget 308 --method approx --epsilon 0.40
python3 cli.py --values 60 100 150 --budget 179 --method bitset --gcd-scale
```

The first two solver examples return 16; approximation returns 302 versus exact
optimum 307; scaling returns 160 in original units. The CLI prints the method
label, budget, best sum, and epsilon for approximation.

JSON files contain `{"values": [6, 10, 15], "budget": 17}`. Use `--budget` only
with inline values; files supply their own budget. Empty input works with
`python3 cli.py --values --budget 10`. Invalid inputs produce an error on stderr
and exit status 2.

## Input contract and algorithms

`solve(values, budget, method="dp", epsilon="0.10", gcd_scale=False)` returns the
largest subset sum within the budget. Each position may be used once; duplicates
remain separate items. The empty subset is valid. Values must be an ordered
sequence of nonnegative integers and the budget a nonnegative integer. Zeros and
empty sequences are accepted. Negative values, booleans, non-integers, sets,
generators, and text/byte sequences are rejected. Inputs are never sorted or mutated.

Every original item is validated before trivial returns. Preparation removes
zero/oversized items and returns the usable total if it fits. Use `solve` for
user input; the three algorithm cores have prepared-input preconditions.

| Method | Core behavior | Core time | Core state |
| --- | --- | --- | --- |
| `dp` | Descending bytearray reachability | O(nB) | O(B) bytes |
| `bitset` | Integer shift/OR with budget mask | O(n ceil((B+1)/w)) digit operations | O(ceil((B+1)/w)) digits |
| `approx` | Sorted merge and multiplicative trimming | O(n² log(B+1)/epsilon) arithmetic operations | O(n log(B+1)/epsilon) integer sums |

Here n/B are prepared size/budget and w is integer digit width. Public calls also
scan the N original items and use O(N) preparation storage. Exact numeric-budget
dependence is pseudopolynomial. Integer operand bit costs and Python object
overhead are separate from these state/operation bounds.

Approximation requires decimal text in `(0, 1)`, parsed with `Fraction`. `0.10`
and `.25` are valid; fraction syntax, scientific notation, signs, whitespace and
floating-point arguments are rejected. Validation applies even on trivial input.
With delta = epsilon/(2n), trimming retains y strictly when `d*y > (d+p)*last`,
where epsilon=p/q and d=2*n*q. Equality is discarded. The answer A satisfies
`(1-epsilon)*OPT <= A <= OPT`. Item order or epsilon may change A; individual
outputs and costs need not vary monotonically.

`--gcd-scale` supports exact methods only. Divide usable values by their GCD g,
solve with `B//g`, and multiply back. Both variants call the original exact cores.
An exact result equal to the budget answers the decision question with yes; a
smaller approximate result does not settle that question.

## Benchmarks and replay

The completed quick run is saved in [results/quick](results/quick/summary.csv).
Use a fresh directory for every run; existing output directories are refused:

```bash
# Choose another unused directory for subsequent reruns.
python3 benchmark.py --profile quick --out results/quick-rerun

# New full run; the completed original is already saved in results/full.
python3 benchmark.py --profile full --out results/full-rerun

# Replay actual saved values, budgets and configurations.
python3 benchmark.py --instances results/quick/instances.json --out results/quick-replay
```

| Setting | Quick | Full |
| --- | --- | --- |
| Seeds | 319 | 319, 320, 321 |
| Timed repeats | 2 | 3 |
| A item counts, budget 20,000 | 25, 100 | 25, 50, 100, 200 |
| B budgets, 80 items | 5,000; 20,000 | 5,000; 20,000; 80,000; 320,000 |
| C epsilon, broad and clustered families | 0.10, 0.40 | 0.05, 0.10, 0.20, 0.40 |
| D factors | 1, 10 | 1, 10, 100 |
| Instances / configurations | 9 / 28 | 40 / 140 |

C uses 60 items and budget 1,000,000. D uses a shared 60-item GCD-one base with
budget 1,500. Both profiles include the separate literature example. Quick/full
overlapping cases have identical inputs, generated using a local seeded `Random`.

Useful options: `--max-dp-work 20000000`, `--max-bitset-states 10000000`,
`--timeout 30`, `--memory-timeout 30`, `--repeats 3`, `--batch-target-ms 20`,
`--max-batch-count 65536`. Settings are saved. Caps apply after GCD scaling;
bitset limits count B+1 states. Trivial cases bypass core-allocation caps.

Timing and memory run sequentially in separate subprocesses. Timing performs one
untimed warm-up, bounded calibration, then recorded repeats with tracing disabled.
Raw records retain whole batch duration/count; summaries use median nanoseconds
per public `solve` call. Batch-loop overhead remains included. Memory tracing
starts after imports/loading around one public call without warm-up. The metric
is **peak traced Python allocation during solve**, not total process memory.

Each directory has `instances.json`, `timings.csv`, `summary.csv`, and
`environment.json`. Completed configurations are flushed incrementally and
summary snapshots are atomic. Failures retain explicit statuses/reasons. Memory
timeouts preserve valid runtimes; incomplete timing configurations retain raw
completed repeats but have no summary median. Missing exact references leave
optimum/error blank. Completed exact answers are compared and approximation
bounds checked using rational inequalities. Relative error is
`100*(OPT-answer)/OPT`, defined as zero when OPT is zero. Paired preprocessing
speedup is `unscaled_median/scaled_median` on the same instance.

See [the output format](docs/benchmark_format.md) for fields, status semantics,
measurement details, calibration metadata and exit codes.

## Figures and tables

The completed full run is in [results/full](results/full/summary.csv). Figures
and supporting tables are indexed in [figures/README.md](figures/README.md).
Install the plotting/report packages in a project-local environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The existing environment was tested with CPython 3.14.8, Matplotlib 3.10.8,
ReportLab 5.1.0 and pypdf 7.1.0. `requirements-lock.txt` records its full package
set; use that file instead to install exact versions on a compatible environment.
Regenerate all figures/tables from saved data without timing any algorithms:

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python plot_results.py --input results/full --out figures
MPLCONFIGDIR=.mplconfig .venv/bin/python -m unittest discover -s tests -v
```

Plots show medians across seed-level medians and seed ranges, not pooled repeat
samples or confidence intervals. Missing runs stay missing. The export verifies
raw duration normalization, summary medians, reachable answers, exact optima,
approximation bounds, errors and paired speedups. Its independent reachability
audit defaults to a maximum 10,000,000 states per case; this is configurable with
`--verification-states`. It retains blank reference/error cells from raw data.

Outputs include three 220-dpi PNGs, `aggregates.csv`, `allocation_table.csv/.md`,
`report_table.csv/.md`, `gcd_speedups.csv/.md`, `omitted_measurements.csv`,
`validation.json`, and `plot_environment.json` with source/input/output hashes.
The allocation table selects the largest completed budget per family/method,
then largest n, at epsilon 0.10. The compact report table uses the largest B
budget completed by all three methods on every seed: 80,000 in this run.

## Final report

The [five-page PDF](output/pdf/DAA_Subset_Sum_Report.pdf) includes the supplied
group details and all six required sections. Its editable source is
[report/build_report.py](report/build_report.py), with report-specific figure
assets and build metadata under `report/`. Regenerate from audited saved data:

```bash
MPLCONFIGDIR=.mplconfig .venv/bin/python report/build_report.py
```

The builder validates the saved full run, embeds the group details, checks page
count/section order, and writes `output/pdf/DAA_Subset_Sum_Report.pdf`. It does
not rerun benchmarks. `--input` selects another compatible full-profile run;
`--out` selects the PDF path. Every delivered page was rendered and inspected.
The final report route uses ReportLab; LaTeX is not required.

## Verification

All 56 unittest methods pass. The phase-2 suite includes 200 seeded instances in
three orders for all exact variants plus four epsilon values: 2,400 exact
comparisons and 2,400 original-subset feasibility/integer-bound checks. Tests also
cover fixed examples, the complete small-input grid, equality trimming, duplicate
items, floor budgets, GCD-one inputs, validation, immutability and actual CLI errors.
Runner tests cover real subprocesses, exceptions, timeouts, failure recovery,
partial records, caps, saved replay, and output-directory protection. Six new
analysis/export tests cover corrupt records, unreachable approximate answers,
resource limits, missing-point aggregation, common-budget selection, paired
speedups, and deterministic regeneration that preserves the raw files.

The quick profile completed all 28 configurations. The full profile has 137
completed timing/memory configurations and 3 planned standard-DP skips; all 411
recorded repeats and all 49 approximate answers passed post-run checks. The
three figures and all five report pages were visually inspected. See the results
analysis for measured trade-offs and limits. Report authoring and group details
are complete. Blackboard submission remains a user action.

The [original implementation plan](docs/implementation_plan.md) is unchanged.
[Phase-1 metadata](docs/environment_phase1.json) is a historical snapshot; current
source hashes/environment are saved with each benchmark run.
