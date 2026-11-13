"""Build the five-page course report from audited saved experiments.

The editable report text and author list live here. Measurements are read from
results/full; this command does not rerun benchmarks or change raw results.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plot_results import aggregate_rows, common_budget_table, read_run, speedup_table

AUTHORS = [
    ("Om Tiwari", "2410110227"),
    ("Vedaant Walia", "2410110374"),
    ("Adya Singhal", "2410110027"),
    ("Manasvi Sharma", "2410110195"),
    ("Shivani", "2410110476"),
]
SECTIONS = ["Problem statement", "Algorithm", "Proof of correctness", "Complexity analysis",
            "Conclusion", "Gaps and future directions"]
SOURCES = [
    ("Demaine, Ku and Solomon. MIT 6.006, Lecture 18: Pseudopolynomial (2020).",
     "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/f62798df9d8020e1c92130bd76a26f20_MIT6_006S20_lec18.pdf"),
    ("M. Goldwasser. Fully Polynomial-Time Approximation Scheme for Subset Sum. Course handout.",
     "https://cs.slu.edu/~goldwasser/courses/slu/csci3100/2016_Fall/handouts/SubsetSumFPTAS.pdf"),
    ("S. Har-Peled. Approximation algorithms III, Section 14.2 (2023).",
     "https://courses.grainger.illinois.edu/CS473/fa2023/lec/notes/16_approx_III.pdf"),
    ("Python 3.14 documentation. Built-in Types; sys.int_info.",
     "https://docs.python.org/3.14/library/stdtypes.html"),
    ("C. Pollett. CS255 lecture, 8 May 2019: Subset Sum Approximation.",
     "https://www.cs.sjsu.edu/faculty/pollett/255.1.19s/Lec20190508.html"),
    ("Python 3.14 documentation. time.perf_counter_ns and tracemalloc.",
     "https://docs.python.org/3.14/library/tracemalloc.html"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_report_figures(aggregates, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from plot_results import COLORS, draw_curve

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                         "axes.labelsize": 9, "axes.spines.right": False, "axes.spines.top": False,
                         "axes.grid": True, "grid.alpha": 0.20, "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.35))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.20, top=0.77, wspace=0.28)
    for ax, experiment, xlabel, title in zip(axes, ("A", "B"), ("Items n", "Budget B"),
                                           ("A: B = 20,000", "B: n = 80")):
        records = [r for r in aggregates if r["experiment"] == experiment]
        for method, label in (("dp", "Standard DP"), ("bitset", "Bitset DP"), ("approx", "Trimming (epsilon 0.10)")):
            draw_curve(ax, [r for r in records if r["method"] == method], COLORS[method], label)
        ticks = sorted({r["parameter"] for r in records})
        if experiment == "B":
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
            ax.text(0.98, 0.04, "DP at 320k skipped", transform=ax.transAxes,
                    ha="right", fontsize=7.5, color=COLORS["dp"])
        ax.set_xticks(ticks)
        ax.set_yscale("log")
        ax.set(title=title, xlabel=xlabel, ylabel="Runtime (ms)")
        ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=3, fontsize=8)
    fig.savefig(output / "runtime_report.png", dpi=300, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.35))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.20, top=0.77, wspace=0.28)
    for ax, method, title in zip(axes, ("dp", "bitset"), ("Standard DP", "Bitset DP")):
        for scaled in (False, True):
            records = [r for r in aggregates if r["experiment"] == "D" and r["method"] == method + ("_gcd" if scaled else "")]
            draw_curve(ax, records, "#007d8a" if scaled else "#ad493b",
                       "With GCD scaling" if scaled else "Without scaling", linestyle="--" if scaled else "-")
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100], ["1", "10", "100"])
        ax.set_yscale("log")
        ax.set(title=title, xlabel="GCD g", ylabel="Runtime (ms)")
        ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=2, fontsize=8)
    fig.savefig(output / "gcd_report.png", dpi=300, facecolor="white")
    plt.close(fig)


def build(input_dir, output_pdf):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from pypdf import PdfReader
    import reportlab
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

    rows, cases, environment, audit = read_run(input_dir)
    if audit["cases"] != 40 or audit["answers_checked"] != 137:
        raise ValueError("this final report expects the verified full-profile dataset")
    assets = Path(__file__).resolve().parent / "assets"
    assets.mkdir(exist_ok=True)
    make_report_figures(aggregate_rows(rows, cases), assets)
    fonts = {}
    for family, prefix in (("DejaVu Serif", "Body"), ("DejaVu Sans", "Sans")):
        for suffix, weight, style in (("", "normal", "normal"), ("Bold", "bold", "normal"),
                                      ("Italic", "normal", "italic"), ("BoldItalic", "bold", "italic")):
            path = font_manager.findfont(font_manager.FontProperties(family=family, weight=weight, style=style))
            fonts[prefix + suffix] = path
            pdfmetrics.registerFont(TTFont(prefix + suffix, path))
        pdfmetrics.registerFontFamily(prefix, normal=prefix, bold=prefix + "Bold", italic=prefix + "Italic", boldItalic=prefix + "BoldItalic")
    mono_path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans Mono"))
    pdfmetrics.registerFont(TTFont("Mono", mono_path))
    fonts["Mono"] = mono_path

    body = ParagraphStyle("Body", fontName="Body", fontSize=11, leading=14.2, spaceAfter=6, alignment=TA_JUSTIFY)
    heading = ParagraphStyle("Heading", fontName="SansBold", fontSize=12.5, leading=16, spaceBefore=8, spaceAfter=6, keepWithNext=True)
    sub = ParagraphStyle("Sub", parent=body, fontName="BodyBold", spaceBefore=5, spaceAfter=3, keepWithNext=True)
    small = ParagraphStyle("Small", fontName="Body", fontSize=9.2, leading=12, spaceAfter=5)
    code_style = ParagraphStyle("Code", fontName="Mono", fontSize=8.9, leading=11.3, spaceBefore=3, spaceAfter=7,
                                backColor=colors.HexColor("#f3f4f5"), borderPadding=6)
    equation = ParagraphStyle("Equation", parent=body, alignment=TA_CENTER, fontSize=11, leading=16, spaceBefore=3, spaceAfter=7)
    caption = ParagraphStyle("Caption", fontName="Body", fontSize=9, leading=11.5, spaceAfter=8)
    title_style = ParagraphStyle("Title", fontName="SansBold", fontSize=17.5, leading=21, spaceAfter=6)
    course_style = ParagraphStyle("Course", fontName="Sans", fontSize=9.5, leading=12, spaceAfter=5)

    story = []
    def p(text, style=body):
        story.append(Paragraph(text, style))
    def h(text):
        p(text, heading)
    def code(text):
        story.append(Preformatted(text, code_style))
    def eq(text):
        p(text, equation)
    def page():
        story.append(PageBreak())
    def figure(filename, text):
        story.append(Image(str(assets / filename), width=493, height=169.1))
        p(text, caption)
    def values(exp, method, field="median_runtime_ns", **filters):
        return [row[field] for row in rows if row["experiment"] == exp and row["method"] == method and
                all(row[key] == value for key, value in filters.items()) and row[field] is not None]
    def median(exp, method, field="median_runtime_ns", **filters):
        return statistics.median(values(exp, method, field, **filters))
    def span(exp, method, field, scale=1, places=2, **filters):
        selected = values(exp, method, field, **filters)
        return f"{min(selected) / scale:.{places}f}-{max(selected) / scale:.{places}f}"

    # Page 1: title, problem and the complete implemented algorithm variant.
    p("Subset Sum Under a Budget:<br/>Comparing Exact and Approximation Algorithms", title_style)
    p("CSD319 - Design and Analysis of Algorithms | Problem 5 | 13 November 2026", course_style)
    for group in (AUTHORS[:2], AUTHORS[2:4], AUTHORS[4:]):
        p("; ".join(f"{escape(name)} ({roll})" for name, roll in group), small)
    h(SECTIONS[0])
    p("Given positive integer items <i>a</i> and a nonnegative budget <i>T</i>, find the largest sum of a subset that does not exceed <i>T</i>:")
    eq("OPT(<i>a</i>, <i>T</i>) = max { Σ<sub>i∈I</sub> a<sub>i</sub> : I ⊆ {0,...,N-1}, Σ<sub>i∈I</sub> a<sub>i</sub> ≤ T }.")
    p("Each position is usable once; equal-valued positions remain separate items. The empty subset is allowed. "
      "For [6, 10, 15] and T=17, OPT=16 from 6+10. This models packing or spending under a limit. "
      "The exact decision question is answered by OPT=T; an approximate answer below T cannot decide it [1,2].")
    p("We compare standard DP, packed bitset DP and a deterministic trimming approximation, measuring runtime, "
      "traced allocation and error. We also evaluate the standard GCD reduction as exact preprocessing; our "
      "contribution is its integration and reproducible evaluation, rather than a new subset-sum algorithm.")
    h(SECTIONS[1])
    p("The public <font name='Mono'>solve</font> validates every original item, removes zeros and values above T, "
      "and preserves order/duplicates. Negative/noninteger values and booleans are rejected. Empty input or T=0 "
      "returns 0; if all usable values fit, return their total. Otherwise let n be the usable count and B=T.")
    p("Standard DP [1]", sub)
    code("R = bytearray(B+1); R[0] = 1\nfor x in values:\n    for s from B down to x: R[s] = R[s] OR R[s-x]\nreturn largest s with R[s] = 1")
    p("Bitset DP [4]", sub)
    code("bits = 1; mask = (1 << (B+1)) - 1\nfor x in values: bits = (bits OR (bits << x)) AND mask\nreturn bits.bit_length() - 1")
    p("Trimming approximation [2,3]", sub)
    code("L = [0]; delta = epsilon/(2*n)\nfor x in values:\n    M = two_pointer_merge_unique(L, L+x), restricted to <= B\n    keep 0; then keep y only if y > last_kept*(1+delta)\n    L = retained candidates\nreturn largest value in L")
    p("Epsilon must be decimal text with 0&lt;ε&lt;1, validated even for trivial cases. For ε=p/q, "
      "use the strict integer comparison <i>d</i>y&gt;(<i>d</i>+p)last, d=2nq; equality is discarded. "
      "For exact GCD variants compute g=gcd(usable values), call the same core on a/g and ⌊B/g⌋, "
      "and multiply back by g. If g=1, call the unscaled core. GCD scaling is not offered for approximation.")
    page()

    # Page 2: concise but complete arguments for the implemented conventions.
    h(SECTIONS[2])
    p("Preparation", sub)
    p("Zeros do not change a sum. A value above the budget cannot occur in a feasible subset because all "
      "values are nonnegative. If the usable total fits, selecting all usable items attains the greatest "
      "possible sum. Otherwise preparation leaves exactly the feasible subset sums relevant to the solvers.")
    p("Standard dynamic programming", sub)
    p("After i items, R[s]=1 exactly when a subset of those positions sums to s. Initially only R[0]=1. "
      "For an item x, a subset either omits it or adds it to a subset of the earlier positions. Thus the new "
      "state is old R[s] OR old R[s-x]. Descending iteration leaves R[s-x] unmodified when it is read, "
      "so x is never reused. Induction proves the invariant; the greatest reachable s≤B is OPT.")
    p("Packed bitset dynamic programming", sub)
    p("Use the same invariant for bit s. Shifting the old integer by x creates exactly the sums using the "
      "current position; OR keeps sums omitting it. Masking deletes only sums above B. Positive future "
      "additions cannot make them feasible again. The highest set bit therefore gives the exact optimum.")
    p("Approximation feasibility and quality", sub)
    p("Every retained value is a sum of processed positions: merging adds the current item only to old sums, "
      "and trimming only deletes candidates. For every exact feasible prefix sum y after i items, we prove "
      "there is a retained representative z such that [3]")
    eq("y / (1+δ)<super>i</super> ≤ z ≤ y,     δ = ε/(2n).")
    p("The claim starts with zero. For an omitted item use its earlier representative. If a subset adds x, "
      "then z+x≥(y+x)/(1+δ)<super>i</super> because x≥0. A candidate at most B either survives or is "
      "represented by a kept value at least candidate/(1+δ). Hence one trimming pass loses at most one "
      "further factor. Discarding candidates above B is safe because the required representative never "
      "exceeds its feasible target. This proves the invariant by induction.")
    p("Apply the invariant to the optimal subset after n items. Since 1+u≤e<super>u</super> and "
      "e<super>-u</super>≥1-u, the returned value A satisfies")
    eq("A ≥ OPT/(1+ε/(2n))<super>n</super> ≥ e<super>-ε/2</super>OPT ≥ (1-ε)OPT.")
    p("Feasibility also gives A≤OPT≤T. This proves the guarantee for the strict comparison and "
      "prepared item count actually used. For the literature instance [104,102,201,101], T=308, ε=0.40, "
      "the implementation gives A=302 while OPT=307 [5]. The guarantee concerns error relative to OPT, "
      "not unused budget; it does not require monotonic output as ε changes.")
    p("GCD reduction", sub)
    p("Every original subset has the same positions in the scaled input and sum g·s′. The equivalence "
      "g·s′≤B if and only if s′≤⌊B/g⌋ preserves exactly the feasible subsets. Maximizing in scaled units "
      "and multiplying by g preserves OPT. For [60,100,150], T=179, solve [6,10,15] at 17 and return 160.")
    page()

    # Page 3: theory and experimental methodology. One combined table.
    h(SECTIONS[3])
    p("Let n and B describe the nontrivial prepared instance, w the bits per packed digit, and M the "
      "largest usable value. Standard DP makes O(nB) updates with O(B) bytes of reachability state. "
      "Its dependence on the numeric budget is pseudopolynomial, not polynomial in log B [1].")
    p("The bitset core uses O(n⌈(B+1)/w⌉) digit operations and O(⌈(B+1)/w⌉) digits, including a "
      "constant number of temporary integers. Python shifts/ORs are not constant-time for arbitrary "
      "integers; the measured interpreter has w=30 bits per digit [4].")
    p("Retained positive sums grow by a factor greater than 1+δ. Their count is "
      "O(log(B+1)/δ)=O(n log(B+1)/ε), also bounded by B+1 integer sums. Each merge/trim is linear "
      "in that count, yielding O(n² log(B+1)/ε) arithmetic operations [2]. Integer operand sizes "
      "and the finite rational encoding of ε add bit costs, so this is an arithmetic-operation bound.")
    p("The public call also scans N original values and allocates O(N) preparation storage. GCD "
      "preprocessing uses a conservative O(n log M) Euclidean-step bound plus scaling; exact cores then "
      "replace B by ⌊B/g⌋. The scaled list uses O(n) extra storage. These counts differ from Python "
      "peak allocation, which includes list/integer object overhead and solver temporaries.")
    p("Table 1 combines core bounds with the largest B-family budget completed by all three methods on "
      "all seeds: n=80, B=80,000, approximation ε=0.10. Runtime and peak allocation are medians across "
      "seeds; error is the maximum observed. State columns describe mathematical core storage.", caption)
    cells = [["Method", "Core time", "Core state", "Time\n(ms)", "Peak\n(KiB)", "Max error\n(%)"]]
    common = {row["method"]: row for row in common_budget_table(rows)}
    for method, label, complexity, state in (
        ("dp", "Standard DP", "O(nB)", "O(B) bytes"),
        ("bitset", "Bitset DP", "O(n⌈(B+1)/w⌉)", "O(⌈(B+1)/w⌉) digits"),
        ("approx", "Trimming", "O(n² log(B+1)/ε)", "O(n log(B+1)/ε) integers"),
    ):
        record = common[method]
        cells.append([label, complexity, state, f"{record['runtime_ms_median']:.4g}",
                      f"{record['peak_kib_median']:.4g}", f"{record['error_pct_max']:.4g}"])
    cell_style = ParagraphStyle("Cell", parent=small, fontSize=8.5, leading=11, spaceAfter=0)
    data = [[Paragraph(escape(cell).replace("\n", "<br/>"), cell_style) for cell in row] for row in cells]
    table = Table(data, colWidths=[66, 120, 126, 57, 59, 65], hAlign="LEFT")
    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eeeeee")),
                               ("LINEABOVE", (0,0), (-1,0), 0.7, colors.black),
                               ("LINEBELOW", (0,0), (-1,0), 0.5, colors.black),
                               ("LINEBELOW", (0,-1), (-1,-1), 0.7, colors.black),
                               ("VALIGN", (0,0), (-1,-1), "TOP"), ("TOPPADDING", (0,0), (-1,-1), 5),
                               ("BOTTOMPADDING", (0,0), (-1,-1), 5), ("LEFTPADDING", (0,0), (-1,-1), 4),
                               ("RIGHTPADDING", (0,0), (-1,-1), 4)]))
    story.append(table)
    p("Experimental design and measurement", sub)
    p("A varies n={25,50,100,200} at B=20,000. B varies budget {5,000;20,000;80,000;320,000} at "
      "n=80. Both draw values uniformly from [max(1,B//n), max(2,3B//n)] and require total&gt;B. "
      "C uses n=60, B=1,000,000 with broad values [1,100,000] or clustered values [49,000,51,000], "
      "testing ε={0.05,0.10,0.20,0.40}. D starts a 60-item base with [20,21], draws the other items "
      "from [20,100], and uses g={1,10,100} with budget 1,500g+(g-1).")
    p("All methods share saved inputs from local Random seeds 319,320,321. Measurements run sequentially "
      "on an Intel i5-13450HX, Linux x86_64, CPython 3.14.8. Timing uses perf_counter_ns, one untimed "
      "warm-up, bounded calibration toward 20 ms (at most 65,536 calls/batch), then three repeats; "
      "the summary stores median ns/call. Complete public-call work is timed with tracing disabled; "
      "startup, loading, generation and writing are excluded. Separate fresh processes trace one call "
      "with tracemalloc [6]. This measures peak traced Python allocation, not process RSS.")
    p("Workers have 30-second timing/memory deadlines. DP runs exceeding n·solver_budget=20,000,000 "
      "and bitsets exceeding 10,000,000 states are recorded as skipped, with limits applied after GCD "
      "scaling. The full run contains 40 instances and 140 configurations: 137 completed, 3 planned DP "
      "skips at B=320,000, 411 timed repeats, and no timeout or correctness failure.")
    page()

    # Page 4: two compact figures and measured interpretation.
    p("Complexity analysis (continued)", heading)
    figure("runtime_report.png", "Figure 1. A/B runtime scaling; both runtime axes are logarithmic. Lines show medians of "
           "seed-level medians and whiskers show seed ranges, not confidence intervals. DP at B=320,000 is missing because of the work cap.")
    p(f"Bitset DP was fastest on every A/B instance. At n=200, B=20,000, median runtime was "
      f"{median('A','bitset',original_n=200)/1e6:.3f} ms for bitsets, "
      f"{median('A','dp',original_n=200)/1e6:.0f} ms for standard DP and "
      f"{median('A','approx',original_n=200)/1e6:.0f} ms for trimming. Approximation need not beat "
      "the exact baseline: its Python merge/list work grows strongly with item count. At B=320,000, "
      f"bitsets completed around {median('B','bitset',original_budget=320000)/1e6:.3f} ms and trimming around "
      f"{median('B','approx',original_budget=320000)/1e6:.1f} ms; no DP time is inferred for the skipped runs.")
    figure("gcd_report.png", "Figure 2. Paired GCD experiment, n=60 and budget 1,500g+(g-1). The same base problem is reused per seed. "
           "Runtime includes divisor computation, scaling, the exact core and rescaling of the result.")
    speedups = {(row["gcd"], row["method"]): row["speedup_median"] for row in speedup_table(rows)}
    p(f"Speedup is unscaled/scaled runtime within each seed, then aggregated. At g=1, median "
      f"speedup was {speedups[1,'dp']:.3f}x for DP and {speedups[1,'bitset']:.3f}x for bitsets: no useful "
      f"compression, with visible overhead for packed DP. At g=10, the respective speedups were "
      f"{speedups[10,'dp']:.2f}x and {speedups[10,'bitset']:.2f}x; at g=100, "
      f"{speedups[100,'dp']:.2f}x and {speedups[100,'bitset']:.2f}x. All original-unit optima were preserved. "
      "At g=100 the core budget falls from 150,099 to 1,500; preprocessing is more visible in fast bitset calls.")
    p(f"For C, increasing ε from 0.05 to 0.40 reduced median trimming runtime from "
      f"{median('C','approx',family='broad',epsilon='0.05')/1e6:.1f} to "
      f"{median('C','approx',family='broad',epsilon='0.40')/1e6:.1f} ms (broad) and "
      f"{median('C','approx',family='clustered',epsilon='0.05')/1e6:.1f} to "
      f"{median('C','approx',family='clustered',epsilon='0.40')/1e6:.2f} ms (clustered). At ε=0.40, "
      f"clustered trimming took {span('C','approx','median_runtime_ns',1e6,family='clustered',epsilon='0.40')} ms "
      f"versus exact bitsets' {span('C','bitset','median_runtime_ns',1e6,family='clustered')} ms, with "
      f"{span('C','approx','relative_error_pct',places=4,family='clustered',epsilon='0.40')}% error. It traced only "
      f"{span('C','approx','peak_traced_bytes',1024,1,family='clustered',epsilon='0.40')} KiB versus about "
      f"{median('C','bitset','peak_traced_bytes',family='clustered')/1024:.1f} KiB. "
      "Broad-family bitsets remained faster at every tested ε.")
    p(f"A post-run audit confirmed all {audit['approximation_feasibility_and_bound_checks']} approximate outputs "
      f"were reachable and met the bound. Error was not always monotonic: broad seed 319 improved from "
      f"{median('C','approx','relative_error_pct',family='broad',seed='319',epsilon='0.05'):.4f}% at ε=0.05 to "
      f"{median('C','approx','relative_error_pct',family='broad',seed='319',epsilon='0.10'):.4f}% at ε=0.10, "
      "well below both guarantee ceilings. Table 1 also shows that trimming can allocate more than exact methods.")
    page()

    # Page 5: conclusions, limitations and compact linked references.
    h(SECTIONS[4])
    p("The Python project provides a common, tested optimization interface for two exact algorithms and "
      "a deterministic FPTAS, together with an exact GCD preprocessing comparison. Packed DP is the "
      "best measured choice on the A/B grids: it stays exact and substantially reduces runtime. "
      "Trimming offers a controlled loss in quality; on clustered large-budget inputs at loose ε it "
      "slightly improves runtime and greatly reduces traced allocation. The data do not support a "
      "general claim that approximation is faster or smaller.")
    p("GCD preprocessing is worthwhile when the inputs share a substantial divisor, but gives no "
      "guaranteed speed benefit when g=1. Our measured improvement comes from shrinking the numeric "
      "state range without changing feasible subsets. Saved sequences, raw durations, independent "
      "memory records, limits and source hashes make these findings reproducible and auditable.")
    h(SECTIONS[5])
    p("The study covers three synthetic seeds on one interpreter and one machine. Small seed ranges "
      "are not confidence intervals, and the reported winners need not generalize to different input "
      "distributions, languages or hardware. Three large ordinary-DP cases were deliberately unmeasured; "
      "their missing values are not estimates. Exact DP remains budget-dependent even with bit packing, "
      "and GCD savings disappear for relatively prime inputs.")
    p("Peak tracemalloc allocation omits untraced native memory and differs from RSS. Python object "
      "overhead, integer operand sizes and batching overhead affect the observations. Epsilon bounds "
      "relative error but does not predict typical error or monotonic outputs. The current program "
      "returns only the best sum, accepts nonnegative values, and does not reconstruct indices.")
    p("Future work could add index reconstruction with an explicit space analysis, test sparse and "
      "adversarial distributions, expand seeds/platforms, and compare meet-in-the-middle for small n "
      "with huge budgets. A full bit-cost treatment and process-RSS measurements would complement "
      "the current arithmetic bounds and traced-allocation metric.")
    p("Reproducibility", sub)
    p("The project retains solvers.py, cli.py, benchmark.py, plot_results.py, tests, results/full and "
      "figures. Benchmarks require fresh output directories; saved inputs can be replayed. The report "
      "builder reads the audited full run and recreates both report figures without measuring algorithms "
      "again. All 56 tests passed. Independent position-subset enumeration checks 200 small cases in "
      "three orders: 2,400 exact-variant comparisons and 2,400 feasibility/bound checks across four ε values.")
    p("References", heading)
    for index, (description, url) in enumerate(SOURCES, 1):
        additional = ""
        if index == 4:
            additional = " <link href='https://docs.python.org/3.14/library/sys.html#sys.int_info' color='#222222'><u>Integer digits</u></link>."
        if index == 6:
            additional = " <link href='https://docs.python.org/3.14/library/time.html#time.perf_counter_ns' color='#222222'><u>Timing API</u></link>."
        p(f"[{index}] {escape(description)} <link href='{url}' color='#222222'><u>Source</u></link>." + additional, small)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    width, height = A4
    def decorate(canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setFont("Sans", 8)
            canvas.setFillColor(colors.HexColor("#555555"))
            canvas.drawString(50, height - 30, "Subset Sum Under a Budget | CSD319")
        canvas.setFont("Sans", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawRightString(width - 50, 27, f"{doc.page} / 5")
        canvas.restoreState()
    doc = SimpleDocTemplate(str(output_pdf), pagesize=A4, leftMargin=50, rightMargin=50,
                            topMargin=46, bottomMargin=43, title="Subset Sum Under a Budget: Comparing Exact and Approximation Algorithms",
                            author="; ".join(name for name, _ in AUTHORS), pageCompression=1)
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    reader = PdfReader(output_pdf)
    text = "\n".join(page.extract_text() for page in reader.pages)
    if len(reader.pages) != 5:
        raise ValueError(f"report must be five pages; built {len(reader.pages)}")
    for name, roll in AUTHORS:
        if name not in text or roll not in text:
            raise ValueError(f"missing author details: {name}")
    lines = [line.strip() for line in text.splitlines()]
    positions = [lines.index(section) if section in lines else -1 for section in SECTIONS]
    if min(positions) < 0 or positions != sorted(positions):
        raise ValueError("required report sections are missing or reordered")
    if "\ufffd" in text or "\u25a0" in text:
        raise ValueError("a replacement or missing-glyph marker appears in the PDF")
    metadata = dict(
        generated_at=datetime.now(timezone.utc).isoformat(), pages=len(reader.pages), authors=AUTHORS,
        reportlab_version=reportlab.Version, python_version=sys.version, sections=SECTIONS,
        builder_sha256=sha(Path(__file__)), pdf_sha256=sha(output_pdf),
        input_hashes={name: sha(input_dir / name) for name in ("instances.json", "summary.csv", "timings.csv", "environment.json")},
        report_figure_hashes={path.name: sha(path) for path in sorted(assets.glob("*.png"))},
        font_hashes={name: sha(Path(path)) for name, path in fonts.items()}, audit=audit,
        reference_urls=[url for _, url in SOURCES],
    )
    (Path(__file__).resolve().parent / "build_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Built {output_pdf} ({len(reader.pages)} pages); author, section and source-data checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/full")
    parser.add_argument("--out", type=Path, default=ROOT / "output/pdf/DAA_Subset_Sum_Report.pdf")
    args = parser.parse_args()
    build(args.input, args.out)


if __name__ == "__main__":
    main()
