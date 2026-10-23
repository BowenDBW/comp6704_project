"""Aggregate run records, replay saved solutions, and produce comparison figures."""
import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "mplcache"),
)
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[k] = "1"
import csv, json, statistics as st, math, shutil, hashlib, argparse, zipfile, io
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=ROOT / "runs/unified")
parser.add_argument(
    "--output",
    type=Path,
    help="Summary destination; defaults to the new run directory, or runs/recorded_summary for archived results",
)
args = parser.parse_args()
DATA = args.input.resolve()
OUT = (
    args.output.resolve()
    if args.output
    else (ROOT / "runs/recorded_summary" if DATA == ROOT / "recorded_results" else DATA)
)
if OUT == ROOT / "recorded_results":
    parser.error("Choose an output outside recorded_results.")
OUT.mkdir(parents=True, exist_ok=True)
archive = (
    zipfile.ZipFile(DATA / "solutions.zip")
    if (DATA / "solutions.zip").is_file()
    else None
)


def solution_file(filename):
    return (
        io.BytesIO(archive.read(filename)) if archive else DATA / "solutions" / filename
    )


assert (DATA / "completion.json").exists()
raw = list(
    csv.DictReader((DATA / "runs_including_warmups.csv").open(encoding="utf-8-sig"))
)
rows = [r for r in raw if int(r["repeat_id"]) > 0]
cold = [r for r in rows if r["mode"] == "cold"]
warm = [r for r in rows if r["mode"] == "warm"]
limits = json.loads((DATA / "environment.json").read_text())["limits"]
assert len(cold) == 2160 and len(warm) == 270 and len(raw) == 2673
assert (
    len(set((r["case_id"], r["algorithm"], r["mode"], r["repeat_id"]) for r in raw))
    == 2673
)
assert all(r["validation_pass"] == "True" for r in raw)
names = ["Tableau", "Revised", "Dual", "IPM"]
sizes = [(3, 5), (5, 10), (10, 20)]


def writecsv(name, data):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


case = []
for name in names:
    for mode in ["cold", "warm"]:
        rr = [r for r in rows if r["algorithm"] == name and r["mode"] == mode]
        for cid in sorted(set(r["case_id"] for r in rr)):
            g = [r for r in rr if r["case_id"] == cid]
            times = [float(r["runtime_seconds"]) * 1000 for r in g]
            assert len(g) == 10
            case.append(
                dict(
                    algorithm=name,
                    mode=mode,
                    case_id=cid,
                    M=int(g[0]["M"]),
                    N=int(g[0]["N"]),
                    alpha=float(g[0]["alpha"]),
                    runs=len(g),
                    median_ms=st.median(times),
                    q1_ms=float(np.quantile(times, 0.25)),
                    q3_ms=float(np.quantile(times, 0.75)),
                    median_iterations=st.median(float(r["iterations"]) for r in g),
                )
            )
summary = []
accuracy = []
for name in names:
    g = [r for r in cold if r["algorithm"] == name]
    accuracy.append(
        {
            "algorithm": name,
            "passed": len(g),
            "total": len(g),
            **{
                k: (min if k.startswith("min_") else max)(float(r[k]) for r in g)
                for k in limits
            },
        }
    )
    for M, N in sizes:
        g = [
            r
            for r in case
            if r["algorithm"] == name and r["mode"] == "cold" and r["M"] == M
        ]
        times = [r["median_ms"] for r in g]
        its = [r["median_iterations"] for r in g]
        summary.append(
            dict(
                algorithm=name,
                M=M,
                N=N,
                cases=len(g),
                median_ms=st.median(times),
                q1_case_ms=float(np.quantile(times, 0.25)),
                q3_case_ms=float(np.quantile(times, 0.75)),
                median_iterations=st.median(its),
                min_iterations=min(its),
                max_iterations=max(its),
            )
        )
ws = []
for M, N in sizes:
    w = [r for r in case if r["mode"] == "warm" and r["M"] == M]
    c = [
        next(
            r
            for r in case
            if r["algorithm"] == "Dual"
            and r["mode"] == "cold"
            and r["case_id"] == v["case_id"]
        )
        for v in w
    ]
    ci = st.mean(r["median_iterations"] for r in c)
    wi = st.mean(r["median_iterations"] for r in w)
    ws.append(
        dict(
            M=M,
            N=N,
            pairs=len(w),
            cold_mean_pivots=ci,
            warm_mean_pivots=wi,
            pivot_reduction_pct=100 * (1 - wi / ci),
            cold_median_ms=st.median(r["median_ms"] for r in c),
            warm_median_ms=st.median(r["median_ms"] for r in w),
            median_paired_speedup=st.median(
                a["median_ms"] / b["median_ms"] for a, b in zip(c, w)
            ),
            zero_pivot_cases=sum(r["median_iterations"] == 0 for r in w),
        )
    )
writecsv("measured_runs.csv", rows)
writecsv("case_summary.csv", case)
writecsv("size_summary.csv", summary)
writecsv("accuracy_summary.csv", accuracy)
writecsv("warm_start_summary.csv", ws)
# Replay every saved solution independently on original input arrays, without re-running solvers.
cache = {}
refs = {
    r["case_id"]: float(r["highs_objective"])
    for r in csv.DictReader(
        (ROOT / "shared/VALIDATION_SUMMARY.csv").open(encoding="utf-8-sig")
    )
}
for row in rows:
    cid = row["case_id"]
    if cid not in cache:
        with np.load(ROOT / "shared/benchmark_v1/instances" / f"{cid}.npz") as d:
            cache[cid] = tuple(d[k].copy() for k in ["A", "b", "c_hat"])
    A, b, c = cache[cid]
    with np.load(
        solution_file(
            f"{cid}_{row['algorithm']}_{row['mode']}_{int(row['repeat_id']):02}.npz"
        ),
        allow_pickle=False,
    ) as d:
        z, y, r = (d[k] for k in ["z", "y", "r"])
    f = float(c @ z)
    v = float(b @ y)
    metrics = dict(
        primal_residual_inf=float(np.max(np.abs(A @ z - b))),
        min_z=float(z.min()),
        dual_residual_inf=float(np.max(np.abs(A.T @ y + r - c))),
        min_dual_slack=float(r.min()),
        relative_duality_gap=abs(f - v) / max(1, abs(f), abs(v)),
        relative_objective_error=abs(f - refs[cid]) / max(1, abs(refs[cid])),
    )
    assert all(
        math.isfinite(metrics[k])
        and (metrics[k] >= val if k.startswith("min_") else metrics[k] <= val)
        for k, val in limits.items()
    ), row
(OUT / "independent_replay.json").write_text(
    json.dumps(
        {
            "saved_solutions_rechecked": len(rows),
            "passed": len(rows),
            "all_54_cases_per_algorithm": True,
            "all_10_repeats_per_case_mode": True,
        },
        indent=2,
    )
)
(OUT / "summary.json").write_text(
    json.dumps({"cold": summary, "accuracy": accuracy, "warm": ws}, indent=2)
)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
)
figdir = OUT / "figures"
figdir.mkdir(exist_ok=True)
colors = ["#5375A5", "#E18C32", "#319984", "#B65377"]


def save(fig, name):
    fig.subplots_adjust(bottom=0.24, top=0.88, wspace=0.25)
    for text in fig.texts:
        if text.get_position()[1] < 0:
            text.set_y(0.015)
    fig.savefig(figdir / (name + ".png"), dpi=240, bbox_inches="tight")
    fig.savefig(figdir / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(9, 5))
for name, color in zip(names, colors):
    g = [r for r in summary if r["algorithm"] == name]
    y = np.array([r["median_ms"] for r in g])
    lo = np.array([r["q1_case_ms"] for r in g])
    hi = np.array([r["q3_case_ms"] for r in g])
    ax.plot(range(3), y, "o-", label=name, color=color, lw=2, ms=7)
    ax.fill_between(range(3), lo, hi, color=color, alpha=0.10)
ax.set(
    xticks=range(3),
    xticklabels=["3 x 5", "5 x 10", "10 x 20"],
    yscale="log",
    xlabel="Transportation problem size (M x N)",
    ylabel="Median solve time (ms, log scale)",
    title="Cold-start runtime on the same computer",
)
ax.legend(ncol=2)
ax.grid(axis="y", alpha=0.2)
fig.text(
    0.5,
    -0.03,
    "Median of 18 case medians per size; shading: case IQR. Each case: 10 timed runs.\nSame Python / NumPy / SciPy; MKL and OpenMP verified at one thread.",
    ha="center",
    fontsize=10,
)
save(fig, "01_runtime")
fig, ax = plt.subplots(figsize=(9, 5))
for name, color in zip(names, colors):
    g = [r for r in summary if r["algorithm"] == name]
    ax.plot(
        range(3),
        [r["median_iterations"] for r in g],
        "o-",
        color=color,
        lw=2,
        label=name + (" (Newton updates)" if name == "IPM" else " (pivots)"),
    )
ax.set(
    xticks=range(3),
    xticklabels=["3 x 5", "5 x 10", "10 x 20"],
    yscale="log",
    xlabel="Transportation problem size (M x N)",
    ylabel="Median update count (log scale)",
    title="Iteration counts across problem sizes",
)
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.text(
    0.5,
    -0.03,
    "Simplex counts include initialization pivots. A pivot and a Newton update are not equal work.",
    ha="center",
    fontsize=10,
)
save(fig, "02_iterations")
fig, ax = plt.subplots(figsize=(9, 5))
vals = [r["relative_objective_error"] for r in accuracy]
ax.bar(names, vals, color=colors)
ax.axhline(1e-7, color="#555555", ls="--", label="Acceptance threshold: 1e-7")
ax.set(
    yscale="log",
    ylim=(1e-17, 2e-6),
    ylabel="Maximum relative objective error",
    title="All four methods meet the shared accuracy requirements",
)
for i, v in enumerate(vals):
    ax.text(i, v * 1.5, f"{v:.2e}", ha="center", fontsize=11)
ax.legend()
fig.text(
    0.5,
    -0.03,
    "540 / 540 cold runs passed per method; all six acceptance criteria checked.",
    ha="center",
    fontsize=10,
)
save(fig, "03_accuracy")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(3)
for ax, keys, label in [
    (axes[0], ("cold_mean_pivots", "warm_mean_pivots"), "Mean pivots"),
    (axes[1], ("cold_median_ms", "warm_median_ms"), "Median solve time (ms)"),
]:
    for shift, key, color, lab in [
        (-0.18, keys[0], colors[0], "Cold"),
        (0.18, keys[1], colors[2], "Warm"),
    ]:
        vals = [r[key] for r in ws]
        ax.bar(x + shift, vals, width=0.34, color=color, label=lab)
        for xx, v in zip(x + shift, vals):
            ax.text(xx, v * 1.12, f"{v:.2f}", ha="center", fontsize=10)
    ax.set(
        xticks=x, xticklabels=["3 x 5", "5 x 10", "10 x 20"], yscale="log", ylabel=label
    )
    ax.set_ylim(top=ax.get_ylim()[1] * 2)
    ax.legend()
fig.suptitle("Dual Simplex: reoptimization after demand changes", fontsize=16)
fig.text(
    0.5,
    -0.03,
    "9 paired perturbed cases per size; 10 runs per case/mode. Warm timing excludes the base solve.\nPivots: mean across cases. Time: median of case medians. No LU reuse.",
    ha="center",
    fontsize=10,
)
save(fig, "04_warm_start")
if archive:
    archive.close()
print(f"Verified {len(rows)} saved solutions. Summaries and figures: {OUT}")
