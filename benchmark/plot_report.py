"""Generate the report comparison figure from a benchmark size summary."""
from pathlib import Path
import os
import argparse
import csv

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "runs" / "mplcache")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=ROOT / "recorded_results")
parser.add_argument("--output", type=Path, default=ROOT / "runs/report_figures")
args = parser.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
with (args.input / "size_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
    rows = {
        (r["algorithm"], int(r["M"]), int(r["N"])): r for r in csv.DictReader(stream)
    }
labels = ["3 × 5", "5 × 10", "10 × 20"]
methods = ["Tableau", "Revised", "Dual", "IPM"]
steps = [
    [
        float(rows[(name, m, n)]["median_iterations"])
        for m, n in [(3, 5), (5, 10), (10, 20)]
    ]
    for name in methods
]
times = [
    [
        round(float(rows[(name, m, n)]["median_ms"]), 2)
        for m, n in [(3, 5), (5, 10), (10, 20)]
    ]
    for name in methods
]
plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.65))
styles = [("o", "#2463a3"), ("s", "#bd6d1a"), ("^", "#27805b"), ("D", "#a74373")]
for ax, data, ylabel, title in zip(
    axes,
    [steps, times],
    ["Median step count", "Median solve time (ms)"],
    ["(a) Iteration counts", "(b) Cold-start runtime"],
):
    for method, ys, (marker, color) in zip(methods, data, styles):
        ax.plot(
            range(3),
            ys,
            label=method,
            color=color,
            marker=marker,
            markerfacecolor="none",
            markersize=5,
            linewidth=1.3,
        )
    ax.set(
        xticks=range(3),
        xticklabels=labels,
        yscale="log",
        xlabel="Warehouses × customers",
        ylabel=ylabel,
        title=title,
    )
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=0.5)
    ax.set_xlim(-0.12, 2.12)
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.015)
)
fig.subplots_adjust(left=0.09, right=0.96, bottom=0.20, top=0.79, wspace=0.33)
fig.savefig(OUT / "comparison_overview.pdf")
fig.savefig(OUT / "comparison_overview.png", dpi=220)
plt.close(fig)
