"""Run all four solvers on the shared benchmark with one numerical thread."""
import os

for k in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ[k] = "1"
import sys, json, csv, time, platform, hashlib, importlib.util, dataclasses, traceback, argparse
from pathlib import Path
import numpy as np
import scipy
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "runs/unified",
    help="New directory for run records and solutions",
)
args = parser.parse_args()
OUT = args.output.resolve()
if OUT.exists():
    parser.error(
        f"Output directory already exists: {OUT}. Choose a new --output directory."
    )
OUT.mkdir(parents=True, exist_ok=False)
(OUT / "solutions").mkdir()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


T = load("bench_tableau", "tableau_simplex/tableau_simplex.py")
R = load("bench_revised", "Revised_Simplex/revised_simplex.py")
D = load("bench_dual", "dual_simplex/dual_simplex.py")
I = load("bench_ipm", "algorithm_ipm/ipm_solver.py")
limits = {
    "primal_residual_inf": 1e-7,
    "min_z": -1e-8,
    "dual_residual_inf": 1e-7,
    "min_dual_slack": -1e-7,
    "relative_duality_gap": 1e-7,
    "relative_objective_error": 1e-7,
}
files = [
    ROOT / name
    for name in json.loads((ROOT / "recorded_results/input_sha256.json").read_text())
]


def hashes():
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
        if p.is_file()
    }


before = hashes()
(OUT / "input_sha256.json").write_text(json.dumps(before, indent=2))
manifest = list(
    csv.DictReader(
        (ROOT / "shared/benchmark_v1/manifest.csv").open(encoding="utf-8-sig")
    )
)
refs = {
    r["case_id"]: float(r["highs_objective"])
    for r in csv.DictReader(
        (ROOT / "shared/VALIDATION_SUMMARY.csv").open(encoding="utf-8-sig")
    )
}
names = ["Tableau", "Revised", "Dual", "IPM"]
bases = {}
base_times = {}
records = []


def run(name, entry, A, b, c, rep, mode="cold", basis=None):
    opts = {"M": int(entry["M"]), "N": int(entry["N"])}
    fn = {"Tableau": T.solve, "Revised": R.solve, "Dual": D.solve, "IPM": I.solve_ipm}[
        name
    ]
    args = (
        (opts,)
        if name == "Revised"
        else ((None, {"basis": basis}) if mode == "warm" else ())
    )
    started = time.perf_counter()
    try:
        ans = fn(A, b, c, *args)
        elapsed = time.perf_counter() - started
        if name == "IPM":
            ans = vars(ans)
    except Exception:
        elapsed = time.perf_counter() - started
        ans = {"status": "exception", "message": traceback.format_exc()}
    row = dict(
        case_id=entry["case_id"],
        M=int(entry["M"]),
        N=int(entry["N"]),
        alpha=entry["alpha"],
        algorithm=name,
        mode=mode,
        repeat_id=rep,
        runtime_seconds=elapsed,
        status=ans["status"],
        iterations=ans.get("iterations"),
        phase1_iterations=ans.get("phase1_iterations"),
        phase2_iterations=ans.get("phase2_iterations"),
        base_runtime_seconds=base_times.get(
            entry["case_id"].replace("demand_perturbed", "base"), ""
        )
        if mode == "warm"
        else "",
        message=ans.get("message", ""),
    )
    row.update({k: None for k in limits})
    row["objective"] = None
    row["validation_pass"] = False
    if all(ans.get(k) is not None for k in ["z", "y", "r"]):
        z, y, r = (ans[k] for k in ["z", "y", "r"])
        f = float(c @ z)
        dual = float(b @ y)
        row.update(
            objective=f,
            primal_residual_inf=float(np.max(np.abs(A @ z - b))),
            min_z=float(z.min()),
            dual_residual_inf=float(np.max(np.abs(A.T @ y + r - c))),
            min_dual_slack=float(r.min()),
            relative_duality_gap=abs(f - dual) / max(1, abs(f), abs(dual)),
            relative_objective_error=abs(f - refs[entry["case_id"]])
            / max(1, abs(refs[entry["case_id"]])),
        )
        row["validation_pass"] = bool(
            ans["status"] == "optimal"
            and all(
                np.isfinite(row[k])
                and (row[k] >= v if k.startswith("min_") else row[k] <= v)
                for k, v in limits.items()
            )
        )
        if rep > 0:
            arrays = {
                k: ans[k] for k in ["z", "y", "r", "basis"] if ans.get(k) is not None
            }
            np.savez_compressed(
                OUT / "solutions" / f"{entry['case_id']}_{name}_{mode}_{rep:02}.npz",
                **arrays,
            )
    records.append(row)
    writer.writerow(row)
    handle.flush()
    return ans, row


fields = [
    "case_id",
    "M",
    "N",
    "alpha",
    "algorithm",
    "mode",
    "repeat_id",
    "runtime_seconds",
    "status",
    "iterations",
    "phase1_iterations",
    "phase2_iterations",
    "base_runtime_seconds",
    "message",
    *limits,
    "objective",
    "validation_pass",
]
with threadpool_limits(limits=1), (OUT / "runs_including_warmups.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    env = {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "os": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "threadpools": threadpool_info(),
        "limits": limits,
        "timing": "External perf_counter around entire solver invocation; excludes input I/O, external validation and output I/O. Solver defaults unchanged. Serial, rotating algorithm order per case/repeat. One warmup (repeat 0) then 10 measured runs.",
        "options": {
            "Tableau": {"tol": 1e-9, "max_iter": 10000, "pivot_rule": "most_negative"},
            "Revised": R.DEFAULT_OPTIONS,
            "Dual": dataclasses.asdict(D.Options()),
            "IPM": dataclasses.asdict(I.IPMOptions()),
        },
        "warm_start": "Retain first measured Dual base optimal basis; no LU reuse; each warm invocation copies basis inside solver; paired perturbation; base time reported separately.",
    }
    assert all(p["num_threads"] == 1 for p in env["threadpools"])
    (OUT / "environment.json").write_text(json.dumps(env, indent=2))
    for idx, entry in enumerate(manifest):
        with np.load(ROOT / "shared/benchmark_v1" / entry["npz_file"]) as data:
            A, b, c = (data[k].copy() for k in ["A", "b", "c_hat"])
        snapshot = [x.copy() for x in (A, b, c)]
        for rep in range(11):
            offset = (idx + rep) % 4
            for name in names[offset:] + names[:offset]:
                ans, row = run(name, entry, A, b, c, rep)
                if (
                    name == "Dual"
                    and entry["kind"] == "base"
                    and rep == 1
                    and row["validation_pass"]
                ):
                    bases[entry["case_id"]] = ans["basis"].copy()
                    base_times[entry["case_id"]] = row["runtime_seconds"]
        assert all(np.array_equal(x, y) for x, y in zip(snapshot, (A, b, c)))
        print(f"cold {idx+1}/54 {entry['case_id']}", flush=True)
    for idx, entry in enumerate(e for e in manifest if e["kind"] != "base"):
        with np.load(ROOT / "shared/benchmark_v1" / entry["npz_file"]) as data:
            A, b, c = (data[k].copy() for k in ["A", "b", "c_hat"])
        key = entry["case_id"].replace("demand_perturbed", "base")
        basis = bases[key]
        original = basis.copy()
        for rep in range(11):
            run("Dual", entry, A, b, c, rep, "warm", basis)
        assert np.array_equal(original, basis)
        print(f"warm {idx+1}/27", flush=True)
    assert hashes() == before
    (OUT / "completion.json").write_text(
        json.dumps(
            {
                "rows": len(records),
                "measured": sum(r["repeat_id"] > 0 for r in records),
                "failed": sum(not r["validation_pass"] for r in records),
                "inputs_and_solver_sources_unchanged": True,
                "threadpools_end": threadpool_info(),
            },
            indent=2,
        )
    )
failed = sum(not row["validation_pass"] for row in records)
print(
    f"COMPLETE: {len(records)} runs; {failed} failed checks; results in {OUT}",
    flush=True,
)
if failed:
    raise SystemExit(1)
