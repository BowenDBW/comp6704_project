# Capacitated Transportation: Comparison of Four LP Methods

COMP6704 group project comparing two-phase tableau simplex, revised simplex,
dual simplex, and Mehrotra's primal-dual interior-point method. All four solvers
minimize transportation cost subject to warehouse supply limits, customer demand
equalities, and route capacities.

The report uses the common-machine measurements in `recorded_results/`.
The numerical inputs and solver implementations correspond to revision
`d3b0b1f23a956c0491d7480fbafc14927398f74f`.

## Installation

Use Python 3.11 and run the following commands in a terminal:

```bash
git clone https://github.com/BowenDBW/comp6704_project.git
cd comp6704_project
python -m pip install -r requirements.txt
```

The recorded environment is Python 3.11.15, NumPy 1.26.4, SciPy 1.11.4, and
Intel Core i7-12700 on Windows, using MKL 2023.1. The runner restricts numerical
thread pools to one thread and saves the actual environment and solver options.

## Reproduce the experiments

```bash
python -X utf8 benchmark/reproduce.py --output runs/reproduction
```

The output directory must not already exist; use a different `--output` path for
another run. The command executes the full benchmark (2,160 measured cold starts,
270 measured dual warm starts, and 243 untimed warmups), verifies every saved
solution, and generates the summary tables and comparison figures. A failed
acceptance check produces a nonzero exit status. The complete run takes several
minutes on the recorded machine.

The two numerical stages can also be run separately:

```bash
python -X utf8 benchmark/run_unified.py --output runs/reproduction
python -X utf8 benchmark/summarize_unified.py --input runs/reproduction
python benchmark/plot_report.py --input runs/reproduction --output runs/reproduction/figures
```

`summarize_unified.py` independently recomputes the six acceptance metrics from
each saved solution and the original input arrays. Outputs include per-call run
CSVs, per-case summaries (`case_summary.csv`, `accuracy_summary.csv`,
`size_summary.csv`, `warm_start_summary.csv`), all saved solution vectors, the
environment record, and `figures/comparison_overview.pdf`.

## Verify the reported results

The archived measurements and all 2,430 saved solutions are included. A quick
check requires only the Python standard library:

```bash
python benchmark/verify_recorded_results.py
```

The checker verifies the file manifest, numerical input and solver hashes, run
coverage, recorded acceptance metrics, reference-objective errors, and all
reported table values. It does not run the solvers.

## Experimental protocol

- Sizes: 3 by 5, 5 by 10, and 10 by 20 warehouses and customers; three capacity
  levels, three seeds, and original or perturbed demands give 54 instances.
- Route capacity is `(3 - 2*alpha) * x_ref` with `alpha` in {0.3, 0.6, 0.9};
  each demand perturbation is independently uniform within plus or minus 5%.
- Each method receives one warmup and ten measured cold solves per instance,
  serially with one numerical thread; method order rotates by instance and
  repetition. Runtime is summarized by the median of ten repeats per instance,
  then by the median across the 18 instances at each size.
- Dual warm starts reuse the basis from the first measured base solve. Warm
  speedup is the median of nine paired cold/warm time ratios per size; pivot
  savings use the arithmetic mean of the nine matched pivot counts.

The six acceptance criteria, evaluated on the unscaled input arrays:

| Metric | Required value |
|---|---:|
| `max(abs(A @ z - b))` | at most `1e-7` |
| `min(z)` | at least `-1e-8` |
| `max(abs(A.T @ y + r - c_hat))` | at most `1e-7` |
| `min(r)` | at least `-1e-7` |
| `abs(f - b.T @ y) / max(1, abs(f), abs(b.T @ y))` | at most `1e-7` |
| `abs(f - f_ref) / max(1, abs(f_ref))` | at most `1e-7` |

Here `f = c_hat.T @ z`. Full-precision HiGHS reference objectives are read from
`shared/VALIDATION_SUMMARY.csv` after solving; the four solvers do not use a
reference optimum or a HiGHS basis for initialization.

## Recorded results

All four methods passed 540 of 540 measured cold runs. Maximum relative
objective errors were `5.89e-16`, `5.87e-16`, `4.71e-16`, and `2.35e-10` for
Tableau, Revised, Dual, and IPM, respectively.

| Size | Tableau median ms | Revised median ms | Dual median ms | IPM median ms |
|---|---:|---:|---:|---:|
| 3 by 5 | 1.28 | 2.72 | 0.80 | 1.28 |
| 5 by 10 | 6.89 | 19.31 | 3.79 | 3.20 |
| 10 by 20 | 240.28 | 437.88 | 183.79 | 27.73 |

Dual basis reuse gives median paired speedups of 4.36, 8.55, and 102.23,
with mean pivot reductions of 97.1% to 99.5%.

## Repository structure

| Directory | Purpose |
|---|---|
| `benchmark/` | Common runner, aggregation, verification, and report plotting |
| `recorded_results/` | Measurements, summaries, environment, and solution archive used in the report |
| `shared/` | Generator, 54 NPZ/MAT instances, model, and reference objectives |
| `tableau_simplex/` | Full-tableau solver and tests |
| `Revised_Simplex/` | Revised simplex solver and individual runner |
| `dual_simplex/` | Dual simplex solver, basis reuse, and tests |
| `algorithm_ipm/` | Mehrotra IPM solver and tests |
| `validation/` | Earlier small-instance validation records |
| `docs/` | Course requirements, literature, and background report materials |
| `runs/` | New local runs; excluded from version control |

Results within individual algorithm directories are earlier validation batches.
The common-machine comparisons in the report use `recorded_results/` exclusively.

## Solver tests

```bash
python -m unittest discover -s dual_simplex -p "test_dual_simplex.py" -v
python -m unittest discover -s tableau_simplex -p "test_tableau_simplex.py" -v
python -X utf8 algorithm_ipm/test_ipm.py
```

Revised simplex is also checked on every instance by the common runner and
independent solution replay.
