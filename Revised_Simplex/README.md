# Part 2: Revised Simplex — Code and Results

**One-sentence result:** The solver passed all 54 released cases in cold-start mode: 540/540 measured runs and 54/54 warm-ups were optimal under the proposed v1 checks.

## Files

| File | Purpose |
|---|---|
| `revised_simplex.py` | Solves one LP. It does not read or write files. |
| `run_revised_simplex.py` | Loads cases, runs the solver, checks results, and writes the CSV. |
| `README.md` | Short running and integration instructions. |
| `results/revised_simplex_results.csv` | Current measured results in the shared result schema. |

## What the code does

In `revised_simplex.py`, `solve(A, b, c_hat, {"M": M, "N": N})` checks and copies the inputs, starts the timer, and coordinates two phases. `_phase_one` adds temporary variables to the demand rows to find a feasible basis, then removes those variables. `_simplex` performs the same revised-simplex pivot loop in both phases: solve the current basis equations, calculate reduced costs, choose an entering column, apply the minimum-ratio test, and replace one basis column. Phase II uses the original transport costs. `_optimality` checks feasibility and dual conditions before an `optimal` status is accepted. The returned dictionary also contains the solution, basis, iteration counts, and an in-memory history.

In `run_revised_simplex.py`, the default run loads the 54 supplied NPZ cases (27 base and 27 demand-perturbed). Each case has one warm-up and ten independent measured cold starts. The runner checks all six proposed v1 criteria, using the HiGHS reference objective only **after** each solve. It writes one CSV row per measured run. A single case can be selected with `--case-id`; use a separate `--output-dir` so that this check does not replace the full results CSV.

## Run

The runner reads the repository's sibling `shared/` folder by default. From `Revised_Simplex`, run:

```sh
python -B run_revised_simplex.py
```

For one quick case check, keep its output separate:

```sh
python -B run_revised_simplex.py --case-id M03_N05_a030_seed2026_base --repeats 1 --output-dir results_check
```

The full run writes `results/revised_simplex_results.csv` after all selected cases finish, replacing that CSV if it exists. The bundled CSV was produced by the original local validation run before this repository copy. For group integration, import `solve` from `revised_simplex.py` and pass the shared `A`, `b`, `c_hat` arrays and integer `M`, `N`. Inputs are not modified; basis indices are zero-based. The internal settings are reduced-cost/direction tolerances of `1e-10`, Phase I tolerance of `1e-8`, and at most 10,000 pivots per phase.

## What the output contains

The current CSV has the 21 columns of the supplied `result_schema.csv`. Each row identifies the case and repeat, reports `status` and `objective`, gives feasibility/optimality metrics, separates Phase I and Phase II pivot counts, and records solve time and settings. It does not store the full `z`, `y`, basis, or iteration history, and the current runner does not generate JSON.

For example, measured repeat 1 of `M03_N05_a030_seed2026_base` reports objective `776.202970694746`, 25 pivots (15 in Phase I and 10 in Phase II), and relative reference error about `1.46e-16`.

| Check across all 540 measured rows | Worst observed value | Proposed v1 limit |
|---|---:|---:|
| Primal residual, maximum | `1.39e-13` | `≤ 1e-7` |
| Minimum component of `z` | `-2.66e-15` | `≥ -1e-8` |
| Dual residual, maximum | `1.78e-15` | `≤ 1e-7` |
| Minimum dual slack | `-8.88e-15` | `≥ -1e-7` |
| Relative primal–dual gap, maximum | `8.09e-16` | `≤ 1e-7` |
| Relative reference objective error, maximum | `9.44e-16` | `≤ 1e-7` |

All 540 rows have `status=optimal` and `mode=cold`. The small negative minimum values are floating-point roundoff and remain in the CSV without being clipped to zero. The CSV also records the 54 successful warm-ups in its messages; warm-ups do not add measured rows.

Each size has 18 cases. The runtime ranges below use each case's median across its ten measured runs; the iteration ranges use total Phase I + Phase II pivots.

| Size `(M,N)` | Median runtime range | Iteration range |
|---|---:|---:|
| `(3,5)` | 1.94–2.68 ms | 19–27 |
| `(5,10)` | 15.76–21.60 ms | 55–76 |
| `(10,20)` | 408.77–589.65 ms | 227–321 |

These are local validation timings, **not** a formal speed comparison with the other algorithms. The group still needs a common machine and fixed runtime/thread settings for that comparison.
