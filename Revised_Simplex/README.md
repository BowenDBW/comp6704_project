# Part 2: Revised Simplex — Code and Results

Common-machine report results and full reproduction: [root README](../README.md).


The solver passed all 54 released cases in cold-start mode: 540/540 measured runs and 54/54 warm-ups were optimal under the shared checks.

## Files

| File | Purpose |
|---|---|
| `revised_simplex.py` | Solves one LP. It does not read or write files. |
| `run_revised_simplex.py` | Loads cases, checks results, and writes the result files. |
| `README.md` | Short running and integration instructions. |
| `results/revised_simplex_results.csv` | All 540 measured runs in the shared result schema. |
| `results/case_summary.csv` | One summary row per case (54 rows). |
| `results/alpha_summary.csv` | Capacity-tightness comparison (9 rows). |
| `results/history/*.csv` | Phase II objective trajectories for 3 representative cases. |

## What the code does

In `revised_simplex.py`, `solve(A, b, c_hat, {"M": M, "N": N})` checks and copies the inputs, starts the timer, and coordinates two phases. `_phase_one` adds temporary variables to the demand rows to find a feasible basis, then removes those variables. `_simplex` performs the same revised-simplex pivot loop in both phases: solve the current basis equations, calculate reduced costs, choose an entering column, apply the minimum-ratio test, and replace one basis column. Phase II uses the original transport costs. `_optimality` checks feasibility and dual conditions before an `optimal` status is accepted. The returned dictionary also contains the solution, basis, iteration counts, and an in-memory history.

In `run_revised_simplex.py`, the default run loads the 54 supplied NPZ cases (27 base and 27 demand-perturbed). Each case has one warm-up and ten independent measured cold starts. The runner checks all six shared criteria, using the HiGHS reference objective only **after** each solve. It writes one row per measured run, case and alpha summaries, and three representative Phase II histories. Use `--small-only` for the shared first case or `--case-id` for another case; choose a separate output folder to preserve the full results.

## Run

The runner reads the repository's sibling `shared/` folder by default. From `Revised_Simplex`, run:

```sh
python -B run_revised_simplex.py
```

From the repository root, the same flags as the other group runners work:

```sh
python -B Revised_Simplex/run_revised_simplex.py --benchmark shared --output runs/revised_small --small-only
python -B Revised_Simplex/run_revised_simplex.py --benchmark shared --output runs/revised_all
```

The first command runs one warm-up and ten measured cold starts for `M03_N05_a030_seed2026_base`; add `--repeats 1` for a faster check. The second runs all 54 cases. The original `--benchmark-dir` and `--output-dir` flags remain valid. A run replaces matching CSVs in its chosen output folder, so use a new folder to keep earlier results. The bundled run-level CSV came from the original local validation run. Its case and alpha summaries were calculated from those same rows without rerunning the solver; the three histories came from separate representative reruns. For group integration, import `solve` from `revised_simplex.py` and pass the shared `A`, `b`, `c_hat` arrays and integer `M`, `N`. Inputs are not modified; basis indices are zero-based. The internal settings are reduced-cost/direction tolerances of `1e-10`, Phase I tolerance of `1e-8`, and at most 10,000 pivots per phase.

## Results

| File | Rows | What it shows |
|---|---:|---|
| `results/revised_simplex_results.csv` | 540 | One measured cold run per row: case and repeat IDs, status, objective, all shared feasibility/optimality checks, Phase I/II pivots, runtime, and environment/settings IDs. Its 21 columns follow `shared/result_schema.csv`. |
| `results/case_summary.csv` | 54 | One row per case: successful/total runs, median runtime, runtime IQR, median pivot counts of successful runs, and the largest available primal residual and relative reference error across measured runs. |
| `results/alpha_summary.csv` | 9 | One row per size and alpha: median objective across six cases and median paired relative change from alpha 0.3. |
| `results/history/*.csv` | 3 files | Iteration, original objective, and relative objective gap for Phase II of the same representative cases used by Tableau and IPM. |

The summaries are calculated from the run-level CSV. Runtime IQR is Q3 minus Q1 across successful measured runs, using inclusive quartiles; it is blank when fewer than two runs succeed. Other median fields are blank if no run succeeds. The alpha comparison matches the same size, seed, and base/perturbed kind before calculating `(objective_alpha - objective_0.3) / max(1, abs(objective_0.3))`; a missing pair leaves that comparison blank. The `successful_cases` and `paired_cases` columns reveal incomplete groups.

Each history contains only Phase II because Phase I minimizes a different, artificial-variable objective. Its relative gap is `abs(objective - reference) / max(1, abs(reference))`, calculated after solving. The bundled histories are separate reruns for the convergence plot, not additional rows in the 540-run timing results. Full `z`, `y`, and basis vectors are not saved, and the runner does not generate JSON.

For example, measured repeat 1 of `M03_N05_a030_seed2026_base` reports objective `776.202970694746`, 25 pivots (15 in Phase I and 10 in Phase II), and relative reference error about `1.46e-16`.

| Check across all 540 measured rows | Worst observed value | Shared limit |
|---|---:|---:|
| Primal residual, maximum | `1.39e-13` | `≤ 1e-7` |
| Minimum component of `z` | `-2.66e-15` | `≥ -1e-8` |
| Dual residual, maximum | `1.78e-15` | `≤ 1e-7` |
| Minimum dual slack | `-8.88e-15` | `≥ -1e-7` |
| Relative primal–dual gap, maximum | `8.09e-16` | `≤ 1e-7` |
| Relative reference objective error, maximum | `9.44e-16` | `≤ 1e-7` |

All 540 rows have `status=optimal` and `mode=cold`. The small negative minimum values are floating-point roundoff and remain in the CSV without being clipped to zero. The CSV also records the 54 successful warm-ups in its messages; warm-ups do not add measured rows.

Each size has 18 cases and 180 measured runs. The runtime range is the minimum to maximum of the 18 per-case median runtimes (ten runs per case). Pivot ranges are the minimum to maximum per-case counts; total pivots include Phase I and Phase II.

| Size `(M,N)` | Successful runs | Median runtime range | Phase I iteration range | Phase II iteration range | Iteration range |
|---|---:|---:|---:|---:|---:|
| `(3,5)` | 180/180 | 1.94–2.68 ms | 12–15 | 5–12 | 19–27 |
| `(5,10)` | 180/180 | 15.76–21.60 ms | 38–48 | 8–34 | 55–76 |
| `(10,20)` | 180/180 | 408.77–589.65 ms | 146–177 | 50–159 | 227–321 |

Across the six matched seed/kind cases at each size and alpha, the median objective and median paired change from alpha 0.3 were:

| Size `(M,N)` | Alpha | Median objective | Median paired change from alpha 0.3 |
|---|---:|---:|---:|
| `(3,5)` | 0.3 | 710.56 | 0.0% |
| `(3,5)` | 0.6 | 788.27 | 12.6% |
| `(3,5)` | 0.9 | 971.94 | 37.9% |
| `(5,10)` | 0.3 | 1084.41 | 0.0% |
| `(5,10)` | 0.6 | 1315.29 | 22.2% |
| `(5,10)` | 0.9 | 1829.53 | 70.4% |
| `(10,20)` | 0.3 | 2319.08 | 0.0% |
| `(10,20)` | 0.6 | 2837.52 | 22.4% |
| `(10,20)` | 0.9 | 3856.91 | 67.3% |

In these generated cases, tighter route capacities raised the optimal transport cost. The percentages are medians of matched relative changes, not ratios of the displayed median objectives.

These are local validation timings, **not** a formal speed comparison with the other algorithms. The group still needs a common machine and fixed runtime/thread settings for that comparison.
