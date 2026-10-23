# COMP6704 Two-Phase Tableau Simplex

Common-machine report results and full reproduction: [root README](../README.md).


Independent implementation of Algorithm 1 for the shared capacitated
transportation LP benchmark. The solver consumes the released `A`, `b`, and
`c_hat` arrays directly and does not regenerate or modify benchmark data.

## Contents

| Path | Purpose |
|---|---|
| `tableau_simplex.py` | Two-phase primal full-tableau solver |
| `test_tableau_simplex.py` | Feasible, infeasible, unbounded, degenerate, redundant-row and benchmark tests |
| `run_benchmark.py` | Protocol-aligned 54-case runner |
| `summarize_results.py` | Per-case and per-size aggregation |
| `plot_results.py` | Runtime, pivot-count and accuracy figure generation |
| `IMPLEMENTATION_VALIDATION_REPORT.md` | Measured validation results and limitations |
| `results/` | Individual validation results, summaries, environment, solutions, histories and figures |

## Solver interface

```python
from tableau_simplex.tableau_simplex import solve

result = solve(
    A,
    b,
    c_hat,
    options={
        "tol": 1e-9,
        "max_iter": 10000,
        "pivot_rule": "most_negative",
    },
)
```

The result contains `z`, `y`, `r`, `basis`, status, objective, runtime,
Phase I/II pivot counts and separate objective histories. `basis` uses
zero-based original-column indices. `x` is retained as an alias of `z` for
backward compatibility with the development version.

## Algorithm

Phase I reverses negative-RHS rows, appends one artificial variable per
equality, and minimizes the artificial-variable sum. The objective row is
canonicalized for the artificial basis. Zero artificial basic variables are
pivoted out; a row is removed only when its canonical original-variable part
is zero, making it redundant.

Phase II restores `c_hat`. The default entering rule is the most negative
nonbasic reduced cost, and the leaving row uses the positive-coefficient
minimum-ratio test. `pivot_rule="bland"` is available. Every pivot performs a
Gauss-Jordan update of the complete constraint and objective tableau.

## Run from repository root

Install the root requirements, then run:

```text
python -m unittest discover -s tableau_simplex -p "test_tableau_simplex.py" -v
python tableau_simplex/run_benchmark.py --benchmark shared --output runs/tableau_all
python tableau_simplex/summarize_results.py runs/tableau_all/runs.csv
python tableau_simplex/plot_results.py runs/tableau_all/runs.csv --output runs/tableau_all/tableau_results_overview.png
```

Use a new output directory for each benchmark run. `run_benchmark.py` refuses
to overwrite an existing directory. It verifies that every shared input file
has the same SHA-256 before and after execution.

For a quick calibration run:

```text
python tableau_simplex/run_benchmark.py --benchmark shared --output runs/tableau_small --small-only
```

## Result protocol

The runner writes the exact `shared/result_schema.csv` fields to `runs.csv`,
uses protocol version `v1`, algorithm label `Tableau Simplex`, cold starts,
one warm-up and ten independent timed repetitions by default. Reference
objectives are loaded only after each solve for external validation.

Runtime includes input validation and copies inside the solver, Phase I setup
and pivots, artificial cleanup, Phase II setup and pivots, and final certificate
recovery. Disk loading, result serialization and reference comparison are
outside the timed solver call.

## Limitations

The solver uses dense float64 tableaux. Each pivot updates `O(mn)` values and
Phase I temporarily stores `n + m` columns, so it is less memory- and
compute-efficient than Revised Simplex. There is no presolve, scaling, sparse
storage, timeout or warm start. The included timings are local validation data,
not a formal cross-algorithm ranking.
