# Tableau Simplex Implementation Validation Report

Date: 2026-10-19. Status: implementation validation PASS under shared
acceptance thresholds. The shared benchmark and generator were not modified.

## Implementation

The solver is an independent Two-Phase Primal Tableau Simplex implementation.
It uses a full dense tableau, explicit Gauss-Jordan pivots, a general artificial
identity basis for Phase I, safe artificial-basic cleanup, canonical Phase II
objective reconstruction, deterministic ratio ties, and optional Bland
entering/leaving rules. It does not call SciPy, HiGHS, CVXPY or another LP
solver during optimization.

## Tests

Eleven automated tests pass. They cover a known optimum, genuine Phase I use,
infeasibility, unboundedness, degeneracy, redundant equalities, negative RHS,
iteration limits, invalid non-finite input, an actual shared benchmark case,
and an independent SciPy HiGHS comparison on a small LP.

## Full benchmark results

All 54 released instances were run as cold starts, with one warm-up and ten
timed repetitions per instance. All 540 timed runs returned `optimal` and met
the shared primal, dual, gap, nonnegativity and reference-objective thresholds.

| Size `(M,N)` | Cases / runs | Median case runtime | Median total pivots | Pivot range |
|---|---:|---:|---:|---:|
| `(3,5)` | 18 / 180 | 1.36 ms | 38.5 | 27-42 |
| `(5,10)` | 18 / 180 | 7.45 ms | 132 | 82-155 |
| `(10,20)` | 18 / 180 | 239.41 ms | 553.5 | 311-608 |

Worst metrics across the 540 development runs:

- primal residual infinity norm: `3.55e-14`
- minimum primal variable: `-1.07e-14`
- minimum dual slack: `-1.07e-14`
- relative duality gap: `3.54e-16`
- relative objective error against HiGHS: `5.89e-16`

These measurements came from Windows, Python 3.13.0, NumPy 2.5.1 and SciPy
1.18.0. They are development validation results. Formal runtime comparison requires all four
algorithms to be rerun on the same approved environment with fixed thread
settings.

The final clean-package run also verified that all 125 files in the shared
release were byte-identical before and after execution.

## Known limitations

The full tableau is dense and does not exploit transportation sparsity. Phase I
uses one artificial variable per equality, so its storage and pivot cost grow
quickly. The implementation has no presolve, scaling, sparse storage, timeout,
basis-factorization updates or warm-start reoptimization.
