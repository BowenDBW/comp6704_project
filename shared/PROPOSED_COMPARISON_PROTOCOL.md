# Benchmark Comparison Protocol

The common runner applies the following protocol to all four solver implementations.

## Inputs and accuracy

Each solver receives the same double-precision `A`, `b`, and `c_hat` arrays from
the 54 released instances. Returned variables retain the original column and row
order. Metrics are evaluated on the unscaled input arrays.

| Metric | Definition | Threshold |
|---|---|---:|
| Primal residual | `max(abs(A @ z - b))` | `<= 1e-7` |
| Primal nonnegativity | `min(z)` | `>= -1e-8` |
| Dual consistency | `max(abs(A.T @ y + r - c_hat))` | `<= 1e-7` |
| Dual feasibility | `min(r)` | `>= -1e-7` |
| Relative gap | `abs(f - b.T @ y)/max(1, abs(f), abs(b.T @ y))` | `<= 1e-7` |
| Reference error | `abs(f - f_ref)/max(1, abs(f_ref))` | `<= 1e-7` |

All values must be finite. `f = c_hat.T @ z`; `f_ref` is loaded from
`VALIDATION_SUMMARY.csv` after solving. Reference values do not enter the solver.
Negative components are retained when evaluating nonnegativity; they are not
clipped to zero. Internal solver tolerances and limits are saved in `environment.json`.

## Timing and counts

Each case receives one warmup and ten measured solves. Calls run serially with
one numerical thread; algorithm order rotates by case and repetition. Timing
includes initialization, factorization, iterations, and internal stopping checks.
Input loading, result saving, and independent verification are outside the timer.
The cold comparison contains 54 cases times four methods times ten repetitions.
Simplex counts include Phase I pivots where applicable; IPM counts Newton iterations.

Runtime summaries take the median of ten repetitions for each case, followed by
the median over the 18 cases at each size. Q1 and Q3 use linear interpolation
across case medians. Maximum errors use all 540 measured cold runs per method.

## Basis reuse

The first measured dual solve of each base instance supplies an optimal basis.
Each of ten warm repetitions copies the same retained indices and solves the
paired demand update. Factorization is recomputed and timed. The earlier base
solve is excluded from warm timing and recorded separately.

Cold and warm results are paired on the same updated instance. Speedup is the
median of nine per-case ratios of median cold time to median warm time, for each
size. Pivot reduction is `100*(1 - mean_warm_pivots/mean_cold_pivots)` over the
same nine cases. Zero-pivot outcomes are included.

## Records and verification

`recorded_results/` contains the common-machine measurements used in the report.
New runs save all measured solution vectors and an environment record. The
summary script independently replays every saved solution against the original
inputs and reference objectives. Full commands are in the [root README](../README.md).
