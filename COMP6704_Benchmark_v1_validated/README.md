# COMP6704 — Shared Transportation LP Benchmark v1.0 (proposed)

**Status:** proposed group-wide standard, pending agreement by all members.  
**Scope:** instance generator only. Four algorithms must solve exactly the same standard-form LP data; initialization strategies can differ.

## 1. Shared mathematical model

For warehouses `i=0,...,M-1` and customers `j=0,...,N-1` (0-based in code):

\[
\min_x \sum_{i,j} c_{ij}x_{ij}\quad \text{subject to}\quad
\sum_j x_{ij}\leq s_i,\quad
\sum_i x_{ij}=d_j,\quad 0\leq x_{ij}\leq u_{ij}.
\]

Convert with nonnegative supply slack `p[i]` and capacity slack `q[i,j]`:

\[
\min_{z\geq0}\;\hat c^T z \quad\mathrm{s.t.}\quad Az=b.
\]

This follows the formulation in the proposal (pages 4–5). The parameter choices below are **additional suggestions**, not fixed by the proposal.

## 2. Exact generator rules

For each `(M,N,seed)`, using independent random draws from `np.random.default_rng(np.random.SeedSequence([seed,M,N,0]))`:

- `c[i,j] ~ Uniform(1,20)` (continuous)
- `d[j] ~ Uniform(10,30)` (continuous)
- `w[i,j] ~ Uniform(0.5,1.5)` (continuous)
- `x_ref[i,j] = w[i,j] * d[j] / sum_i w[i,j]`
- `s[i] = 1.2 * sum_j x_ref[i,j]`
- `u[i,j] = [1+2*(1-alpha)] * x_ref[i,j]`

The generated LP is provably feasible because `x_ref` meets all demand equalities and does not exceed supplies or capacities. It is a **feasibility witness, not an optimal solution or mandated solver initialization**.

Sizes `(M,N)`: `(3,5), (5,10), (10,20)`; alpha: `0.3,0.6,0.9`; seeds: `2026,2027,2028`. This produces **27 base instances**. For fixed `(M,N,seed)`, changing alpha changes **only the capacity part of b**; costs, demands, supplies, and A remain exactly the same.

A paired perturbed instance is generated with separate RNG stream `SeedSequence([seed,M,N,1])`:

- `delta[j] ~ Uniform(-0.05,0.05)`
- `d_new[j] = d[j]*(1+delta[j])`
- `A_new = A`, `c_hat_new = c_hat`
- `b_new` differs **only in the demand segment**.

Since the tightest capacity factor is `1.2`, and supplies have 20% headroom, every ±5% demand-perturbed instance is also feasible via a scaled version of `x_ref`. This produces **27 paired perturbed instances** for Dual Simplex RHS reoptimization experiments.

## 3. Input matrix convention (must be identical across all solvers)

**Variable order**: `z = [x(:), p, q(:)]`, where `x` and `q` are in **row-major order** (warehouse first, customer second):

- `x[i,j]` column index `i*N+j`;
- `p[i]` column index `M*N+i`;
- `q[i,j]` column index `M*N+M+i*N+j`.

**Constraint row order**:

- Supply: rows `0...M-1` (RHS `s`)
- Demand: rows `M...M+N-1` (RHS `d`)
- Route capacities: rows `M+N...M+N+M*N-1` (RHS `u` in row-major order)

So

\[
A\in\mathbb{R}^{(M+N+MN)\times(2MN+M)},\quad
b\in\mathbb{R}^{M+N+MN},\quad
\hat c\in\mathbb{R}^{2MN+M}.
\]

Example `(M,N)=(5,10)`: `A` shape `(65,105)`, `b` shape `(65,)`, `c_hat` shape `(105,)`. Objective: `c_hat @ z`. All columns have lower bound zero, and there are **no other constraints** in the standard form.

**Do not use MATLAB's default column-major vectorization to build your own x ordering.** Read the already generated A/b/c_hat or explicitly use row-major indexing `i*N+j`. The `.mat` files store the same numerical arrays, with b/c_hat as column vectors.

## 4. Output files and naming

- `instances/M03_N05_a030_seed2026_base.npz` and `.mat`
- `instances/M03_N05_a030_seed2026_demand_perturbed.npz` and `.mat`
- `manifest.csv` — one row per case, with dimensions, alpha, seed, relative paths, SHA-256 checksum, and optional HiGHS reference optimum
- `benchmark_config.json` — version, rules, ordering, and reproducibility specification

**Each `.npz`/`.mat` contains exactly three arrays:** `A`, `b`, `c_hat`.

Python:

```python
import numpy as np
with np.load('instances/M05_N10_a060_seed2026_base.npz') as data:
    A, b, c_hat = data['A'], data['b'], data['c_hat']
# Feed A, b, c_hat into your own algorithm.
```

MATLAB:

```matlab
S = load('instances/M05_N10_a060_seed2026_base.mat');
A = S.A; b = S.b; c_hat = S.c_hat;
% Feed A, b, c_hat into your own solver.
```

## 5. How to run the generator

Requires Python 3, NumPy, SciPy.

```bash
python instance_generator.py --output benchmark_v1 --validate-highs
```

- The script checks nonnegative feasibility witnesses and `A@z=b` for every case.
- `--validate-highs` additionally independently solves each case and places reference optimal objectives in `manifest.csv` (HiGHS is **validation only**, never used instead of student-implemented solvers).
- `manifest.csv` offers a reproducible test list. Share the **same generated files**, not just the same random seeds.

## 6. Group conventions to confirm

1. Model and standard-form variable / constraint ordering exactly as above.
2. Use the proposed 27 base instances and 27 paired RHS perturbations.
3. Use one fixed language/runtime benchmark environment for comparable wall-clock times (e.g. same PC, same Python environment and BLAS thread configuration), or explicitly caveat timings if different environments are used.
4. All methods report objective value, feasibility residual `||Az-b||_inf`, min(z), iteration/pivot counts, runtime, and termination status. Compare objective to `manifest.csv` reference values; a solver claiming optimality should meet the agreed primal/dual tolerance.
5. **Algorithm starting points are not forced to match**: Primal Simplex, Dual Simplex and Primal-Dual IPM have different admissible initialization requirements. For RHS warm-start, reuse the prior optimal basis with only the demand segment of b changed.

## 7. Division of responsibility

**Instance generator owner**: maintains generator code, `manifest.csv`, standard form convention, data release, and independent validity checks. Algorithm owners implement solvers and return results keyed by `case_id`. The group compares results on identical released instances.
