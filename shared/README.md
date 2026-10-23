# Shared Capacitated Transportation Benchmark

For warehouses `i = 0,...,M-1` and customers `j = 0,...,N-1`, solve

```text
minimize    sum_ij c_ij * x_ij
subject to  sum_j x_ij <= s_i
            sum_i x_ij  = d_j
            0 <= x_ij <= u_ij
```

All variables are continuous. Supply is an upper bound and demand must be met
exactly. The instances are synthetic and use consistent abstract cost and
quantity units.

## Standard form

Adding supply slacks `p` and route capacity slacks `q` gives
`min c_hat.T @ z`, subject to `A @ z = b` and `z >= 0`.

- Variable order: `z = [X.ravel(order='C'), p, Q.ravel(order='C')]`.
- Shipment column: `i*N + j`.
- Supply slack column: `M*N + i`.
- Capacity slack column: `M*N + M + i*N + j`.
- Row order: `M` supply rows, `N` demand rows, `M*N` capacity rows.
- `A.shape = (M + N + M*N, 2*M*N + M)`; slack objective coefficients are zero.

All physical capacities are already encoded in `A` and `b`. Full-precision
reference objectives are in `VALIDATION_SUMMARY.csv`; the objective values in
`benchmark_v1/manifest.csv` are rounded and are not used for tight error checks.

## Generation

For each size and seed, NumPy's generator uses
`SeedSequence([seed, M, N, 0])` and draws, in order:

```text
c[i,j] ~ Uniform(1,20)
d[j]   ~ Uniform(10,30)
w[i,j] ~ Uniform(0.5,1.5)
x_ref[i,j] = w[i,j] * d[j] / sum_i w[i,j]
s[i] = 1.2 * sum_j x_ref[i,j]
u[i,j] = (3 - 2*alpha) * x_ref[i,j]
```

Sizes are `(3,5)`, `(5,10)`, `(10,20)`; `alpha` is `0.3`, `0.6`, `0.9`;
seeds are `2026`, `2027`, `2028`. The grid contains 27 base instances.
For fixed size and seed, changing alpha changes only route capacities.

The independent stream `SeedSequence([seed, M, N, 1])` draws
`delta[j] ~ Uniform(-0.05,0.05)` and sets `d_new[j] = d[j]*(1+delta[j])`.
The 27 paired instances retain the same `A`, costs, supplies, and capacities.
The perturbation vector is shared across capacity levels for a fixed size and seed.
The reference allocation proves feasibility; solvers do not use it as an optimum
or prescribed initialization.

## Files and commands

`benchmark_v1/instances/` contains the same arrays in NPZ and MAT formats.
Only `A`, `b`, and `c_hat` are stored. The manifest identifies all 54 cases.
To regenerate into a separate directory from the repository root:

```bash
python shared/instance_generator.py --output runs/generated_benchmark --validate-highs
```

To load an existing case:

```python
import numpy as np
with np.load('shared/benchmark_v1/instances/M03_N05_a030_seed2026_base.npz', allow_pickle=False) as data:
    A, b, c_hat = data['A'], data['b'], data['c_hat']
```

The [comparison protocol](PROPOSED_COMPARISON_PROTOCOL.md) defines the shared
acceptance metrics and aggregation. Full reproduction commands are in the
[repository README](../README.md).
