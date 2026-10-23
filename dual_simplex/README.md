# Dual Simplex

`dual_simplex.py` implements dual revised simplex for the shared transportation
model using NumPy and SciPy linear algebra.

## Interface

```python
from dual_simplex.dual_simplex import solve, Options
base = solve(A, b, c_hat, Options())
warm = solve(A, b_new, c_hat, warm_start={'basis': base['basis']})
```

Inputs use the ordering in [shared/README.md](../shared/README.md). The result
contains `z`, `y`, `r`, `basis`, `history`, `status`, and numerical metrics.
Basis indices are zero-based original column indices. Inputs and supplied basis
indices are copied internally.

## Algorithm

Cold initialization selects one minimum-cost route per customer, every supply
slack, and every route capacity slack. Demand equations determine the selected
shipments; slack equations determine the remaining basic variables. The basis
is nonsingular, and its shipment reduced costs are
`c[i,j] - min_k c[k,j] >= 0`. Negative slack values are repaired by dual pivots.

Each iteration recomputes a dense LU factorization. The leaving variable has
the smallest global index among basic values below `-1e-8`. Entering columns
have transformed row coefficients below `-1e-12`; the minimum dual ratio selects
the entrant, with exact ties broken by column index. Internal dual tolerance
is `1e-9`. Limits are 10,000 pivots and 120 seconds.

Basis reuse retains only column indices. The solver recomputes the factorization
and basic values for the new RHS. Reduced costs remain unchanged when the
matrix and objective are unchanged. The method returns `numerical_failure`
when no eligible pivot is available under the numerical tolerances.

## Tests and results

From the repository root:

```bash
python -m unittest discover -s dual_simplex -p "test_dual_simplex.py" -v
python dual_simplex/run_validation.py --benchmark shared --output runs/dual_small --small-only
```

`results_small/` and `results_all/` contain earlier individual validation batches.
The common-machine report results and complete four-method reproduction commands
are in the [root README](../README.md). The unified runner retains the first
measured base basis; the older individual runner retains its final base basis.
