# Primal-Dual Interior-Point Method

`ipm_solver.py` implements Mehrotra's predictor-corrector method for
`min c_hat.T @ z`, `A @ z = b`, `z >= 0`.

```python
from algorithm_ipm.ipm_solver import solve_ipm, IPMOptions
result = solve_ipm(A, b, c_hat, IPMOptions())
```

The result contains primal and dual variables, reduced costs, status, iteration
count, residuals, and iteration history. Initialization uses shifted least-squares
estimates. Predictor and corrector directions share the normal-equation
factorization. Cholesky failure triggers diagonal regularization using levels
`0`, `1e-12`, `1e-10`, `1e-8`, and `1e-6`, with one refinement step.

The fraction-to-boundary parameter is
`min(0.9995, max(0.995, 1-mu))`. Internal primal, dual, and relative-gap tolerances
are `1e-9`, with at most 100 iterations. The solver does not call an external LP
optimizer. HiGHS is used in tests and for independent reference objectives.

Run the tests from the repository root:

```bash
python -X utf8 algorithm_ipm/test_ipm.py
```

`results/` contains the earlier individual benchmark and three convergence
histories. The report comparison uses the common-machine results in
`recorded_results/`. Full reproduction commands are in the [root README](../README.md).
