# Dual Simplex implementation validation

Date: 2026-10-16. **PASS for the shared benchmark and shared acceptance criteria.** This is a local implementation validation, separate from the common-machine comparison in `recorded_results/`.

## Implementation

dual_simplex.py implements dual simplex pivots using fresh dense LU factorizations. Cold start selects each customer's cheapest transport route plus every supply/capacity slack, giving an invertible dual-feasible basis analytically. It uses no external LP solution or basis; phase1_iterations=0 because no auxiliary Phase I is needed. Warm start copies only the algorithm's own base optimal basis indices; all factorizations and RHS updates are timed.

## Small shared interface case

A has shape (23,33), b length 23 and c_hat length 33. The base objective is 776.202970694746; perturbed objective is 775.08896194828753. All original input arrays and warm basis remain unchanged. For each of three case/mode combinations: 1 warmup and 10 independent timed runs, all passed.

|case_id|mode|pivots|median milliseconds (local)|
|---|---|---|---|
|M03_N05_a030_seed2026_base|cold|8|0.7191|
|M03_N05_a030_seed2026_demand_perturbed|cold|8|0.7034|
|M03_N05_a030_seed2026_demand_perturbed|warm|2|0.3312|

These sub-millisecond timings are sensitive to machine/runtime overhead and are not a general speedup claim.

## Full benchmark

- 54 cold-start cases (27 base + 27 demand-perturbed), and 27 paired warm-start cases.
- 810 timed runs + 81 warmups; zero failures in either category.
- Max primal residual: 9.3036689463588118e-14.
- Max relative objective error against full-precision HiGHS references: 5.8700927219471515e-16.
- Max relative duality gap: 3.4499537476997051e-16.
- All runs satisfy min(z)>=-1e-8, dual residual<=1e-7 and min(r)>=-1e-7 as well as the primal/gap tests. No output clipping.
- Cold pivot count range: 6 to 909. Warm pivot count range: 0 to 5.
- 8/27 warm cases required zero pivots; the old basis remained primal feasible. This is expected behavior, not missing work.
- All 840 saved timed solutions across small and full runs were reloaded and checked independently for primal feasibility, reduced costs, basis support and trace length.

The runner does not pass reference objectives to solve. It computes reference errors only after solve returns, and independently reevaluates all common certificates on the supplied original arrays.

## Tests and environment

Six unittest methods passed: hand-computed route-capacity pivot; tied-cost degeneracy and unchanged-RHS zero-pivot warm solve; 15 additional random small transport cases with cold/warm comparison against external HiGHS and input/state preservation checks; iteration/time limits and malformed input; infeasible example not falsely marked optimal; rejection of a non-dual-feasible warm basis.

Python 3.11.15, NumPy 1.26.4, SciPy 1.11.4. MKL and OpenMP actual thread counts were limited to 1 with threadpoolctl. CPU/RAM details where available are in HARDWARE.json. Per-run environment/options and execution order are recorded in results_*/environment.json. Algorithm randomness: none. Default resource limits: 10000 pivots, 120 seconds, checked at iteration boundaries. Formal group resource limits remain pending.

Times include input/state copying, cold initialization or warm preparation, dense factorization, history/metric computation and stopping checks. File loading, saving and external validation are excluded. Repeats always restart from fresh cold initialization or the same saved base basis. base_runtime_seconds identifies the actual last base solve that supplied that saved basis; base cost is reported separately from warm reoptimization cost.

## Scope and preservation

This implementation recognizes the exact shared matrix and zero slack objective structure for cold start. It does not claim arbitrary-LP initialization, reliable generic infeasibility/unboundedness certificates, or optimized rank-one basis updates. A no-eligible-pivot event is reported as numerical_failure rather than an unsupported infeasibility claim. Floating-point pivot tolerances may matter outside the tested data.

All 125 shared-package files are unchanged, and the deployed shared package matches the workspace copy. All 112 original Generator/data files match their original SHA-256 snapshot. No shared benchmark parameter, matrix, code or distribution ZIP was changed for this implementation. This solver and its evidence reside in a separate directory.

Next step: use this interface and results in the group's joint integration; agree final runtime environment and resource limits before comparing against the other three implementations.
