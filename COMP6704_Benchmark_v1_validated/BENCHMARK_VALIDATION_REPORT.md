# COMP6704 Benchmark Validation Report

Date: 2026-10-16 (Asia/Hong_Kong). **Technical result: PASS, 54/54 instances.** Group adoption of the proposed specification is still pending. No model, parameter, source implementation, or original generated data was changed. No Tableau Simplex, Revised Simplex, Dual Simplex, or IPM implementation was added.

Documentation addendum (2026-10-16): SHARED_PROBLEM_SPECIFICATION.md now unifies the problem narrative, assumptions, symbols and physical meanings. PROPOSED_COMPARISON_PROTOCOL.md supplies proposed v1 stopping metrics, interfaces, timing and warm-start conventions, with ENVIRONMENT_TEMPLATE.md and result_schema.csv. These are working proposals, not group approval or validation of student solvers. The numerical evidence below is unchanged; this documentation-only update does not regenerate instances. GROUP_MESSAGE_ZH.md is the distribution message.

## 1. Mathematical model and feasibility

Minimize sum_ij c_ij*x_ij subject to sum_j x_ij <= s_i, sum_i x_ij = d_j, and 0 <= x_ij <= u_ij. Introduce nonnegative p_i=s_i-sum_j x_ij and q_ij=u_ij-x_ij. All algorithms solve min c_hat.T@z subject to A@z=b, z>=0; slack costs are zero.

For each (M,N,seed), stream 0 is np.random.default_rng(np.random.SeedSequence([seed,M,N,0])) and sequentially samples costs U[1,20), demand U[10,30), and positive weights U[0.5,1.5). The witness is x_ij=w_ij/sum_i(w_ij)*d_j. Supplies are 1.2 times its row totals; capacities are (3-2*alpha)*x_ij. These factors are 2.4, 1.8, and 1.2. Separate stream 1 produces delta_j in [-0.05,0.05), with d'_j=d_j*(1+delta_j). Scaling witness columns by d'_j/d_j yields a feasible perturbed allocation: row totals and each route allocation rise by at most 5%, below the 20% minimum headroom. The witness is not an optimum or required initialization.

A has full row rank: supply slack columns force supply-row coefficients of a row dependence to zero, capacity slacks similarly force capacity-row coefficients to zero, and disjoint demand-row supports then force the remaining coefficients to zero.

## 2. Exact setup and environment

- Sizes: (3,5), (5,10), (10,20).
- Alpha: 0.3, 0.6, 0.9; seeds: 2026, 2027, 2028.
- 27 base and 27 paired demand-perturbed instances, 54 total.
- Python 3.11.15 (Anaconda), NumPy 1.26.4, SciPy 1.11.4; PCG64 via default_rng; Windows build 26200.
- Interpreter: E:/anaconda/python.exe. The system-default Python lacked NumPy; the bundled runtime lacked SciPy. Existing Anaconda dependencies were used without changing source or installing packages.
- First command, run from this release directory: `python instance_generator.py --output benchmark_v1 --validate-highs` (python resolved explicitly to the interpreter above).
- A separate Python process ran the identical copied source with `--output work/repro_benchmark_v1 --validate-highs` from the audit workspace. Only the destination differed.
- Independent validator: `python validate_benchmark.py benchmark_v1 <second-generation-directory>`.
- First generation: 54 successful built-in HiGHS validations. Second generation: another 54. Independent saved-data validation: 54 standard-form solves plus 54 original-form solves. No timings from these checks are algorithm-comparison results.

## 3. Dimensions and ordering

| M | N | A shape | b shape (NPZ) | c_hat shape (NPZ) |
|---|---|---|---|---|
|3|5|23 x 33|23|33|
|5|10|65 x 105|65|105|
|10|20|230 x 410|230|410|

Using zero-based indices: x_ij column i*N+j; p_i column M*N+i; q_ij column M*N+M+i*N+j. Rows are supply [0,M), demand [M,M+N), then row-major capacities. MAT stores b and c_hat as column vectors. Both formats contain only three user arrays A,b,c_hat (MAT also has loader-provided header metadata). Read the supplied matrix directly; MATLAB's default x(:) is not this row-major ordering.

## 4. Validation results

All 54 instances passed exact grid/count, shape, finite-input, independent Kronecker/block reconstruction of A, RNG/RHS/cost reconstruction, and nonnegative feasible-witness checks. Every generated NPZ and MAT in both runs loaded successfully (216 array files). Both JSON configurations and manifests were readable; first manifest dimensions, objectives and NPZ checksums were checked. Full 54-row results are in VALIDATION_SUMMARY.csv.

| Check | Result |
|---|---|
|Constructive witness feasibility|54/54; max infinity residual 1.4210854715202004e-14|
|HiGHS success/status 0 and finite primal vector/objective|54/54 independent saved-data solves|
|HiGHS primal feasibility|54/54; max infinity residual 2.1316282072803006e-14|
|HiGHS dual feasibility|max violation 1.7763568394002505e-15|
|Absolute primal-dual objective gap|max 9.0949470177292824e-13|
|Original inequality-model optimum vs standard form|54/54, absolute difference <= 1e-7|
|NPZ/MAT A,b,c_hat equality|54/54, exact after vector shape normalization|
|Base/perturbed invariants|27/27, A and c_hat unchanged; only demand RHS changes|
|Adjacent-alpha comparisons|36/36 across both kinds; A,c_hat,supply,demand identical; every capacity decreases strictly|
|Second-generation A,b,c_hat equality|54/54, exact for both formats|
|Original source/data preservation|112/112 files unchanged by SHA-256|

Thresholds: witness equality/nonnegativity 1e-8; solver equality <1e-7; solver nonnegativity 1e-8; dual violation and absolute duality gap <=1e-7. Exact array checks use np.array_equal, not a tolerance. Solver primal variables, objectives and marginals were finite. HiGHS upper-bound residuals can be +infinity because standard-form variables have no finite upper bound; this is expected metadata, not a nonfinite solution.

## 5. Reference objectives

Reference optimum range: 449.07651248888965 to 3904.3855240640569. All 54 values are in VALIDATION_SUMMARY.csv with round-trip float precision. Original manifest.csv retains the generator's 12-significant-digit formatting; its largest absolute rounding difference is 4.8282799980370328e-09. Use the validation CSV for tighter objective comparisons. No unique optimum or solver basis is asserted or distributed.

## 6. Numerical versus byte reproducibility

Exact numerical equality was confirmed from independent generation, not merely inferred from equal seeds. This establishes reproducibility in the recorded environment, not a promise across arbitrary future NumPy versions.

NPZ byte hashes matched in 54/54 pairs; MAT byte hashes matched in 0/54 pairs. For 54/54 MAT pairs, every byte after the 128-byte MAT header matched, localizing the difference to file header metadata. Both MAT loads yielded exactly identical arrays. Byte-for-byte reproducibility is therefore not claimed for the full release; numerical reproducibility passes.

## 7. Confirmed issues and severity

No mathematical or implementation error was found for the requested configuration. Low-severity limitations: `assert` checks are disabled by `python -O`; original manifest objectives are rounded; built-in validation alone omits MAT reload, full finiteness checks and a second generation; exporting into an occupied directory can overwrite matching files or leave partial output after failure. These do not invalidate this release: ordinary Python, fresh output directories and the independent validator were used. See PHASE1_AUDIT.md for the pre-generation review. Source corrections are not required and none were made.

Optional improvements are dependency pinning in future generator releases, explicit exceptions, full-precision manifest objectives and MAT hashes. This package supplies environment versions, full-precision validation objectives and both-format hashes without modifying the generator. The old data directory was preserved, not recertified as a separate release.

## 8. Distribution recommendation and remaining decisions

The package is technically ready for group review and distribution as a **proposed** benchmark. Group approval remains needed for the model/order, 27+27 instance grid, shared timing environment/thread settings, primal/dual stopping tolerances, and cold-start versus warm-start timing conventions (including initialization cost). Existing README.md remains the mathematical specification; START_HERE.md provides the release workflow. A feasible witness is not a basic feasible solution. A Dual Simplex warm start requires the member's own valid prior optimal basis; NPZ/MAT files do not contain one. An RHS change preserves reduced costs for a fixed basis, but this audit does not claim every perturbation forces a pivot.

Recommended next step: circulate this validated package for specification approval, then have each member load one common case and report objective and primal residual before implementing the complete comparison protocol.
