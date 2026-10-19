# Phase 1 source audit

Reviewed the complete original README.md and instance_generator.py before generation. No source changes are needed for the specified benchmark.

The model is min sum(c*x), with row sums of x <= s, column sums = d, and 0 <= x <= u. Standard form uses z=[x in C order,p,q in C order], supply/demand/capacity rows, and zero slack costs. All indices and dimensions are correct. A has full row rank: p and q columns eliminate supply/capacity coefficients in any row dependence, then each demand row has disjoint x support.

RNG uses SeedSequence([seed,M,N,0]) for sequential cost, demand and weight draws; stream 1 draws demand perturbations. Alpha does not affect RNG. x_ref distributes each demand across strictly positive weights. Supplies are 1.2 times witness row totals; capacities are (3-2*alpha)*x_ref, with factors 2.4,1.8,1.2. The base witness is strictly feasible. The perturbed witness multiplies column j by 1+delta_j in [0.95,1.05], hence respects both 1.2 headroom bounds. Only demand RHS changes. A and objective stay fixed across alpha and perturbation.

## Confirmed limitations (not model errors)

- Low severity: assertion-based checks disappear under python -O. Use ordinary Python execution. External validation uses explicit exceptions.
- Low severity: manifest reference objectives use 12 significant digits; rounding is unsuitable as an exact float reference for extremely tight absolute objective tolerances. The validation CSV adds full float precision and quantifies the rounding error.
- Low severity: built-in checks do not reload MAT files, explicitly test all solver values for finiteness, or perform a second numerical generation. These are validation coverage gaps addressed by the separate validator.
- Low severity: export overwrites matching names in an existing output folder and a failed run can leave partial output. Use a fresh destination as done here.

## Optional improvements, not implemented

Pin dependency versions for future reruns; replace assert with explicit validation exceptions; add MAT hashes and full precision objective storage. Current configuration records the RNG recipe but does not itself pin the NumPy version. Same-environment reproducibility will be tested, not inferred from seeds.

No formulation, parameter, original source, or original data modifications are authorized or required. Specification remains proposed pending group approval. No student solver is implemented in this audit.
