# Upload package validation

Status: PASS.

- Six unit tests passed; see `unit_tests.txt`.
- The root README Python example executed successfully; see `readme_example.txt`.
- `small/` contains a fresh run from the upload package: base cold, perturbed cold, and perturbed warm, with 30 timed runs and 3 warmups, all passing.
- `relocation_small/` contains the same check after copying the package to E:\Work&Academic\PhD\COMP6704\COMP6704_GitHub_Ready. All 30 timed runs and 3 warmups passed there too.
- Maximum primal residual in each small check: 7.105427357601002e-15. Maximum relative objective error: 1.4667585696982353e-16.
- Original source snapshots were checked by SHA-256: all 125 shared files, 1699 Dual Simplex delivery files, and 112 original generator-package files remained unchanged. See `PACKAGE_CHECKS.json`.
- Copied solver, runner, and shared payload remain byte-identical. Only copied documentation paths and the copied delivery manifest were updated; the previous manifest is preserved here.
- The original full-suite results are included in `dual_simplex/`. The full suite was not rerun during this packaging task.

Timing records are separate batches; do not combine them into a common-environment four-algorithm comparison.
