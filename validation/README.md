# Earlier Small-Instance Validation

`small/` and `relocation_small/` contain earlier checks of the dual solver on
the first base instance and its demand update. Each batch contains 30 measured
calls and three warmups, with no failed acceptance checks.

These records are separate from the common-machine comparison in
`recorded_results/`. Current reproduction and verification commands are in the
[root README](../README.md).
