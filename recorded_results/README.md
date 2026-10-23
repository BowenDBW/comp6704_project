# Recorded Common-Machine Benchmark

The CSV files contain the measurements used in the report: 2,160 cold solves
and 270 dual warm solves, plus 243 warmups in the complete run log.
`solutions.zip` contains the 2,430 corresponding NPZ solution files.

`input_sha256.json` records the numerical inputs, generator and validator,
reference tables, and four solver implementations used for the measurements.
Documentation is excluded from this numerical-source manifest. Solver revision
and the recorded environment are identified in `environment.json`.

`case_summary.csv`, `size_summary.csv`, `accuracy_summary.csv`, and
`warm_start_summary.csv` aggregate the measured rows. The scripts in `benchmark/`
recalculate the summaries, replay the saved vectors, and generate figures.
Commands are in the [root README](../README.md). Newly generated outputs belong
in `runs/`, leaving the reported measurements unchanged.
