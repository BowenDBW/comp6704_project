# GitHub Integration Notes

Upload this entire `tableau_simplex/` directory to the repository root beside
`dual_simplex/`, `algorithm_ipm/`, and `shared/`. Do not upload another copy of
the benchmark and do not upload local `__pycache__` directories.

Two repository-root edits are then needed.

## Root README status

Change the Tableau Simplex status row to:

```text
| Tableau Simplex | 已实现 Two-Phase Full-Tableau Simplex；54/54 实例、540/540 次计时通过 |
```

Add this quick-run command:

```text
python -m unittest discover -s tableau_simplex -p "test_tableau_simplex.py" -v
python tableau_simplex/run_benchmark.py --benchmark shared --output runs/tableau_all
```

The root README currently also marks IPM as waiting even though
`algorithm_ipm/` exists; that unrelated stale row should be corrected by the
IPM owner or repository maintainer.

## Root requirements

Append this line to the root `requirements.txt`:

```text
-r tableau_simplex/requirements.txt
```

The current root file references only `dual_simplex/requirements.txt`.

## Browser upload checklist

- Upload source, tests, runner, documentation and the tracked `results/` data.
- Do not upload `__pycache__/`, `.pyc`, local virtual environments, or temporary
  `runs/` directories.
- Preserve `shared/` exactly as it is.
- After upload, run the unit-test command from the repository root.
- Use a fresh `runs/tableau_all` output path for any new benchmark execution.
