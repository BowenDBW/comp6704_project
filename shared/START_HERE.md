# Start here

Technical validation PASS (54/54); specification remains proposed pending group approval.

## 本次文档补充（2026-10-16，proposed protocol v1）

先阅读 [SHARED_PROBLEM_SPECIFICATION.md](SHARED_PROBLEM_SPECIFICATION.md)：从共同运输优化问题、建模假设、变量名字/定义/物理意义，到 standard-form 和报告组织统一说明。随后阅读 [PROPOSED_COMPARISON_PROTOCOL.md](PROPOSED_COMPARISON_PROTOCOL.md)：当前暂定容差、接口、计时与 cold/warm-start 规则。正式实验前由小组确认并填写 [ENVIRONMENT_TEMPLATE.md](ENVIRONMENT_TEMPLATE.md)。

[result_schema.csv](result_schema.csv) 是算法结果表头模板；[GROUP_MESSAGE_ZH.md](GROUP_MESSAGE_ZH.md) 是可直接发给组员的中文说明。新文档不改变原 README、Generator 或任何 benchmark 数据。下面的加载与复现说明仍适用；协议中的明确指标公式作为当前联调工作草案。

1. Read README.md for the mathematical specification and exact ordering.
2. Read BENCHMARK_VALIDATION_REPORT.md for evidence and limitations.
3. Select a case_id from VALIDATION_SUMMARY.csv. Its highs_objective is the full-precision reference; use the same case_id for all four algorithms.
4. Load benchmark_v1/instances/<case_id>.npz or .mat. Solve min c_hat.T@z, A@z=b, z>=0. There are no extra bounds to add; capacities are already encoded in A,b.
5. Report objective, ||A@z-b||_inf, min(z), termination status, iteration/pivot count and runtime. Agree primal/dual and objective tolerances with the group. Suggested objective diagnostic: abs(f-f_ref)/max(1,abs(f_ref)), together with primal feasibility.

```python
import numpy as np
case_id = 'M03_N05_a030_seed2026_base'
with np.load(f'benchmark_v1/instances/{case_id}.npz', allow_pickle=False) as data:
    A, b, c_hat = data['A'], data['b'], data['c_hat']
# z = your_solver(A, b, c_hat)
# objective = c_hat @ z
# residual = np.linalg.norm(A @ z - b, ord=np.inf)
# X = z[:3*5].reshape(3,5)  # row-major transport allocation
```

```matlab
case_id = 'M03_N05_a030_seed2026_base';
S = load(fullfile('benchmark_v1','instances',[case_id '.mat']));
A = S.A; b = S.b; c_hat = S.c_hat;
% z = your_solver(A,b,c_hat);
% objective = c_hat' * z;
% residual = norm(A*z-b,inf);
% X = reshape(z(1:3*5),5,3).'; % recover row-major allocation
```

For Dual Simplex RHS experiments use the matching _demand_perturbed case, retaining your own optimal basis from _base. The instance files contain data only, not a solver basis or mandated initialization.

## Reproduce verification

Use Python 3.11.15 with requirements-validated.txt. From this folder, generate into a new folder to preserve the delivered files:

```text
python instance_generator.py --output rerun_benchmark_v1 --validate-highs
python validate_benchmark.py benchmark_v1 rerun_benchmark_v1
```

The validator reloads both formats, rebuilds the specification independently, solves standard/original forms and checks all comparisons with explicit exceptions. It refreshes VALIDATION_SUMMARY.csv and VALIDATION_RESULTS.json; the supplied narrative report and source-integrity snapshot document the original audit run. Do not run the original generator with python -O. No student algorithm is included.

## Package contents

- README.md and instance_generator.py: unchanged copies of the originals.
- benchmark_v1/: 54 NPZ + 54 MAT files, manifest.csv, benchmark_config.json.
- BENCHMARK_VALIDATION_REPORT.md and PHASE1_AUDIT.md: final and pre-generation evidence.
- VALIDATION_SUMMARY.csv: 54 per-case results, objectives and both-format hashes.
- VALIDATION_RESULTS.json: machine-readable aggregate results/environment.
- SOURCE_INTEGRITY.json: hashes confirming the 112 original files were preserved.
- validate_benchmark.py: independent validator; requirements-validated.txt: tested dependencies.

The separate temporary regeneration directory is not needed for distribution. No MATLAB process was run; MAT consistency was checked by scipy.io.loadmat. All copies and originals were hash-verified.
