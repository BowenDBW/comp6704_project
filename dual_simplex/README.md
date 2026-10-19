# COMP6704 Dual Simplex 实现

独立实现目录；共享 Generator、benchmark specification、数据及分发 ZIP 均保持不变。算法仅使用 NumPy 和 SciPy 的线性代数例程，`dual_simplex.py` 不调用 HiGHS/linprog 或其他 LP solver。HiGHS 仅在测试中充当外部 oracle；正式实例的 reference 从共享 CSV 在求解后读取。

## 1. 接口与运行

Python 3.11.15、NumPy 1.26.4、SciPy 1.11.4；runner 还需要 threadpoolctl。此实现当前面向共享 capacitated transportation LP，未宣称是通用 LP 软件。

```python
import numpy as np
from dual_simplex import solve, Options

with np.load('PATH_TO_SHARED_PACKAGE/benchmark_v1/instances/M03_N05_a030_seed2026_base.npz') as f:
    A, b, c_hat = f['A'], f['b'], f['c_hat']
base = solve(A, b, c_hat, Options())
assert base['status'] == 'optimal'

with np.load('PATH_TO_SHARED_PACKAGE/benchmark_v1/instances/M03_N05_a030_seed2026_demand_perturbed.npz') as f:
    b_new = f['b']
cold = solve(A, b_new, c_hat)
warm = solve(A, b_new, c_hat, warm_start={'basis': base['basis']})
```

`solve` 内部复制输入和 basis，返回 z、y、r、basis、迭代历史、协议指标、status 和 runtime_seconds。basis 是 **零基**原始列索引。输入矩阵必须采用共享排序；cold-start 会核对其结构与 slack 零成本。warm-start 接收一个有效且 dual-feasible 的 basis，不自动修复不满足 dual feasibility 的 basis。非法形状、非有限输入或非法索引抛 ValueError；运行中的数值问题返回 numerical_failure。

默认迭代上限 10000 pivots，时间上限 120 秒，在迭代边界检查；它们是本实现的本地验证设置，尚非全组批准的资源上限。共同精度阈值与 proposed protocol 一致，另用 pivot_tol=1e-12、internal_dual_tol=1e-9。不同机器可有不同微小浮点路径。

从本目录运行，`--benchmark` 指向含 START_HERE.md 的完整共享包目录，`--output` 必须是尚不存在的新目录：

```text
python -m unittest -v test_dual_simplex
python run_validation.py --benchmark ../shared --output ../runs/dual_small --small-only
python run_validation.py --benchmark ../shared --output ../runs/dual_all
```

先在仓库根目录安装 requirements.txt 所列依赖。完整验证包含 54 个实例 cold-start 和 27 个 perturbed warm-start；每个 case/mode 先预热 1 次、独立运行 10 次。所有线程池在 runner 内限制为 1 并记录实际设置。数据加载、reference 对照、文件保存不计入 solver runtime。

## 2. 算法说明：合法的 cold-start

设每个客户 j 的最低成本仓库为 i*(j)=argmin_i c_ij，成本相同时选择最小仓库索引。初始 basis 按顺序选：全部 M 个 p 列、N 个 x_(i*(j),j) 列、全部 MN 个 q 列。列数正好 L=M+N+MN。需求行在所选 x 列中为单位阵，再由 p/q 单位列消去供给/容量行，因此 B 非奇异。

该 basis 解满足所选运输量为 d_j，其余运输量为 0，p_i=s_i−sum_(j:i*(j)=i)d_j，q_ij=u_ij−x_ij。p 或 q 可以为负，正是 Dual Simplex 要消除的 primal infeasibility。

解 Bᵀy=c_B 得到供给/容量行的 y 为 0，需求行的 y_j=min_i c_ij。于是所有运输变量 reduced cost 为 c_ij−min_i c_ij≥0，slack reduced costs 为 0。初始 basis 因而 dual feasible。此构造不使用 optimal solution、reference objective、Generator witness 或 HiGHS basis，也不需要辅助 Phase I。初始化工作计入时间；phase1_iterations=0 表示没有 Phase I pivots，并不意味着初始化无成本。

## 3. Dual pivot

每轮重新 LU 分解 B，求 x_B=B^{-1}b、y=B^{-T}c_B 和 r=c_hat−Aᵀy。非基变量设为 0；负的 basic 数值不截断。

若共同 primal/dual/gap 条件通过，返回 optimal。否则选所有 x_B<-1e-8 中**原始列索引最小**的 basic variable 离基，设其 basis row 为 h。用 Bᵀv=e_h 求 v，再计算 a_bar=vᵀA。只考虑 nonbasic 且 a_bar_j<-1e-12 的列；选择 r_j/(-a_bar_j) 最小者入基，精确相同时选择最小列索引。这是 dual Bland 风格的选择规则；不宣称浮点容差下自动获得精确算术的所有终止保证。

内部允许至多 1e-9 的 reduced-cost 负舍入误差；仅在 ratio test 中将该微小负值按 0 处理。返回的 r 和验证统计均保留实际数值。下一轮重新核验 dual feasibility，漂移超限则失败，不静默宣称 optimal。

若没有可用 pivot，返回 numerical_failure 并解释数值阈值问题；本版本未实现通用 infeasibility/unboundedness certificate。共享数据已验证为可行有界；这一限制不影响已通过的实例。每轮重新分解有利于清晰验证，但不是高性能 basis-update 实现。

## 4. Warm-start 与计时

warm-start 仅复用 base optimal basis 的列索引，不复用 LU。每次调用在计时范围内复制 basis、核对索引、重新分解并计算 B^{-1}b_new。A、c_hat 不变时 reduced costs 不依赖 b，原 dual feasibility 得以保留。每次重复从相同 base basis 开始，不从前次 perturbed optimum 开始。

base_runtime_seconds 是产生该保存 basis 的那次 base solve 时间（runner 的第 10 次 base cold 运行），不计入 warm runtime；base 所有运行和 median 另外保存。讨论完整使用成本时可计算这次 base time + warm time。零 pivot warm 解允许出现，不强制扰动旧 basis。

计时包含验证输入、复制、构造 basis、每轮 LU、指标与 history 记录和内部停止检查；排除外部对照、保存及绘图。时间是本地实现验证数据，尚不是全组正式 runtime ranking。

## 5. 交付与结果

- dual_simplex.py：算法及 Options。
- test_dual_simplex.py：手算案例、退化/tie、随机 oracle 对照、输入不变性及失败路径测试。
- run_validation.py：协议化 runner，不在 solver 内使用 reference。
- results_small/：统一小实例，30 timed runs + 3 warmups。
- results_all/：全量 810 timed runs + 81 warmups。
- 各 results 目录：runs.csv 为逐次原始结果；summary.csv 为每 case/mode 的 median、成功率及 pivot 数；solutions/ 保存每次 z/y/r/basis 和 JSON history；environment.json、validation.json、input_sha256.json 保存环境、统计和共享包校验。
- IMPLEMENTATION_VALIDATION_REPORT.md：实际完成的验证及限制。

runs.csv 包含共享 result_schema.csv 的全部字段，另加 validation_pass 与 initial_min_basic。无效结果不伪装为零值。报告 objective/reference accuracy 与 primal/dual certificates 是独立检查。残差与 gap 全部在原始 A,b,c_hat 上计算。
