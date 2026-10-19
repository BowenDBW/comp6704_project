# COMP6704：Capacitated Transportation LP 算法比较

本仓库提供统一的运输 LP、已验证 benchmark，以及 Dual Simplex 和 Tableau Simplex 实现，供小组在相同输入及精度要求下比较四种算法。

| 方法 | 本仓库当前状态 |
|---|---|
| Tableau Simplex | 已实现 Two-Phase Full-Tableau Simplex；54/54 实例、540/540 次计时通过 |
| Revised Simplex | 等待负责人加入代码及结果 |
| Dual Simplex | 已实现并验证 cold-start 与 demand-perturbed warm-start |
| Primal-Dual Interior-Point Method | 等待负责人加入代码及结果 |

尚未提供全部四种方法在同一环境下的实验结果，不能据此给出四算法性能排名。共享 protocol 的原始文件保留 proposed 标记，小组确认后应记录确认日期和正式环境。

## 目录

```text
shared/                   完整共享模型、Generator、实例及 reference objectives
dual_simplex/             算法、测试、runner、说明及原有全部实验结果
tableau_simplex/          两阶段全表单纯形算法、测试、runner 及完整验证结果
validation/               本上传包的小实例复跑结果与单元测试记录
requirements.txt          根目录依赖安装入口
runs/                     用户自行复跑的输出，Git 默认忽略
```

`dual_simplex/results_small/` 和 `dual_simplex/results_all/` 保留原验证结果及逐次解/轨迹；`validation/small/` 是本上传目录中新执行的小实例验证。不同批次的 runtime 可能不同，不能混作同一批测量。

## 快速运行

使用 Python 3.11；原验证环境为 Python 3.11.15、NumPy 1.26.4、SciPy 1.11.4。在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s dual_simplex -p "test_dual_simplex.py" -v
python -m unittest discover -s tableau_simplex -p "test_tableau_simplex.py" -v
python dual_simplex/run_validation.py --benchmark shared --output runs/dual_small --small-only
python tableau_simplex/run_benchmark.py --benchmark shared --output runs/tableau_small --small-only
```

小实例包含 base cold、perturbed cold、perturbed warm 三种 case/mode，各 1 次 warmup 与 10 次计时，总计 30 次计时和 3 次 warmup。成功时输出 `all_pass: true`。

运行完整 benchmark：

```bash
python dual_simplex/run_validation.py --benchmark shared --output runs/dual_all
python tableau_simplex/run_benchmark.py --benchmark shared --output runs/tableau_all
```

完整验证包含 54 个 cold-start 与 27 个 warm-start case/mode。输出目录必须尚不存在；重复运行时换成 `runs/dual_small_02` 等新名字，避免覆盖既有记录。

runner 以线程池限制为 1 的方式运行并记录环境。计时包含输入复制、初始化、LU、pivot 和 trace 收集，排除磁盘 I/O 与外部 reference 检查。

## 共同优化问题

仓库 i 向客户 j 运送连续货量 x_ij，最小化总运输成本：

```text
minimize    sum_ij c_ij * x_ij
subject to  sum_j x_ij <= s_i       每个仓库的供给上限
            sum_i x_ij  = d_j       每个客户的需求
            0 <= x_ij <= u_ij       每条路线的容量
```

加入未用供给 p_i 和未用路线容量 q_ij 后，所有算法读取并求解：

```text
min c_hat.T @ z,   A @ z = b,   z >= 0
z = [row-major X; p; row-major Q]
```

约束按供给、需求、容量排列。容量已经编码在 A 和 b 中，无需额外添加 bounds。矩阵维度为 `(M+N+MN, 2MN+M)`。

详细定义见 [共同模型](shared/SHARED_PROBLEM_SPECIFICATION.md)、[共享入口](shared/START_HERE.md) 和 [比较协议](shared/PROPOSED_COMPARISON_PROTOCOL.md)。

## 加载一个实例并运行 Dual Simplex

在仓库根目录运行以下 Python 代码：

```python
from pathlib import Path
import csv
import numpy as np
from dual_simplex.dual_simplex import solve

shared = Path("shared")
case_id = "M03_N05_a030_seed2026_base"
with np.load(shared / "benchmark_v1/instances" / f"{case_id}.npz", allow_pickle=False) as data:
    A, b, c_hat = data["A"], data["b"], data["c_hat"]

base = solve(A, b, c_hat)
assert base["status"] == "optimal"
X = base["z"][:15].reshape(3, 5)  # 本例为 3 个仓库、5 个客户
print("objective:", base["objective"])

with (shared / "VALIDATION_SUMMARY.csv").open(encoding="utf-8-sig", newline="") as f:
    references = {row["case_id"]: float(row["highs_objective"]) for row in csv.DictReader(f)}
f_ref = references[case_id]
relative_error = abs(base["objective"] - f_ref) / max(1.0, abs(f_ref))
print("reference:", f_ref, "relative error:", relative_error)

perturbed_id = "M03_N05_a030_seed2026_demand_perturbed"
with np.load(shared / "benchmark_v1/instances" / f"{perturbed_id}.npz", allow_pickle=False) as data:
    assert np.array_equal(A, data["A"])
    assert np.array_equal(c_hat, data["c_hat"])
    b_new = data["b"]

cold = solve(A, b_new, c_hat)
warm = solve(A, b_new, c_hat, warm_start={"basis": base["basis"]})
assert cold["status"] == warm["status"] == "optimal"
print("perturbed cold / warm pivots:", cold["iterations"], warm["iterations"])
```

`solve` 返回 `z, y, r, basis, history, status` 及精度、时间指标。basis 使用零基原始列索引。warm-start 只复制 basis indices，不复用 LU；每次重复从同一个 base optimal basis 开始。

## Benchmark 与 reference

- 规模 `(3,5), (5,10), (10,20)`；alpha 为 `0.3, 0.6, 0.9`；seed 为 `2026, 2027, 2028`。
- 27 个 base 加 27 个 demand-perturbed 实例，共 54 个 LP；每个同时提供 NPZ 和 MAT。
- 同一 base/perturbed pair 的 A 和 c_hat 完全相同，仅需求 RHS 改变。
- 完整精度的 reference objective 位于 `shared/VALIDATION_SUMMARY.csv` 的 `highs_objective` 列，通过 `case_id` 对应。
- reference 只在 solver 返回后用于验证，不能传入学生求解器用于初始化或优化。

可核对的小实例结果：base cold 8 pivots，objective 约 `776.202970694746`；perturbed cold 8 pivots、warm 2 pivots，objective 均约 `775.088961948288`。浮点路径可能随环境变化，正式判断以容差与 reference 检查为准。

## 方法与已有结果的边界

`dual_simplex.py` 自行执行 basis 构造、Dual Simplex pivot 及停止判断，只使用 NumPy/SciPy 的线性代数例程，不调用外部 LP solver。测试中的 HiGHS 用作独立 oracle。

本实现 cold-start 针对共享运输矩阵和零 slack 成本，不是通用 LP 初始化软件。出现数值阈值下无候选等问题时返回 `numerical_failure`，未实现通用 infeasibility/unboundedness certificate。

已有全量验证记录为 54 cold + 27 warm case/mode，810 次计时及 81 次 warmup 全部通过。详见 [算法验证报告](dual_simplex/IMPLEMENTATION_VALIDATION_REPORT.md)。这些 runtime 来自本地环境，正式四方法时间比较仍需共同环境。

Tableau Simplex 已完成 54 个 cold-start 实例、540 次计时及 54 次 warmup，全部通过精度检查。详见 [Tableau 验证报告](tableau_simplex/IMPLEMENTATION_VALIDATION_REPORT.md)。这些 runtime 同样仅作为本地验证结果。

## 小组协作与上传

共享实例与 Generator 保持一致；其他算法使用独立目录，例如 `revised_simplex/` 和 `primal_dual_ipm/`，避免覆盖 `shared/`。

`.gitignore` 忽略缓存、虚拟环境和 `runs/`；已有验证结果保留追踪。浏览器上传时也不要选择本机复跑产生的 `runs/` 或缓存目录，浏览器不会替你按本地忽略规则筛选。无需上传分发 ZIP 本身。

仓库已有课程资料、文献和报告草稿，保留在 `docs/`；后续报告和 PPT 也可在该目录协作更新。当前没有替三位组员添加虚构实现或结果，也没有代替小组选择开源许可证。仓库发布前按课程要求及小组约定确认可见性，并把真实 repository URL 写入最终 report。
