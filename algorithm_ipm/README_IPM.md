# COMP6704 — Primal-Dual Interior-Point Method(第 4 算法)交付说明

日期:2026-10-19。作者:IPM 负责人。对应共享包:`../COMP6704_Benchmark_v1_validated`(proposed v1)。

## 1. 一句话结论

**54/54 实例全部 optimal(540/540 次运行),六项协议验收指标全部满足,最差相对目标误差 2.35e-10(阈值 1e-7)。** 可以直接拿去和其他三位组员的结果按 case_id 对齐比较。

## 2. 文件清单

| 文件 | 内容 |
|---|---|
| `ipm_solver.py` | 求解器核心:Mehrotra 预测-校正原对偶内点法,仅依赖 NumPy,不调用任何外部 LP solver |
| `test_ipm.py` | 19 项自检(已知最优解、随机 LP 对拍 HiGHS、row-major 恢复、确定性),全部 PASS |
| `run_calibration.py` | 协议第 3 节第一步联调:`M03_N05_a030_seed2026_base`,维度核对 + 全指标,**PASS** |
| `run_benchmark.py` | 全量实验:54 实例 ×(1 次预热 + 10 次计时),按 `result_schema.csv` 表头输出 |
| `results/ipm_results_v1.csv` | 540 行原始结果(每行一个 run,字段与协议表头一致) |
| `results/case_summary.csv` | 每实例聚合:median runtime、median iterations、各指标最差值 |
| `results/ipm_results_summary.json` | 机器可读汇总 + 环境快照 + 阈值核对 |
| `results/environment_record.json` | CPU/内存/系统/Python/NumPy/SciPy/BLAS/线程设置(对应 ENVIRONMENT_TEMPLATE) |
| `results/solutions/*.npz` | 每实例第 1 次计时运行的 z, y, r(case_id 索引,协议允许的自存格式) |
| `results/history/*.csv` | 3 个代表实例的逐迭代收敛历史(mu、残差、步长、sigma) |

## 3. 实现要点(协议第 2 节要求记录的内部设置)

- **算法**:Mehrotra (1992) predictor-corrector。每次迭代组装正规方程 `(A W A^T) Δy = rhs`,`W = diag(z/r)`,一次分解、预测与校正共用;Cholesky 失败时按 `(0, 1e-12, 1e-10, 1e-8, 1e-6)` 逐级加对角正则;对未正则化矩阵做 1 次迭代精化。
- **初始化**(计入计时,协议第 4 节):最小二乘启发式——`y0 = argmin‖Aᵀy−c‖`,`r0 = c−Aᵀy0` 平移至严格正;`z0 = argmin‖Az−b‖` 同样平移。平移量 `max(0,−min(v))+1`,保证所有分量 ≥1。
- **分数到边界**:自适应 `τ = min(0.9995, max(0.995, 1−μ))`,原始/对偶步长分开计算。
- **内部停止条件**(比协议验收阈值严):`‖Az−b‖∞ ≤ 1e-9`(绝对)、`‖Aᵀy+r−ĉ‖∞ ≤ 1e-9`(绝对)、相对对偶间隙 ≤ 1e-9。μ 仅作诊断记录,**不单独作为停止条件**(协议明令禁止)。
- **失败状态**:只返回 `optimal / iteration_limit / numerical_failure`;数值异常时回退到历史最优点(残差最小者),绝不返回垃圾解。
- **确定性**:求解器无 RNG,同输入逐位一致(测试 [4] 验证)。
- **float64** 全程(协议要求);min_z / min_dual_slack 保留符号,不截断。

## 4. 结果摘要(本机,OPENBLAS_NUM_THREADS=1)

| 规模 | 实例类型 | median runtime 范围 | 迭代数范围 | 最差相对目标误差 |
|---|---|---|---|---|
| (3,5) | base ×9 | 0.67–0.87 ms | 8–9 | 1.1e-12 |
| (3,5) | perturbed ×9 | 0.59–0.74 ms | 7–8 | 2.4e-10 |
| (5,10) | base ×9 | 1.50–1.97 ms | 8–11 | 1.1e-10 |
| (5,10) | perturbed ×9 | 1.44–2.29 ms | 8–10 | 2.2e-10 |
| (10,20) | base ×9 | 18.67–25.83 ms | 9–11 | 2.0e-10 |
| (10,20) | perturbed ×9 | 19.97–25.41 ms | 9–11 | 2.3e-10 |

全实验最差值(540 次运行):

- primal residual ≤ 6.5e-10(阈值 1e-7)
- min_z ≥ 2.4e-14(阈值 −1e-8;IPM 路径保持严格正)
- dual residual ≤ 8.7e-12(阈值 1e-7)
- min_dual_slack ≥ 9.1e-14(阈值 −1e-7)
- relative duality gap ≤ 8.1e-10(阈值 1e-7)
- relative objective error ≤ 2.4e-10(阈值 1e-7)

## 5. 环境记录(正式计时前需全组冻结)

- Windows 11 Pro 26300,AMD Ryzen(24 逻辑核),RAM 46.9 GB
- Python 3.13.5(Anaconda),NumPy 2.5.1,SciPy 1.18.0,OpenBLAS 0.3.33
- **BLAS 线程固定为 1**(`run_benchmark.py` 在导入 NumPy 前设置并写入环境记录)。原因:正规方程矩阵 ≤230×230,OpenBLAS 默认 24 线程在小矩阵上的同步开销使最大实例慢约 30 倍(实测 774 ms → 25.8 ms)。**其他人比较 runtime 时必须使用相同线程设置,否则计时不可比。**
- 环境 id:`ENV_Windows_PY313_NP251`(写在每个结果行的 environment_id 字段)。

## 6. 协议合规说明

- 计时范围:`perf_counter` 只包住 `solve_ipm()` 调用(含初始化),文件加载、指标计算、参考值核对在计时区外。冷启动,每次运行独立,无状态复用。
- HiGHS 参考值仅在求解后从 `VALIDATION_SUMMARY.csv` 读取用于 `relative_objective_error`;`test_ipm.py` 中的 `linprog` 仅用于开发期对拍,不在基准管线里。
- perturbed 实例以 **cold** 模式一并跑通(参考值已提供);成对 warm-start reoptimization 实验归 Dual Simplex 负责人,本表不与其 cold 结果混排。
- `phase1_iterations / phase2_iterations / base_runtime_seconds` 按 IPM/cold 的适用性留空,未填 0。

## 7. 复现

```bash
cd algorithm_ipm
python -X utf8 test_ipm.py          # 19 项自检
python -X utf8 run_calibration.py   # 协议第一步联调,PASS
python -X utf8 run_benchmark.py     # 全量 54 实例,约 40 秒
```

## 8. 待小组确认的事项

1. 协议仍是 proposed v1;本结果基于 `PROPOSED_COMPARISON_PROTOCOL.md` 中暂定阈值(1e-7 系列)。全组冻结后如有变化只需重跑 `run_benchmark.py`。
2. 正式 runtime 比较需在同一台机器、相同 BLAS 线程设置下进行;本机数据可作为候选,但需按协议以 ENVIRONMENT_TEMPLATE.md 正式登记。
3. 给报告用的观察(可写进小组报告):IPM 迭代数对规模几乎不敏感(7–11 次),主要成本是每次迭代的正规方程组装与分解;规模 (3,5)→(10,20) 时间约增 30 倍,迭代数基本不变——这正是内点法与单纯形法的典型差异点,适合与 Simplex 组员的迭代/pivot 数对比。
