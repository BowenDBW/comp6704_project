# Proposed Comparison Protocol v1

日期：2026-10-16。状态：**proposed，先用于开发与接口联调，正式实验前由全组确认**。本协议不改变 Generator、LP 或 benchmark 参数，不表示四种算法已经实现或验证。

## 1. 数据与共同问题

四种方法均解相同的 min c_hat.T@z，A@z=b，z>=0。问题背景、参数物理意义与符号见 SHARED_PROBLEM_SPECIFICATION.md；读取方式见 START_HERE.md。使用分发包的 27 base + 27 perturbed，按 case_id 对齐结果。内部可采用不同数据结构或数值缩放，但必须返回原始变量顺序，并在原始 A,b,c_hat 上评估。不得删除/添加物理约束或更改成本。

## 2. 精度与停止规则

均使用 float64/double。下面的 norm_inf 表示最大绝对分量。返回值及统计量必须有限，否则记录 numerical_failure。

| 指标 | 定义 | 暂定阈值 |
|---|---|---|
| primal_residual_inf | norm_inf(Az−b) | ≤1e-7 |
| min_z | min(z) | ≥−1e-8 |
| dual_residual_inf | norm_inf(Aᵀy+r−c_hat) | ≤1e-7 |
| min_dual_slack | min(r) | ≥−1e-7 |
| relative_duality_gap | abs(c_hatᵀz−bᵀy)/max(1,abs(c_hatᵀz),abs(bᵀy)) | ≤1e-7 |
| relative_objective_error | abs(f−f_ref)/max(1,abs(f_ref)) | ≤1e-7，求解后检查 |

共同的 optimal 声明需要前五项满足；reference objective 不进入 solver，不作为停止条件。第六项是与 HiGHS 的独立结果核对，不能单靠 objective 接近判断成功。暂不统一内部 pivot、rank、line-search 和 regularization tolerance，但必须记录其设置，不得把内部容差替代上述共同验收指标。

Simplex 方法：对自己的最终 basis B 解 Bᵀy=c_B，并计算 r=c_hat−Aᵀy，检查 reduced costs、primal feasibility 及共同 gap。Tableau 中若保存了等价 dual 信息也可恢复 y。IPM：使用自身 y,r，检查 dual equation、非负性和 gap；可以额外记录平均互补量 μ=zᵀr/V，但不能仅凭 μ 小就结束。

小的负数应保留用于 min_z/min_dual_slack 统计，不在评估前截断为 0。最大迭代数和超时设置由各实现记录，正式计时前统一确认；达到上限要返回 iteration_limit 或 time_limit，不记为 optimal。

## 3. 接口联调与结果字段

第一步统一加载 M03_N05_a030_seed2026_base，核对 A=(23,33)、b 长度 23、c_hat 长度 33，再运行自己的算法。Dual Simplex 后续使用同名 _demand_perturbed 配对案例。未完成 solver 的成员先验证加载和维度即可，不用 HiGHS 冒充自己的结果。

概念接口：result=solve(A,b,c_hat,options,warm_start=None)。语言可不同；A,b,c_hat 是只读共享输入。返回 z 和 y；IPM 同时返回 r，Simplex 可以从 y 计算 r。支持 warm-start 的方法另存 basis/state；basis 包含 L 个有效、互异的原始列索引，传递时说明零基/一基约定。

每次运行至少记录下列字段，可用 result_schema.csv 的表头：

| 字段 | 含义 |
|---|---|
| protocol_version,case_id,algorithm,mode,repeat_id | v1；实例；方法；cold/warm；重复编号 1…10 |
| status | optimal / iteration_limit / time_limit / numerical_failure / infeasible_claim / unbounded_claim |
| objective,primal_residual_inf,min_z | 原始尺度下的 primal 结果 |
| dual_residual_inf,min_dual_slack,relative_duality_gap | 共同 optimality 检查 |
| relative_objective_error | 外部读取 reference 后计算 |
| iterations,phase1_iterations,phase2_iterations | Simplex 统计所有 pivot，包括初始化阶段；IPM 统计 Newton updates；不适用阶段留空 |
| runtime_seconds,base_runtime_seconds | 当前 solve 总时间；warm 模式另报 base solve 时间，cold 留空 |
| environment_id,options_id,message | 环境与设置记录的标识；状态解释 |

z/y/r 和 basis 可存于各算法自己的结果文件，以 case_id、algorithm、mode、repeat_id 对齐，不改动输入文件。尚未可用的字段留空，不能填 0 假装已测。infeasible/unbounded 声明与本包已验证的可行有界实例相冲突，应调查，不能当作有效 benchmark 结论。

## 4. 计时环境与重复实验

开发可用各自机器；正式 runtime comparison 由同一台电脑运行，固定运行时版本、软件依赖及 BLAS/thread 设置。先用同一语言/runtime 作为主要比较环境；若组员使用不同语言，单列结果并说明语言/库影响，不归因于算法本身。现有 Anaconda 版本是数据验证环境，并非已经选定的正式计时机器。

计时使用 monotonic wall-clock：Python perf_counter，MATLAB tic/toc。范围从 solver 接收已加载的输入开始，到返回结果结束，包含 Phase I/dual-feasible 初始化、solver 内部数据拷贝、因子分解、presolve/scaling、pivot/Newton steps 和内部停止检查；排除磁盘读取、绘图、打印、HiGHS 和外部 reference 检查。必须标注任何不同边界。

每个 case/algorithm/mode 先作 1 次不计入统计的预热，再运行 10 次，报告 median runtime，并保留所有原始行与成功率。不得只汇总成功运行而隐去失败；若 10 次中有失败，报告次数并将成功运行时间的统计明确标为条件统计。迭代数不跨不同算法当作等价工作量。固定并记录算法自身 RNG seed（若使用），不把它与实例 seed 混淆。

在 ENVIRONMENT_TEMPLATE.md 中记录 CPU、RAM、OS、语言/runtime、NumPy/SciPy 或 MATLAB 版本、BLAS、实际线程设置和算法 options。正式运行的机器/线程数、资源上限与执行顺序尚待小组选择；不要编造已固定配置。

## 5. Cold-start 与 warm-start

Cold-start：每个实例和每次计时重复都从各算法自己的合法初始化开始，不复用其他实例或前一次重复的 optimal state。初始化策略允许不同，但必须说明且计入时间。

Warm-start 为单独的 RHS reoptimization 实验：用自己实现的算法得到 base 的 optimal basis，固定 A,c_hat，仅替换为配对 perturbed 的 b。Dual Simplex 复用该 basis；重新计算 B^{-1}b、必要的 basis 校验/因子分解等准备工作均计入 warm runtime。若复用 base 因子分解，必须记录这一复用范围；不允许借用 HiGHS 的 optimum/basis 初始化。

每次 warm 重复从同一个已保存的 base optimal state 的独立副本开始；计入该副本创建/恢复及后续更新成本，不能从上一次已求好的 perturbed optimum 开始。base 求解时间不计入 reoptimization 时间，但另报 base_runtime_seconds；如讨论一次完整使用过程，同时给出 base+warm 总时间。

主要配对比较为同一 perturbed case 的 Dual Simplex cold vs warm。四方法的 cold-start 比较另列；不要把 Dual warm 与其他方法 cold 混成无条件排名。若其他算法也实现 warm-start，单列其 state 与初始化协议。若旧 basis 已 primal feasible，零 pivot 是有效结果，应如实记录；不能强制制造 pivot。

## 6. Reference 与协议冻结

在 VALIDATION_SUMMARY.csv 按 case_id 找 highs_objective（完整 float 精度）；原 manifest 的 objective 仅 12 位有效数字。数据验证 PASS 不等于自写 solver 已 PASS。

请各成员确认：共同模型与排序、实例集合、上述公式与阈值、输出字段、计时范围及 warm-start 规则。正式实验前补齐机器/线程/资源上限/算法设置后记录确认日期并冻结版本。此前可按本 proposed v1 联调；后续修改应全组同步，不静默改口径。
