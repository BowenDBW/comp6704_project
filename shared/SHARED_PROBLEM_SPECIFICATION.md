# 运输问题定义与符号说明

## 1. 共同研究问题

项目题目：Comparative Analysis of LP Algorithms for Capacitated Transportation Optimization。

考虑 M 个仓库向 N 个客户运输一种可连续分配的货物。每个仓库有有限可用供给，每个客户的需求必须恰好满足，每条仓库到客户的运输路线有容量限制。给定单位运输成本，选择各路线运输量，使总成本最小。

四种算法求解完全相同的 capacitated transportation LP：Tableau Simplex、Revised Simplex、Dual Simplex、Primal-Dual Interior-Point Method。研究目标是比较它们的求解精度、计算时间、迭代行为，以及 RHS 变化后的 reoptimization 表现；不预设某种算法必然最好。

模型是单商品、单阶段、确定性、连续变量的合成 benchmark。每对仓库/客户均有一条路线。供给是上限，允许不用完；需求是等式，不允许缺货或超额交付。不存在整数决策、固定启用费用或额外运输时间约束。成本和货量使用一致的抽象单位，不将合成数据解释为真实企业数据。

## 2. 符号、代码名称与物理意义

| 符号 | 代码名称 | 维度、范围 | 定义与意义 |
|---|---|---|---|
| M, N | M,N；函数内 m,n | 正整数 | 仓库数量、客户数量 |
| i, j | i,j | 代码从 0 开始 | 仓库索引、客户索引 |
| x_ij | X[i,j]；Generator 内 witness_x 为可行见证 | M×N，非负 | 仓库 i 运往客户 j 的货量，真正的决策变量 |
| c_ij | costs[i,j] | M×N，正数 | 单位货量的运输成本；不是决策变量 |
| s_i | supplies[i] | M，正数 | 仓库 i 可用供给上限 |
| d_j | demands[j] | N，正数 | 客户 j 必须被满足的需求 |
| u_ij | capacities[i,j] | M×N，正数 | 路线 i→j 的货量上限 |
| p_i | supply_slack[i] | M，非负 | 未使用供给：s_i−sum_j x_ij |
| q_ij | capacity_slack[i,j] | M×N，非负 | 未使用路线容量：u_ij−x_ij |
| z | z | 2MN+M | 堆叠后的标准形式决策向量，含 x,p,q |
| A | A | (M+N+MN)×(2MN+M) | 等式约束系数矩阵 |
| b | b | M+N+MN | 按顺序堆叠供给、需求、容量的 RHS |
| ĉ | c_hat | 2MN+M | x 的运输成本与 slack 的零成本 |
| f | objective | 标量 | 总运输成本 ĉᵀz |
| f_ref | highs_objective | 标量 | HiGHS 数值 reference optimum，仅用于外部验证 |
| α | alpha | 0.3,0.6,0.9 | 容量紧张程度参数；越大，容量越紧，并非实际利用率 |
| w_ij | weights[i,j] | M×N，正数 | 分配需求的随机权重，无运输决策含义 |
| x_ref | witness_x | M×N | 构造实例的可行见证，不是 optimal solution，也不保证是 BFS |
| δ_j | deltas[j] | N，[-0.05,0.05) | 相对需求扰动；d'_j=d_j(1+δ_j) |
| seed | seed | 2026,2027,2028 | RNG 初始化参数，不是实例全部身份信息 |
| y | y | M+N+MN，无符号限制 | 等式约束的 dual multiplier；按 supply/demand/capacity 分段 |
| r | dual_slack | 2MN+M，dual feasible 时非负 | r=ĉ−Aᵀy；也对应 reduced costs 的标准形式表达 |
| μ | mu | 非负标量 | IPM 中可用 zᵀr/(2MN+M) 表示平均互补量 |

为避免冲突，文档统一用 s/supplies 表示物理供给，用 r/dual_slack 表示 dual slack；不要同时把两者写成 s。运输容量 slack q 与 dual slack r 不是同一个量。NPZ/MAT 仅含 A,b,c_hat；其他名称用于解释和算法内部。

## 3. 数学模型与 standard form

\[
\min_{x}\sum_{i=0}^{M-1}\sum_{j=0}^{N-1}c_{ij}x_{ij},\qquad
\sum_jx_{ij}\le s_i,\quad \sum_ix_{ij}=d_j,\quad 0\le x_{ij}\le u_{ij}.
\]

引入 p_i、q_ij 后：sum_j x_ij+p_i=s_i，sum_i x_ij=d_j，x_ij+q_ij=u_ij，且 x,p,q≥0。统一求解 min ĉᵀz，Az=b，z≥0。不能再把 p、q 当作额外收费或额外决策限制。

令 K=MN，L=M+N+MN，V=2MN+M。变量顺序为 z=[vec_C(X);p;vec_C(Q)]，ĉ=[vec_C(costs);0_M;0_K]，b=[supplies;demands;vec_C(capacities)]。vec_C 为 row-major，即同一个仓库下先遍历所有客户。

Python 零基索引：x_ij 列 i*N+j，p_i 列 K+i，q_ij 列 K+M+i*N+j。行顺序：供给 [0,M)，需求 [M,M+N)，容量 [M+N,L)。MATLAB 对应索引加 1。

| M,N | A shape | b length | c_hat/z length |
|---|---|---|---|
|3,5|23×33|23|33|
|5,10|65×105|65|105|
|10,20|230×410|230|410|

从已加载数据提取物理参数：Python 中 supplies=b[:M]，demands=b[M:M+N]，capacities=b[M+N:].reshape(M,N)，costs=c_hat[:K].reshape(M,N)。MATLAB 中 capacities=reshape(b(M+N+1:end),N,M).'，costs=reshape(c_hat(1:K),N,M).'。同理从 z 恢复 X。不要用 MATLAB 的 X(:) 直接替代 row-major vec_C。

## 4. 实例构造与实验边界

沿用统一生成器的 RNG、分布和参数：3 个规模×3 个 alpha×3 个 seed=27 个 base，再配对 27 个 demand-perturbed。s=1.2*row_sum(x_ref)，u=(3−2α)*x_ref。增加 α 仅降低 capacity RHS；扰动仅改变 demand RHS。可行性证明和 54/54 数值验证见 BENCHMARK_VALIDATION_REPORT.md。

正的 x_ref 和 slack 提供可行见证，不强制作为四种算法相同的起点；Simplex 所需 basis 与 IPM 所需内部初值不同。Perturbed 实例仍可行，但并不保证旧 optimal basis 一定失去 primal feasibility 或一定需要 pivot。
