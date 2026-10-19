# Dual Simplex 代码理解与小组汇报说明

这份说明对应 `COMP6704_Dual_Simplex/dual_simplex.py` 的实际实现，以及 `results_small` 中保存的统一小实例结果。核心思路是：**先构造 dual feasible、可能 primal infeasible 的 basis，再通过 dual simplex pivots 消除负的 basic variables；需求变化时，复用已有 optimal basis，重新恢复 primal feasibility。**

算法自己完成 basis 构造、pivot 和停止判断。SciPy 在 solver 内只提供 LU 线性方程求解；HiGHS 的 reference objective 在 solver 返回之后才用于外部检查。

## 1 从共同运输问题到代码输入

我们有 M 个仓库、N 个客户。x_ij 表示仓库 i 向客户 j 的运输量，c_ij 是单位运输成本，s_i 是供给上限，d_j 是需求，u_ij 是路线容量。模型为

\[
\begin{aligned}
\min_x\quad &\sum_{i,j}c_{ij}x_{ij}\\
\text{s.t.}\quad &\sum_jx_{ij}\le s_i,\\
&\sum_ix_{ij}=d_j,\\
&0\le x_{ij}\le u_{ij}.
\end{aligned}
\]

引入未使用供给 p_i 和未使用路线容量 q_ij：

\[
\sum_jx_{ij}+p_i=s_i,\qquad x_{ij}+q_{ij}=u_{ij}.
\]

所有变量 x,p,q 都要求非负，得到共享标准形式

\[
\min_{z\ge0}\ \hat c^Tz,\qquad Az=b.
\]

代码中 `c` 是输入 `c_hat` 的局部副本，不是单独的 M×N 成本矩阵。变量向量为

\[
z=[\operatorname{vec}_C(X);p;\operatorname{vec}_C(Q)],\qquad
\hat c=[\operatorname{vec}_C(C);0;0].
\]

`vec_C` 表示 row-major：先遍历仓库 0 的所有客户，再遍历仓库 1，以此类推。约束行依次是 supply、demand、capacity。

下文所有数组索引均从 0 开始。定义 K=MN、L=M+N+MN、V=2MN+M，则 A 有 L 行、V 列。统一小实例 M=3、N=5，所以 K=15、L=23、V=33。

| 内容 | 小实例中的列索引 | 物理意义 |
|---|---|---|
| x_ij | i×5+j，即 0…14 | 运输量 |
| p_i | 15+i，即 15…17 | 未使用仓库供给 |
| q_ij | 18+i×5+j，即 18…32 | 未使用路线容量 |

## 2 Basis 到底是什么

`basis` 是 L 个列索引组成的数组；`B=A[:,basis]` 才是对应的 L×L basis matrix。选择的列必须线性无关，所以 B 必须可逆。

把变量分为 basic variables z_B 和 nonbasic variables z_N，令 z_N=0，并解

\[
Bz_B=b.
\]

就得到该 basis 对应的 basic solution。它自动满足等式 Az=b，但 z_B 可能有负值。只有 z_B≥0 时，才是 basic feasible solution，简称 BFS。

因此，**basic solution 与 basic feasible solution 不能混为一谈**。Dual Simplex 允许从前者出发，但要求对应的 dual solution 已经可行。

标准形式的 dual 是

\[
\max_y\ b^Ty,\qquad A^Ty\le\hat c.
\]

因为 primal 的约束是等式，所以 y 本身没有非负限制。定义

\[
r=\hat c-A^Ty.
\]

`r` 是 dual slack，也是所有列的 reduced costs。Dual feasibility 等价于 r≥0。给定 basis，我们取

\[
B^Ty=\hat c_B.
\]

这样 basic columns 的 reduced costs 为 0。代码使用 `lu_solve(...,trans=1)` 解这个转置方程。

注意：s_i 表示物理供给，r 表示 dual slack，q_ij 表示路线容量 slack。它们不是同一个对象。

## 3 Cold start 如何构造 dual feasible basis

### 3.1 每个客户先选最便宜的仓库

对客户 j，定义

\[
i^*(j)\in\arg\min_i c_{ij}.
\]

若成本相同，`np.argmin` 取最小仓库索引。把以下变量选为 basic variables：

1. 全部 M 个 supply slack p_i；
2. 每个客户对应的一个最便宜运输变量 x_(i*(j),j)，共 N 个；
3. 全部 MN 个 capacity slack q_ij。

总数是 M+N+MN=L，恰好等于约束行数。代码为

```python
cheapest = np.argmin(c[:k].reshape(m,n), axis=0)*n + np.arange(n)
basis = np.r_[np.arange(k,k+m), cheapest, np.arange(k+m,V)].astype(int)
```

`cheapest` 存的是运输变量的**全局列索引**，不是仓库编号。

### 3.2 为什么这些列线性无关

设 demand rows 在 N 个选中运输列上的子矩阵。每个客户恰好选一条路线，所以该子矩阵是 I_N。p 列只作用于 supply rows，形成 I_M；q 列只作用于 capacity rows，形成 I_K。按代码的行列顺序，可写成

\[
B=\begin{bmatrix}
I_M&S_*&0\\
0&I_N&0\\
0&E_*&I_K
\end{bmatrix},
\]

其中 S_* 表示所选路线来自哪个仓库，E_* 把所选路线映射到对应容量行。由中间行块先求运输量，再求 p、q，因此 B 非奇异。选列不是碰运气，也不需要通过 HiGHS 寻找 basis。

### 3.3 初始 basic solution 的物理含义

非基运输变量全为 0；选中路线先承担对应客户的全部需求：

\[
x_{i^*(j),j}=d_j.
\]

其余 basic variables 由等式确定：

\[
p_i=s_i-\sum_{j:i^*(j)=i}d_j,\qquad q_{ij}=u_{ij}-x_{ij}.
\]

直观上，先把每个客户全部交给最便宜的仓库，再用 p、q 表示“剩余量”。若仓库供给或路线容量不足，相应 slack 就会为负。

负 slack 说明当前 basic solution 不能实际执行，并不表示我们改变了物理模型。Dual Simplex 后续正是通过换 basis，把这些负值消除。

### 3.4 为什么初始 basis 一定 dual feasible

将 y 按约束分段为供给乘子 y^S、需求乘子 y^D、容量乘子 y^U。因为所有 p_i 都在 basis 内且成本为 0，由 Bᵀy=c_B 得

\[
y_i^S=0.
\]

同理，所有 q_ij 都在 basis 内，所以

\[
y_{ij}^U=0.
\]

选中的运输列满足

\[
y_{i^*(j)}^S+y_j^D+y_{i^*(j),j}^U=c_{i^*(j),j},
\]

因此 y_j^D=min_i c_ij。任意运输列的 reduced cost 为

\[
r_{ij}=c_{ij}-y_i^S-y_j^D-y_{ij}^U
=c_{ij}-\min_{i'}c_{i'j}\ge0.
\]

p 和 q 的 reduced costs 均为 0。因此全部 r≥0，初始 basis 确实 dual feasible。

这利用了本共享模型的矩阵结构、每个客户至少有路线以及 slack 的零成本；不是任意 LP 都能直接这样初始化。代码的 `transportation_shape` 会核对标准矩阵结构和 slack costs。

### 3.5 是否需要 Phase I

当前构造直接满足 Dual Simplex 起步条件，所以没有额外的辅助 Phase I，返回 `phase1_iterations=0`。但构造 basis、验证结构、LU 分解都需要计算，这些时间包含在 `runtime_seconds` 中。

Generator 的 `x_ref` 是另一个可行见证，当前 cold-start 没有使用它，也不依赖它提供 BFS。

## 4 每轮代码具体做什么

```text
读取并复制 A、b、c_hat
    ↓
cold：构造 cheapest-route basis
warm：复制传入的 base optimal basis
    ↓
构造 B 并进行 LU 分解
    ↓
求 z_B、y、r，计算原始尺度下的验证指标
    ↓
满足全部停止条件 → 返回 optimal
    ↓ 否
检查资源上限与 dual feasibility
    ↓
选 leaving row → dual ratio test 选 entering column
    ↓
替换 basis 中的一列，回到 LU 分解
```

实际代码先检查内部 dual feasibility，再检查共同 optimality 条件，之后检查资源上限。没有显式计算矩阵逆：

```python
B = A[:, basis]
lu = lu_factor(B, check_finite=False)
xb = lu_solve(lu, b, check_finite=False)
y = lu_solve(lu, c[basis], trans=1, check_finite=False)
r = c - A.T @ y
z = np.zeros(V)
z[basis] = xb
```

这是基于线性方程求解的 Dual Simplex 实现。虽然使用了 basis 和 LU，与 Revised Simplex 有相似的计算工具，但**pivot 的可行性保持方向和选择规则是 Dual Simplex**：保持 dual feasibility，恢复 primal feasibility。

每次换基后重新 LU 分解，代码容易核验；没有实现 LU 更新或大型稀疏优化，因此不应把它宣传为高性能工业 solver。

## 5 Pivot 规则及其推导

### 5.1 Leaving variable 不是取最负的变量

代码找到 z_B 中所有小于 −1e-8 的分量，在这些变量中选择**原始列索引最小**的那个离基：

```python
bad = np.flatnonzero(xb < -opt.nonnegative_tol)
leaving = int(bad[np.argmin(basis[bad])])
```

`leaving` 是该变量在 basis 数组中的位置 h；`basis[leaving]` 才是其在原始 A 中的列索引。规则是 dual Bland 风格，主要用于确定性选取与处理退化。不能把汇报写成“每次都选最负的变量”，因为这与代码不符。

### 5.2 为什么只选负的 tableau 系数

对当前 basis，令

\[
\bar b=B^{-1}b,\qquad \bar A=B^{-1}A.
\]

离基行 h 满足

\[
z_{B_h}+\sum_{j\in\mathcal N}\bar a_{hj}z_j=\bar b_h<0.
\]

若将 z_(B_h) 设为 0，只让一个 nonbasic variable z_e 入基，则

\[
z_e=\frac{\bar b_h}{\bar a_{he}}.
\]

因为分子为负，要使新 basic variable z_e≥0，必须有 \bar a_he<0。因此只考虑负系数的非基列。

代码不形成整个 B^{-1}A，而是解 Bᵀv=e_h，再计算 `row=v@A`，得到需要的第 h 行。候选条件是 `row < -1e-12`，用于排除太小的 pivot。

### 5.3 Dual ratio test 为什么保持 dual feasibility

对候选入基列 e，令

\[
t=\frac{r_e}{-\bar a_{he}}\ge0,\qquad y'=y-t v.
\]

则新的 reduced costs 是

\[
r'_j=\hat c_j-a_j^Ty'=r_j+t\bar a_{hj}.
\]

若 \bar a_hj≥0，r'_j 不会降低；若 \bar a_hj<0，要保持 r'_j≥0，必须满足

\[
t\le\frac{r_j}{-\bar a_{hj}}.
\]

因此选择

\[
e\in\arg\min_{j\in\mathcal N:\bar a_{hj}<0}
\frac{r_j}{-\bar a_{hj}}
\]

就能让入基列 reduced cost 变成 0，同时保留其他列的非负 reduced costs。旧的 basic columns 中，离基列的 reduced cost 变成 t，其他 basic columns 仍为 0。

代码对应为

```python
ratios = np.maximum(r[candidates], 0) / (-row[candidates])
entering = int(candidates[np.argmin(ratios)])
basis[leaving] = entering
```

候选索引按升序排列，精确 ratio ties 选择最小列索引。`maximum(...,0)` 只处理内部允许的微小负舍入误差；实际返回的 r 并没有被截断。下一轮会重新求 y、r，若 dual feasibility 漂移超过 1e-9，就返回数值失败。

### 5.4 每次 pivot 是否都会让所有负值变小

不一定。它使新入基变量成为非负，但其他 basic values 也会随换基变化，可能仍有负值，甚至新出现负值。保持的是 dual feasibility，而不是每一步都让所有 primal infeasibility 指标单调下降。

在精确算术下，本次 dual objective 的变化为

\[
b^Ty'-b^Ty=-t\bar b_h\ge0.
\]

所以 dual lower bound 不下降。当前 basic solution 的 objective 也等于这个 dual objective，但在 primal infeasible 时，它不能解释为可执行运输方案的成本。

## 6 小实例的 cold start 和第一次 pivot

实例为 `M03_N05_a030_seed2026_base`。其成本矩阵如下，展示值保留 6 位小数，实际代码使用完整 float64：

| 仓库/客户 | j=0 | j=1 | j=2 | j=3 | j=4 |
|---|---:|---:|---:|---:|---:|
| i=0 |15.426148|14.071723|11.433077|17.300127|12.239378|
| i=1 |18.751785|6.551270|15.493372|10.424119|2.933905|
| i=2 |6.017635|17.067358|13.802648|16.768463|1.995054|

每个客户选出的仓库依次是 [2,1,0,1,2]，运输变量列索引为 [10,6,2,8,14]。完整初始 basis 为

```text
[15,16,17,10,6,2,8,14,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
```

初始解中恰有三个负值：

| 原始列 | 变量 | 值 | 含义 |
|---|---|---:|---|
|20|q[0,2]|−6.362002360|仓库 0 到客户 2 的路线超容量|
|28|q[2,0]|−3.484526067|仓库 2 到客户 0 的路线超容量|
|32|q[2,4]|−1.781137747|仓库 2 到客户 4 的路线超容量|

按最小列索引规则，列 20 离基。该行的两条候选路线为

| 候选列 | 变量 | tableau coefficient | reduced cost | dual ratio |
|---|---|---:|---:|---:|
|7|x[1,2]|−1|4.060295462|4.060295462|
|12|x[2,2]|−1|2.369571060|2.369571060|

列 12 的 ratio 更小，所以 x[2,2] 入基。新入基货量为

\[
\frac{-6.3620023596}{-1}=6.3620023596.
\]

物理上，将客户 2 原本全交给仓库 0 的部分货量转给仓库 2，使仓库 0 的那条路线不再超容量。新 basis 仍可能违反其他供给/容量约束，因此算法继续 pivot。

这一步目标值从 723.894988361 上升为 738.970205039。目标值上升是 dual lower bound 改进，不是“最小化算法越做越差”。初始点含负 slack，不是可行运输方案。

## 7 Base 的完整 8 次 pivot 轨迹

下表来自保存的第 1 次 timed run。第 k 行表示已经完成 k 次 pivot 后的状态，最后一列是下一步操作。

| 已完成 pivots | 当前 objective | min(z) | 下一步 leaving → entering |
|---:|---:|---:|---|
|0|723.894988361|−6.362002360|q[0,2] → x[2,2]|
|1|738.970205039|−3.484526067|p[2] → x[1,4]|
|2|742.096900219|−3.484526067|p[1] → x[0,3]|
|3|755.758227243|−3.484526067|q[2,0] → x[0,0]|
|4|761.311351157|−1.935323341|x[0,3] → p[1]|
|5|771.609653789|−1.935323341|x[1,4] → p[2]|
|6|771.754411167|−1.781137747|q[2,4] → x[1,4]|
|7|773.426635030|−0.437608039|p[1] → x[2,3]|
|8|776.202970695|0|停止，optimal|

这里 slack 也可以入基或离基，运输变量也可以离基。换基是在完整的 z=[x;p;q] 空间中进行，不是永远只在运输变量之间换。

注意第 1 次 pivot 后的下一步是 p[2] 离基。它并不是简单延续“逐条修复最初三条超容量路线”：修复一条路线会改变仓库分配，进而可能产生新的负 supply slack。

## 8 为什么 residual 和 gap 为零仍不能停止

这是理解本例最关键的一点。每个 basic solution 都令 z_N=0，并解 Bz_B=b，所以在数值精度内 Az=b，即使 z_B 有负数，等式 residual 也可以为 0。

同时，由 Bᵀy=c_B，有

\[
\hat c^Tz=\hat c_B^Tz_B=y^TBz_B=y^Tb.
\]

因此，对当前 basic solution，primal-dual objective difference 也可以为 0。上面的初始点正是：residual=0、gap=0，但 min(z)=−6.362…。

**只有 primal feasibility 和 dual feasibility 同时成立，零 gap 才能用于最优性判断。** 这里 primal feasibility 包含 z≥0，而不只是 Az=b。因此代码必须检查 `min_z`。

最终停止使用：

| 指标 | 阈值 |
|---|---|
| norm_inf(Az−b) | ≤1e-7 |
| min(z) | ≥−1e-8 |
| norm_inf(Aᵀy+r−c_hat) | ≤1e-7 |
| min(r) | ≥−1e-7 |
| abs(c_hatᵀz−bᵀy)/max(1,abs(c_hatᵀz),abs(bᵀy)) | ≤1e-7 |

此外，代码每轮先用更紧的 `internal_dual_tol=1e-9` 检查 reduced costs 的负漂移。`r=c_hat−Aᵀy` 是直接计算的，所以 dual equation residual 也主要是数值一致性检查，不能替代 r≥0。

Reference objective error 不属于停止条件，只在返回结果后由 runner 计算。

## 9 Warm start 为什么能够直接复用 basis

Base 求解结束后，保存的是原始列索引 `base['basis']`。需求扰动仅改变 b 的 demand segment，A 和 c_hat 不变。

固定同一个 basis B，则

\[
y=B^{-T}\hat c_B,\qquad r=\hat c-A^Ty
\]

都不依赖 b。所以，base 的 dual-feasible basis 在需求变化后仍 dual feasible。

但是 basic values 会变：

\[
z'_B=B^{-1}b'=z_B+B^{-1}(b'-b).
\]

其中某些分量可能变负。此时恰好又满足 Dual Simplex 的起步条件：dual feasible、primal infeasible。

调用方式是

```python
base = solve(A, b, c_hat)
warm = solve(A, b_new, c_hat,
             warm_start={'basis': base['basis']})
```

当前实现复用的是 **basis 结构**，不是把旧 z 当作新解，也不是只“接着运行上次结果”。`solve` 在计时内复制 basis、检查索引，重新提取 B、LU 分解并计算新 z_B。没有复用 LU factorization，没有借用 HiGHS basis。

若新 z_B 已全部非负，那么这个 basis 对新 LP 已同时 primal/dual feasible，算法可直接以 0 pivot 返回 optimal。全量验证中 27 个 warm cases 有 8 个出现这种情况。

## 10 小实例的 warm start 只需要 2 次 pivot

配对实例是 `M03_N05_a030_seed2026_demand_perturbed`。从 base optimal basis 出发，用新 b 重算 basic values 后，初始 min(z)=−0.684928428，说明旧 basis 不再 primal feasible。

| 已完成 pivots | 当前 objective | min(z) | 下一步 leaving → entering |
|---:|---:|---:|---|
|0|770.971535938|−0.684928428|q[1,1] → x[0,1]|
|1|771.777086677|−0.522020180|x[2,3] → p[1]|
|2|775.088961948|0|停止，optimal|

和从头构造 cheapest-route basis 相比，warm-start 从已有的最优 basis 结构附近开始，本例只需两次换基就恢复可行性。

| 运行方式 | 最终 objective | pivots |
|---|---:|---:|
| Base cold-start |776.202970695|8|
| Perturbed cold-start |775.088961948|8|
| Perturbed warm-start |775.088961948|2|

正确的 cold/warm 对比是后两行，因为它们求解的是**相同 perturbed LP**。不能直接比较 base 与 perturbed 的目标值，来判断哪种算法更好，因为需求已经变化。

Warm-start 目标值不必等于 base 目标值。本例 perturbation 含正负需求变化，新的最优成本较低；这与算法是否 warm-start 无关。

`results_small/summary.csv` 的本地 median runtime 为：base cold 约 0.7191 ms，perturbed cold 约 0.7034 ms，perturbed warm 约 0.3312 ms。每项预热 1 次，再独立运行 10 次。这些时间是本机观察，不能由 8→2 pivots 推出固定 4 倍 runtime speedup，也不是最终组内跨算法排名。

## 11 代码文件和结果怎么阅读

建议按以下顺序阅读：

1. `dual_simplex.py` 的 `Options`：共同容差、内部 pivot tolerance、10000 pivot 和 120 秒默认上限。
2. `transportation_shape`：识别矩阵结构，为 cold-start 确定 m,n。
3. `solve` 的初始化分支：cold 构造 basis，warm 复制 basis。
4. `while True` 前半段：求 basic solution、dual variables 和指标。
5. `while True` 后半段：离基规则、候选列、ratio test 和 basis 替换。
6. `run_validation.py`：外部 reference 对照、重复计时、保存结果及输入哈希验证。

| 返回字段或文件 | 用途 |
|---|---|
| `z` | 全部运输量及 slack，按共享顺序排列 |
| `y`,`r` | dual multiplier 和实际 reduced costs |
| `basis` | 最终 basis 的零基列索引，可用于配对 warm-start |
| `history` | 每个 basis 状态的指标及下一次 pivot 信息 |
| `iterations` | 已完成的 pivots；8 pivots 对应 9 个 history 状态 |
| `status`,`message` | 是否 optimal 或遇到资源/数值问题 |
| `runtime_seconds` | 内部计时，包含初始化、分解、history 与停止检查 |
| `runs.csv` | 每次 timed run 的协议结果 |
| `summary.csv` | 每个 case/mode 的成功数、median runtime 和 pivots |
| `solutions/*.npz` | 保存的 z,y,r,basis |
| `solutions/*.json` | 保存的 history、Options 及指标 |

重复 warm 计时每次都从同一个 base basis 的独立副本开始，而不是从前一次已求好的 perturbed optimum 开始。产生该 basis 的 base 求解时间另行保存，不计入 reoptimization 时间；完整使用成本应另报 base+warm。

## 12 验证结果与实现范围

当前共享 benchmark 全部通过：54 个实例 cold-start，加上 27 个 perturbed warm-start，共 81 个 case/mode，每项 10 次正式运行及 1 次预热。810 次正式运行与 81 次预热均通过。

- 最大 primal residual：9.30e-14。
- 最大 relative objective error：5.87e-16。
- 最大 relative duality gap：3.45e-16。
- Cold pivots 范围 6…909，warm pivots 范围 0…5。

另外通过了手算案例、退化/成本相同、随机小模型与 HiGHS 对照、错误输入及资源上限测试。测试用 HiGHS 验证结果，不参与 solver 的迭代。

没有合适 pivot 时，本实现返回 `numerical_failure`，而不是在浮点阈值下直接宣称模型 infeasible。当前共享实例的可行有界性已有独立验证。本版本是课程 benchmark 的清晰可核验实现，未实现通用 LP 初始化、通用 infeasibility/unboundedness certificate 或高性能 basis 更新。

## 13 可直接用于汇报的讲稿

> 我负责的是共享 transportation LP 的 Dual Simplex。它与 primal simplex 的区别在于：每轮保持 dual feasibility，通过换基恢复 primal feasibility。
>
> 在 cold-start 中，我利用运输模型的结构，为每个客户选择成本最低的一条路线，再把全部 supply 和 capacity slack 放入 basis。这样 basis 可逆；供给和容量的 dual multipliers 为零，需求 multiplier 等于该客户的最低路线成本，因此所有 reduced costs 非负。这个起点可能有负 slack，但满足 Dual Simplex 的起步要求，不需要借用 HiGHS 的 basis。
>
> 每轮我求解 basis 线性方程得到 basic values 和 dual variables。若仍有负 basic variable，就按最小原始列索引选离基变量；在该 tableau row 的负系数中，通过 dual ratio test 选入基变量，以保持 reduced costs 非负。直到变量非负且共同精度条件满足。
>
> 对需求扰动，A 和成本不变，所以原 optimal basis 的 reduced costs 不变。只需要用新 RHS 重算 basic values，再用 Dual Simplex 修复负值。统一小实例中，perturbed cold-start 需要 8 次 pivot，复用 base basis 后只需 2 次，最终 objective 都是约 775.088962。这个结果展示了本例中 basis reuse 的作用；正式 runtime 比较会按照全组统一环境进行。

## 14 汇报时常见问题

**为什么 cold-start 不是可行解也可以开始？** 因为 Dual Simplex 要求起点 dual feasible，允许 primal infeasible；不是允许最终解违反约束。

**为什么不用简单的全 slack basis？** Demand constraints 是等式，没有 demand slack。只有 p 和 q 共 M+MN 列，少了 N 列；每个客户选一个运输列正好补齐并形成非奇异 B。

**为什么在做最小化，iteration objective 却升高？** 中间 basic solution 可能不可行，其 objective 对应 dual lower bound。我们改善 lower bound，直到它对应一个可行 primal solution。

**为什么 gap 为零还不停止？** 对任何当前 basic solution 都有 c_BᵀB^{-1}b=bᵀB^{-T}c_B，即使部分 basic values 为负。必须同时检查 z≥0 与 r≥0。

**是不是 warm-start 复制了最优解，所以没有真正求解？** 复制的是 base LP 的 basis 索引。新 b 下重算 basic values 后可能不可行，仍需 pivot；本例确实做了 2 次。

**是否保证每次 warm-start 都更快？** 不保证。它的效果取决于扰动和旧 basis；本例减少 pivot，全量 warm pivots 为 0…5，但 runtime 还包含分解、复制和检查开销。

**为什么还使用 SciPy？** 使用的是线性代数 LU 运算；Dual Simplex 的初始化、pivot、停止条件均在本代码中实现。HiGHS 只作为外部验证 reference。

## 15 本说明的数据来源

代码：`COMP6704_Dual_Simplex/dual_simplex.py`。

小实例轨迹：`results_small/solutions/M03_N05_a030_seed2026_base_cold_01.json` 与 `results_small/solutions/M03_N05_a030_seed2026_demand_perturbed_warm_01.json`；计时：`results_small/summary.csv`；全量指标：`results_all/validation.json`。初始 basis、成本矩阵及第一次 ratio test 的数值直接从共享 NPZ 重建核对。表格展示值经过舍入，计算仍使用原始 float64。
