# Related Works：运输与物流优化中的线性规划方法综述

> 适用主题：COMP6704 Group Project — Linear Programming（Transportation and Logistics Optimization）
>
> 文献 PDF 均存于 `docs/papers/`（25 篇，arXiv 编号见文末清单）。经典奠基文献（Hitchcock 1941、Kantorovich 1939、Dantzig 1951、Karmarkar 1984）无公开 PDF，引用信息见文末。

---

## 一、方法分类（五大类 + 应用层）

### 1. 单纯形类方法（Simplex Family）

沿可行域顶点移动的经典精确解法，包括原始/修正/对偶单纯形法，以及利用运输问题网络结构的**网络单纯形法（Network Simplex）**。

| 文献 | 贡献 |
|---|---|
| Huangfu & Hall 2015 (`1503.01889`) | 对偶修正单纯形法的并行化——解决单纯形难以并行的核心痛点 |
| Watanabe et al. 2017 (`1706.04302`) | 最大流问题上的网络单纯形实现 |
| Holzhauser et al. 2016 (`1607.02284`) | 预算约束最小费用流的网络单纯形法 |
| Cornelissen & Manthey 2015 (`1504.08251`) | 网络单纯形与最小平均环消去法的 smoothed analysis（理论复杂度分析） |
| HiGHS (Huangfu & Hall 2018, 见文末) | 工业级 LP 求解器，对偶单纯形为核心引擎 |

**特点**：精确顶点解、天然产生基（basis）结构，是运输问题最经典匹配的方法；对偶值/灵敏度分析（影子价格）免费获得——这正是物流中"运力约束边际价值"的来源。

### 2. 内点法（Interior-Point Methods, IPM）

多项式复杂度的路径跟踪方法，从可行域内部逼近最优解。

| 文献 | 贡献 |
|---|---|
| Mehlhorn & Saxena 2015 (`1510.03339`) | 极简 IPM 教程（入门推导用） |
| Lin et al. 2018 (`1805.12344`) | ADMM-Based IPM——用 ADMM 加速 IPM 中的牛顿方程求解，面向大规模 LP |
| Cornelis & Vanroose 2021 (`2105.01333`) | 正则化非精确 IPM 的收敛性分析——解决牛顿方程病态/非精确求解下的收敛问题 |

**特点**：最坏情况多项式复杂度，大规模稀疏问题上迭代次数少（几十次即收敛）；但每次迭代需解大规模线性方程组（稀疏 Cholesky 分解），难以并行。

### 3. 一阶方法与 GPU 大规模求解（First-Order / GPU Solvers）⭐ 近年最热

以 Primal-Dual Hybrid Gradient（PDHG，即 PDLP 系列）为代表的可并行一阶方法，是 2023–2026 年 LP 求解器的主战场。

| 文献 | 贡献 |
|---|---|
| Applegate et al. 2025 (`2501.07018`) | **PDLP**（Google）：把 PDHG 加重启机制做成实用大规模 LP 求解器，可解百万级变量 |
| Lu, Peng & Yang 2025 (`2507.14051`) | **cuPDLPx**：GPU 一阶求解器的性能增强（预条件、缩放等） |
| Li et al. 2026 (`2601.07628`) | **D-PDLP**：扩展到分布式多 GPU 系统 |
| Rothberg 2026 (`2603.03150`) | **混合 PDHG 与 IPM**：一阶方法做粗解 + IPM 精化的混合框架 |
| Cederberg & Boyd 2026 (`2604.23951`) | 面向 GPU 一阶求解器的预求解（presolve）技术 |
| Prasad & Sharma 2026 (`2606.08638`) | GPU LP 求解器参数自动调优的泛化保证 |

**特点**：迭代便宜、天然适合 GPU/分布式并行，规模上限远超 IPM；但**解精度只有中等**（相对误差 1e-4 ~ 1e-6），尾部收敛慢，对缩放/病态敏感。

### 4. 结构利用与分解方法（Structure-Exploiting / Decomposition）

利用运输/物流问题的网络结构或可分解性，把大 LP 拆成小问题求解。

| 文献 | 贡献 |
|---|---|
| Dai, Zhao & Liu 2019 (`1903.07469`) | 分裂式多商品流问题的 LP 高效算法 |
| van den Brand & Zhang 2023 (`2304.12992`) | 借助单商品流技术加速高精度多商品流求解 |
| Guan, Hijazi & Van Hentenryck 2026 (`2607.16618`) | 造纸业端到端供应链规划：**列生成 + Benders 分解**的工业级应用 |
| Saraswathi & Kümmerle 2026 (`2609.09295`) | 固定费用网络流（fixed-charge network flow）的支持集发现（IRLS）——LP 松弛 + 支持识别 |

**特点**：node-arc 公式化下多商品流 LP 规模爆炸，分解方法（Dantzig-Wolfe/列生成、Benders）是唯一出路；代价是主问题-子问题迭代有 tailing-off 现象。

### 5. 学习增强优化（Learning-Augmented Optimization）⭐ 前沿方向

用机器学习加速传统求解器（warm-start、参数选择、加速子过程），不改变求解器的精确性保证。

| 文献 | 贡献 |
|---|---|
| Sul et al. 2026 (`2610.01546`) | 强化学习加速 PDHG 求 LP（学习步长/重启策略） |
| Prasad & Sharma 2026 (`2606.08638`) | （兼属第 3 类）GPU LP 参数调优的泛化界 |
| Saraswathi & Kümmerle 2026 (`2609.09295`) | （兼属第 4 类）学习辅助识别固定费用网络流的 LP 支持集 |

**特点**：目前主要做"加速器"而非替代精确求解；泛化性和可解释性是争议焦点。

### 6. 运输问题专用理论与算法（Problem-Specific Theory）

| 文献 | 贡献 |
|---|---|
| Bargetto, Della Croce & Scatamacchia 2023 (`2302.10826`) | **Iterated Inside Out**：运输问题新的精确算法，挑战单纯形/IPM 在经典运输问题上的地位 |
| Nickel et al. 2026 (`2602.15151`) | Monge 费用结构下运输问题的新理论及其在离散有序中位问题上的应用 |
| Bai 2024 (`2407.06481`) | Sinkhorn（熵正则化 OT）与 LP 求解器在部分运输问题上的系统对比 |
| Im & Wolkowicz 2022 (`2203.02795`) | **LP 退化性（degeneracy）、严格可行性、稳定性的再审视**——运输问题是高度退化 LP 的典型代表 |

### 7. 应用层（Applications in Transport & Logistics）

| 文献 | 贡献 |
|---|---|
| Bridgelall 2022 (`2211.07345`) | **LP 教程：供应链与运输物流优化**（问题建模的最佳起点）⭐ |
| Munari, Dollevoet & Spliet 2016 (`1606.01935`) | 车辆路径问题（VRP）的广义 LP 公式化框架 |
| Su et al. 2025 (`2506.10311`) | 公交+无人机多式联运货运（最后一公里配送） |

---

## 二、各方法的 Bottleneck（报告"动机/讨论"部分可直接用）

### Simplex 类
1. **最坏情况指数复杂度**（Klee–Minty 构造）；平均表现好但缺乏强理论保证，smoothed analysis 是目前最好的解释（`1504.08251`）。
2. **退化性（degeneracy）**：运输问题的 LP 是高度退化的（供需约束的基解大量重复），导致单纯形迭代停滞/循环风险、基交换次数暴增（`2203.02795` 对此有专门理论分析）。**这是运输问题 + 单纯形组合最典型的 bottleneck，适合作为实验中的 sensitivity analysis 切入点**。
3. **本质串行**：基反转（basis update）按顺序进行，并行化只能加速子步骤（定价、比值检验），加速比有限（`1503.01889`）。

### Interior-Point
1. **每次迭代的稀疏线性方程组求解**（牛顿系统）主导开销：稀疏 Cholesky 分解内存和计算随问题规模增长快，且难以 GPU 化。
2. **病态性**：逼近最优时中心路径参数 μ→0，牛顿方程条件数爆炸；需正则化或非精确求解技巧（`2105.01333`）。
3. **解是内部点**：物流应用往往需要基解/顶点解（方便解释"哪些运输线路被启用"），需要 crossover 步骤转回顶点解。

### 一阶方法（PDHG/GPU）
1. **精度天花板**：只能达到中等精度（~1e-6 相对误差），而单纯形/IPM 可到机器精度；供应链规划等需要精确对账的场景是硬伤。
2. **尾部收敛慢 + 对缩放病态敏感**：依赖重启策略、预条件、presolve 等工程手段（`2501.07018`、`2604.23951`），参数调优本身成为研究问题（`2606.08638`）。
3. **无基结构**：不产生对偶基/影子价格的直观解释，灵敏度分析能力弱于单纯形。

### 结构利用/分解
1. **适用范围受限**：网络单纯形只对纯网络流 LP 有效，加入固定费用、非线性成本即失效（`1607.02284`、`2609.09295` 正是在修补这个缺口）。
2. **列生成的 tailing-off**：主问题迭代后期每步改进微小，收敛慢（`2607.16618`）；定价子问题本身可能很难。

### 学习增强
1. **泛化性**：在训练分布外的实例上加速效果不保证（`2606.08638` 给出泛化界）。
2. **信任问题**：学习组件不能破坏精确性与可复现性，只做加速角色。

### 跨方法的核心矛盾（可作为报告的 "gap" 叙事）
> **精度 vs 规模 vs 并行性三者不可兼得**：单纯形（高精度、顶点解、串行）— IPM（高精度、多项式、难并行）— 一阶方法（可并行、大规模、低精度）。最新的研究趋势是**混合化**（`2603.03150` PDHG+IPM、`1805.12344` ADMM+IPM）和**学习加速**（`2610.01546`）。而运输问题的高度退化性（`2203.02795`）和天然网络结构（网络单纯形）让它在方法对比实验中是一个极有信息量的 benchmark。

---

## 三、对课程项目的启示（方法选型建议）

结合 "diversity of comparison methods" 评分点，4 人分工建议：

| 成员 | 实现方法 | 覆盖类别 | 对应文献 |
|---|---|---|---|
| A | 原始/修正单纯形法（含 Bland 规则防循环） | 类别 1 | `1510.03339` 类教程 + Chvátal 教材 |
| B | 对偶单纯形法 | 类别 1 | `1503.01889` |
| C | 原始-对偶内点法 | 类别 2 | `1510.03339`、`2105.01333` |
| D | 网络单纯形法（利用运输问题网络结构） | 类别 4 | `1607.02284`、`1706.04302` |

实验设计上可再加一个第三方基线（HiGHS/SciPy 内置求解器）对照，以及退化实例上的灵敏度分析——正好呼应 `2203.02795` 与一阶方法文献中反复提到的 bottleneck。

---

## 附：文献清单（docs/papers/）

### arXiv 下载（25 篇）

**综述/教程**
- [2022] Bridgelall, R. *Tutorial and Practice in Linear Programming: Optimization Problems in Supply Chain and Transport Logistics.* arXiv:2211.07345
- [2015] Mehlhorn, K., & Saxena, S. *A Still Simpler Way of Introducing the Interior-Point Method for Linear Programming.* arXiv:1510.03339

**Simplex 类**
- [2015] Huangfu, Q., & Hall, J. A. J. *Parallelizing the dual revised simplex method.* arXiv:1503.01889
- [2017] Watanabe, S., et al. *Network Simplex Algorithm associated with the Maximum Flow Problem.* arXiv:1706.04302
- [2016] Holzhauser, M., Krumke, S. O., & Thielen, C. *A Network Simplex Method for the Budget-Constrained Minimum Cost Flow Problem.* arXiv:1607.02284
- [2015] Cornelissen, K., & Manthey, B. *Smoothed Analysis of the Minimum-Mean Cycle Canceling Algorithm and the Network Simplex Algorithm.* arXiv:1504.08251

**Interior-Point 类**
- [2018] Lin, T., Ma, S., Ye, Y., & Zhang, S. *An ADMM-Based Interior-Point Method for Large-Scale Linear Programming.* arXiv:1805.12344
- [2021] Cornelis, J., & Vanroose, W. *Convergence analysis of a regularized inexact interior-point method for linear programming problems.* arXiv:2105.01333

**一阶方法 / GPU**
- [2025] Applegate, D., et al. *PDLP: A Practical First-Order Method for Large-Scale Linear Programming.* arXiv:2501.07018
- [2025] Lu, H., Peng, Z., & Yang, J. *cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear Programming.* arXiv:2507.14051
- [2026] Li, H., et al. *D-PDLP: Scaling PDLP to Distributed Multi-GPU Systems.* arXiv:2601.07628
- [2026] Rothberg, E. *Hybridizing PDHG and Interior-Point Methods.* arXiv:2603.03150
- [2026] Cederberg, D., & Boyd, S. *Presolving for GPU-Accelerated First-Order LP Solvers.* arXiv:2604.23951
- [2026] Prasad, S., & Sharma, D. *Parameter Tuning with Generalization Guarantees for GPU-Accelerated Linear Programming.* arXiv:2606.08638

**结构利用 / 分解**
- [2019] Dai, L., Zhao, H., & Liu, Z. *Solving Splitted Multi-Commodity Flow Problem by Efficient Linear Programming Algorithm.* arXiv:1903.07469
- [2023] van den Brand, J., & Zhang, D. *Faster High Accuracy Multi-Commodity Flow from Single-Commodity Techniques.* arXiv:2304.12992
- [2026] Guan, C., Hijazi, A., & Van Hentenryck, P. *End-to-End Supply Chain Planning in the Paper Industry Via Column Generation and Benders Decomposition.* arXiv:2607.16618
- [2026] Saraswathi, S., & Kümmerle, C. *Support Discovery With Iteratively Reweighted Least Squares for Fixed-Charge Network Flow.* arXiv:2609.09295

**运输问题理论**
- [2023] Bargetto, R., Della Croce, F., & Scatamacchia, R. *ITERATED INSIDE OUT: a new exact algorithm for the transportation problem.* arXiv:2302.10826
- [2026] Nickel, S., et al. *Revisiting transportation problems under Monge costs with applications to the Discrete Ordered Median Problem.* arXiv:2602.15151
- [2024] Bai, Y. *Sinkhorn algorithms and linear programming solvers for optimal partial transport problems.* arXiv:2407.06481
- [2022] Im, J., & Wolkowicz, H. *Revisiting Degeneracy, Strict Feasibility, Stability, in Linear Programming.* arXiv:2203.02795

**学习增强**
- [2026] Sul, J., et al. *Reinforcement Learning to Accelerate Primal-Dual Hybrid Gradient for Linear Programming.* arXiv:2610.01546

**应用**
- [2016] Munari, P., Dollevoet, T., & Spliet, R. *A generalized formulation for vehicle routing problems.* arXiv:1606.01935
- [2025] Su, E., et al. *The Freight Multimodal Transport Problem with Buses and Drones: An Integrated Approach for Last-Mile Delivery.* arXiv:2506.10311

### 经典文献（无公开 PDF，需图书馆数据库获取，引用必放）

- Hitchcock, F. L. (1941). *The Distribution of a Product from Several Sources to Numerous Localities.* Journal of Mathematics and Physics, 20(1-4), 224–230. — 运输问题 LP 的最早公式化
- Kantorovich, L. V. (1939/1960). *Mathematical Methods of Organizing and Planning Production.* Management Science, 6(4), 366–422. — LP 思想起源
- Dantzig, G. B. (1951). *Application of the Simplex Method to a Transportation Problem.* In Activity Analysis of Production and Allocation (Ch. XXIII). — 单纯形法首次应用于运输问题
- Chvátal, V. (1983). *Linear Programming.* W. H. Freeman. — 教材（方法实现参考）
- Karmarkar, N. (1984). *A New Polynomial-Time Algorithm for Linear Programming.* Combinatorica, 4(4), 373–395. — 内点法开端
- Huangfu, Q., & Hall, J. A. J. (2018). *HiGHS: High-performance software for linear optimization.* Mathematical Programming Computation, 10, 109–130. DOI: 10.1007/s12532-017-0130-5（开放获取，实验基线求解器的引用）
