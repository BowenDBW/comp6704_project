各位，共享的 COMP6704 Benchmark v1 分发包已更新。除了 Instance Generator 和 54 个实例（27 base + 27 demand-perturbed），这次也补齐了统一问题背景、建模假设、全部变量/参数的名字、定义与物理意义、standard-form 排序、算法接口，以及 proposed comparison protocol，方便大家从同一个优化问题出发编写代码、报告和展示。

数据已通过维度、可行性、SciPy HiGHS、NPZ/MAT 一致性和独立重复生成检查，54/54 通过；最大 HiGHS primal residual 约 2.13e-14。原代码和原始 data 未改动。此次更新仅补充文档与结果模板，不改变实例。

解压后从 START_HERE.md 开始，先看 SHARED_PROBLEM_SPECIFICATION.md，再看 PROPOSED_COMPARISON_PROTOCOL.md。Python 用 np.load 加载 benchmark_v1/instances/ 下的 .npz，MATLAB 用 load 加载 .mat，取 A、b、c_hat，统一求解 min c_hatᵀz，Az=b，z≥0。请注意 row-major 变量顺序，直接使用已提供的矩阵。Reference objective 在根目录 VALIDATION_SUMMARY.csv 中按 case_id 查找 highs_objective 列；它只用于求解后的 accuracy 检查，不是算法停止条件。

接口联调先统一使用 M03_N05_a030_seed2026_base。我们先按 proposed protocol 开发：primal residual≤1e-7、min(z)≥−1e-8、relative objective error≤1e-7；dual feasibility/gap 的具体公式和阈值见协议。返回结果字段见 result_schema.csv。正式 runtime 比较统一机器和环境，计入初始化与求解，排除文件读取和 reference 验证；每项预热 1 次、独立运行 10 次，报告 median runtime。Dual Simplex 的 cold/warm reoptimization 单独比较，warm 复用自己的 base optimal basis，base 求解时间另报。

目前这些规则仍是 proposed v1，方便大家先联调，并不代表全组已经批准。请大家确认模型、符号/排序、实例集合、停止容差、计时范围及 warm-start 规则；正式实验前再补齐统一机器、线程设置和资源上限，冻结版本。如果发现不一致，先组内同步，不各自修改 benchmark。现有验证结论针对数据；四种自写算法仍需各自实现和验证。
