# 正式实验环境记录模板

状态：待填写/待组内确认。复制此模板用于每套正式环境；不要把空项视为已批准配置。

- environment_id：
- protocol_version：proposed v1
- 确认人、日期：
- CPU、RAM、OS/build：
- 语言及 runtime 版本：
- NumPy/SciPy 或 MATLAB 版本：
- BLAS/LAPACK 实现、实际线程数及设置方法：
- algorithm/version/source revision：
- options_id，pivot/初始化/line-search/regularization 等设置：
- 内部 tolerance；共同验收 tolerance 见协议：
- 算法 RNG seed（如使用）：
- iteration/time limits：
- 实例及算法执行顺序：
- 同机/后台负载控制：
- warm state 范围（basis、factorization 等）：
- 与协议不同的地方及全组确认记录：

每个 case/algorithm/mode：1 次预热，10 次独立计时。输出原始记录、成功率及 median runtime，保留失败信息。现有数据验证环境为 Python 3.11.15、NumPy 1.26.4、SciPy 1.11.4；正式计时环境尚未选定。
