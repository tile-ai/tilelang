---
name: tilelang-perf-tuning
description: "对 Ascend950 上已有正确实现的 TileLang 算子执行性能调优：锁定用户 case，建立设备侧基线，按证据迭代候选，并完成精度、性能和搜索覆盖验收。"
---

# TileLang 性能调优

平台固定为 Ascend950；后端由调用方明确传入，PTO 使用 `TILELANG_DEFAULT_TARGET=pto`，AscendC 使用 `TILELANG_DEFAULT_TARGET=ascend`。全流程保持同一后端，不在源码中硬编码 target。

## 输入门禁

开始前必须取得：

- 工程根目录、目标算子文件、公开 wrapper/dispatch 和现有精度测试；
- 用户直接确认的完整性能 case 参数，包括 shape、dtype、属性和可选数值目标；
- 当前实现已通过的精度证据；
- 物理设备、实际导入的 TileLang 路径和目标后端。

case 不完整时只请求补充，不从 pytest、benchmark 或公开 metadata 擅自生成、删减或重新分组。确认后写入 `cases.csv` 或等价结构化文件并记录 SHA256；case 变更必须重新确认。

## 隔离目录与状态

在用户指定位置或工程内 `operators/<op>_<timestamp>/` 创建隔离目录，不修改原工程。只复制能独立运行算子和测试的最小文件集，首次确定清单后由所有候选复用。

```text
working/                         当前最佳 B*
candidates/Ci/                  从 B* 独立创建的候选
profiling/                      B0、候选和最终采集
final_optimized/                最终验收后的交付代码
optimization-search-coverage.json
perf-report.md
timing.json
```

- `B0`：不可变原始基线。
- `B*`：当前已验证最佳，初始为 B0。
- `Ci`：只从当前 B* 创建，不在未晋升候选上继续叠加。

候选未晋升时不得覆盖 `working/`；候选晋升后才按首次文件清单更新 B*。

## 工作流

### 1. 建立事实与基线

完整阅读 [性能核心](references/performance-core.md)。读取算子、wrapper、测试、实际 dispatch、当前 TileLang API/lowering 和同仓相似实现；不得按算子名或记忆推测执行路径。

从 B0 最新源码生成独立 profiling 入口，内嵌全部锁定 case，并按 [Profiling 合约](references/profiling-contract.md) 对全部 case 采集设备侧基线。记录源码 SHA、case SHA、kernel 名、launch 映射、设备、频率、warmup 和正式次数。

逐 case 建立：

- 实际执行路径、任务树、tile/core、尾块和 buffer 生命周期；
- 逻辑 GM 字节、计算量、整数索引/分支/cast/同步和固定开销；
- 硬件物理容量、当前 lowering 的有效容量、显式/常驻/多版本 footprint；
- kernel latency、主要 pipe、核间尾差和可核验下界。

首次修改源码前按 [覆盖账本](references/coverage-ledger.md) 创建并校验 `optimization-search-coverage.json`。

### 2. 路由候选知识

按当前瓶颈读取，而不是默认加载全部 reference：

- Elementwise、gather/scatter、transpose、layout、quant 或不规则访存：读取 [Vector 与 Memory 路线](references/vector-memory-routes.md)。
- Reduction/Norm/Softmax、GEMM/Quant、Scan/Recurrent、Attention 或 RoPE：只读取 [算子族模式](references/operator-patterns.md) 中对应小节。
- 多 kernel 或混合 AIC/AIV：同时读取 [Profiling 合约](references/profiling-contract.md) 的整链计时部分。

历史候选、模板和其他设备结果只用于提出假设。只有当前接口、版本、后端、数据路径和 case 兼容，且当前环境重新完成编译、精度与同口径 profiling 后才能晋升。

### 3. 建立候选池

第一次修改前最多保留四个 active candidates，后续根据新证据重排。优先级为：

1. 消除重复算法工作、GM 往返或完整中间物化；
2. 改善连续搬运、片上驻留、任务映射和负载均衡；
3. 调整 tile/core、Vector/Cube 数据流和精度路径；
4. 满足资格时再测试流水；
5. 最后考虑低收益的局部语法或指令替换。

每个候选记录可证伪假设、适用 case、六轴身份、预计消除的工作量、收益上界、API/lowering 依据、容量、精度风险、验证成本和回退点。六轴为：工作粒度、任务映射、物理数据流、精度、storage/pipeline、尾块。任一轴实质变化均分配新候选 ID。

### 4. 单假设迭代

每轮：

1. 从 B* 创建独立 `Ci`，只实施一个主要假设；必要的布局、同步或 lowering 依赖可作为同一数据流改动，但不得混入无关优化。
2. 编译并运行候选实际覆盖的定向精度 case；失败时修复或淘汰，不放宽容差、不改 reference、不跳 case。
3. 从 Ci 最新源码重新生成 profiling 入口，对全部目标 case 按同一口径采集。
4. 比较 Ci、B*、B0，检查总时间、单 case 回归、原瓶颈、核间均衡和新瓶颈。
5. 追加 coverage `RESULT`，更新 best metrics、其余候选排序和新触发的结构义务；创建下一个候选前运行普通校验。

性能接近噪声时交替复测 `B* → Ci → B* → Ci`。只有精度通过、全部 case 完成同口径采集、总体收益超过噪声且无不可接受单 case 回归时才晋升。pytest/host benchmark 可快速筛选，但不能支持最终晋升或瓶颈归因。

### 5. 停止与最终验收

满足以下任一条件才停止：达到用户目标；接近有证据的硬件/算法下界；连续候选无超噪声收益且没有更强假设；剩余高收益路线被当前 API/lowering/容量/语义确定阻断；继续优化会破坏精度或覆盖。

停止前运行 coverage `--final`。未达目标时，只有全部结构义务和逐 case 下界闭环后，才可用 `--final --allow-unmet-convergence` 声明证据收敛。设备或 profiler 故障只能记 `BLOCKED`，不是收敛证据。

最终：

1. 对 B* 运行一次现有相关完整测试套；
2. 从 B0 与 B* 最新源码重新生成入口并同口径复测全部 case；
3. 只有完整测试全绿且最终性能优于 B0 时，将 B* 归档到 `final_optimized/`；否则交付 B0 或明确标为未完成候选；
4. 生成 `perf-report.md`，包含实现状态、目标状态、全部 case 对比、最差回归、关键迭代、停止原因、未实施高收益方向、环境/SHA/命令、墙钟和产物路径。

实现状态只使用 `VERIFIED`、`NO_CHANGE`、`CANDIDATE`、`BLOCKED`；目标状态只使用 `MET`、`CONVERGED_WITHOUT_NUMERIC_TARGET`、`NOT_MET_CONVERGED`、`NOT_MET_UNFINISHED`。

## NPU 命令约束

所有访问 Ascend NPU 的命令显式使用：

```bash
env -u ASCEND_RT_VISIBLE_DEVICES TILELANG_DEFAULT_TARGET=<pto|ascend> <command>
```

需要指定设备时由入口的 `--device <physical_id>` 选择。先做一次设备健康检查；已知故障设备不得用于候选归因。测试工程有明确设备绑定约定时保持其正确口径，并在报告说明差异。

## 脚本

- `scripts/perf_summary.py`：归档 `msprof op` 的八类 CSV 并生成目标 kernel 摘要。
- `scripts/validate_optimization_search_coverage.py`：校验 coverage 普通过程记录与最终停止门禁。

脚本以本 Skill 的真实目录定位，不依赖当前工作目录。
