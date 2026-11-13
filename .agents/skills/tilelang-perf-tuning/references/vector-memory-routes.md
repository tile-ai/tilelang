# Vector 与 Memory 路线

适用于 Elementwise、Gather/Scatter、Transpose/Layout、Quant、短记录打包和其他 Vector/不规则访存 kernel。

## 工作树与 tile 边界

先将计算写为：

```text
外层 item（row/token/segment/patch）
└── 内层独立 chunk（channel/vector tile/pixel round）
```

若外层 item 少于可用核，且 chunk 间无归约、顺序依赖或重叠写，必须把 `(item, chunk)` 展平作为候选；同时比较任务太小导致的固定开销。`T.serial` 跨步循环与 `T.Persistent` 都只是候选，不按名称判断快慢。

tile/group 候选从五类真实边界产生：最大完整连续语义单元；SIMD lane 和有效 payload；DMA burst/二维整行/对齐/stride；计入常驻、padding 和 versions 后的容量；一波、两波、稳态多波及尾 tile 比例。去重后形成小集合。一个小 group 的失败只否决该值，不能关闭完整连续单元或容量上限候选。

## 四类物理路线

| 路线 | 物理结构 | 主要成本 |
|---|---|---|
| `SCALAR_OR_SIMT` | 每线程独立索引、分支和读写 | 线程上下文、标量地址、在飞 load、活跃线程 |
| `MEMORY_INDEXED_VECTOR` | Vector lane 对 UB/可支持层级做 gather/scatter | 离散访问、索引寄存器、随路扩宽、lane 利用 |
| `CONTIGUOUS_LOAD_REGISTER_REORDER` | 连续 load 后在寄存器 select/shuffle/interleave/pack | 连续读写、重排指令、跨寄存器与 live state |
| `MATERIALIZED_TRANSFORM` | 先在 UB/L1 物化 planar/transpose/scratch 再计算 | 额外完整读写、容量、同步、物化遍数 |

最后算术在 Vector register 中不代表整条路线是 register-resident；只要完整 payload 曾写入临时布局，就属于 materialized transform。

## Indexed 与 Register Reorder

对每个可重复热循环窗口记录 `source_span_bytes`、`consumed_unique_bytes`、density、`window_vregs`、索引规律和复用次数。

- 窗口可装入合理寄存器 footprint，密度高或可复用：优先验证连续 load + register reorder。
- 跨度大、只消费少量元素、完整窗口会严重过取或溢出：优先验证 indexed Vector。
- 分段连续：比较“分段连续 load + 段内重排”和 gather。
- 原生 MTE 可表达连续段时，和逐 lane 访问比较完整 GM/UB 路线。

核对每条 intrinsic 的输入/输出 dtype、lane、part、扩宽、predicate、base/index 单位和 lowering。一次实现若包含多余转换、lane 修复或中间物化，只淘汰该完整链；仍有更低成本代表时必须重开路线。

跨多个寄存器时将索引重映射到局部窗口，再合并或写出；不能把全局 lane index 直接交给单寄存器 select。动态 lane offset 与动态 intrinsic base 是不同能力，某个动态 stage base 失败不能外推成所有动态索引都不支持。

## 正交组合

候选身份至少包含 `(tile/group, task mapping, physical dataflow, precision, pipeline, tail)`。一个轴可能改变另一个轴的连续 payload、固定成本或主瓶颈，因此“基线 tile + 新路线”“新 tile + 基线路线”“新 tile + 新路线”是三个不同裁决。两个单轴失败不能替代组合。

跨路线比较尽量固定其他轴；无法固定时明确列出差异，结果只裁决完整实现链。

## 驻留、搬运和 L2

- 首先减少 GM 往返和重复读取，再追求 overlap。
- 跨迭代不变的权重、scale、小表、索引和状态尽量单版本常驻。
- 连续输出优先宽 store；随机写、重复目标和原子语义分别处理。
- L2 策略只在当前 API/lowering 有依据时 A/B；输入、权重和输出可能需要不同策略。命中率上升但总时间不降时撤销。
- 大 tile 同时影响 burst、任务数、尾部、版本 footprint 和寄存器压力；不能只追求容量上限。

## 流水资格与实现

只有同时满足以下条件才建立流水候选：profiling 显示 MTE 与 Compute 都有可重叠时间；每核有足够独立完整迭代；UB/L1 能容纳所有跨迭代存活 buffer 的版本；依赖允许下一轮 copy in 与上一轮 Compute/copy out 重叠。

先画每个 buffer 的 `copy in → compute → copy out` 生命周期。输入、输出、索引和临时量只要跨迭代仍存活都要版本化；只读常驻量保持单版本。full-wave 使用固定 extent，尾部单独处理。

自动版本化要求单份逻辑 tile、唯一生产者/消费者、固定 rank/extent、无隐藏别名、全部活跃 buffer 被识别。`num_stages > 1`、两个 buffer、编译成功或运行正确都不能证明流水生效。

自动组合失败只淘汰该组合。容量与 API 允许时继续裁决显式 stage storage、手动 annotation、静态 stage body 和 `T.Pipelined` full-wave。自动完整生效后无需重复手动等价实现。

流水生效必须由同 tile/算术/任务映射的 stage-1/2 配对证明：latency 超噪声改善、有效带宽或算力改善、对应 overlap 增强，必要时生成 IR 中存在正确版本轮转与同步。否则记录“编译通过但未生效”。

## 尾部与顺序稳定性

- GM 只访问 valid，UB 按完整指令 footprint 分配。
- full/tail 分离不得改变公开语义或遗漏允许余数类。
- 多版本、同步、Persistent 或编译 warning 候选，在同一进程覆盖全部 case，并用逆序或交错 case 顺序复测。
- 顺序相关漂移、偶发错误、死锁或内存破坏淘汰精确组合并保留证据。
