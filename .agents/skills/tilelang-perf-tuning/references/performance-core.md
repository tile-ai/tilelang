# 性能核心

本文件用于每次性能调优，只保留会改变候选排序或验收结论的通用规则。

## 事实优先级

1. 当前算子、公开 wrapper/dispatch、测试和锁定 case；
2. 本轮编译、精度和设备侧 profiling；
3. 实际导入的 TileLang 源码、目标后端 lowering、同仓已运行实现；
4. 历史候选、其他设备数据和设计资料。

低优先级资料不能覆盖高优先级实测。定位实际 TileLang：

```bash
python -c 'import pathlib,tilelang; print(pathlib.Path(tilelang.__file__).resolve()); print(getattr(tilelang,"__version__","unknown"))'
```

API、参数、布局、同步、L2 策略和多版本能力必须在该源码、lowering、同版本示例或当前编译结果中确认。找不到依据时标为待验证或 `DESIGN_ONLY`，不凭记忆实现。

## 基线模型

逐 case 记录：

```text
语义路径 → dispatch → primfunc/kernel → task mapping → tile loop → execution domain
```

同时建立五类成本：

- **逻辑工作**：每个输出依赖的维度、重复计算、公共子表达式、循环不变量；
- **地址/控制**：整数乘除/取模、位运算、cast、分支、索引和每线程 live state；
- **数据移动**：GM/UB/L1/L0 的有效与物理读写、中间物化、重复搬运、同步；
- **并行结构**：独立任务数、实际核数、最长/最短核工作量、完整波和尾波；
- **数值工作**：Vector/Cube 运算、归约顺序、累加 dtype 和近似指令。

逻辑工作下降只是收益上界，不能写成实测加速。

### 资源账本

物理容量来自当前设备证据；编译器有效容量从当前 lowering 或编译结果确认。不得继承历史机器的固定安全 cap。

```text
显式 footprint = Σ(product(padded_shape) × dtype_bytes × versions)
总 footprint = 显式 + 常驻数据 + 已证明临时量 + 有依据的安全余量
剩余容量 = 当前 lowering 有效上限 - 总 footprint
```

增加或移除 `SimtVF`、buffer version、padding、LUT/index、临时布局或执行域后重新计算。GM 尾块只访问 valid，UB 仍按 SIMD/DMA 真实可触达 footprint 分配。

## 瓶颈归因

pipe ratio 是线索，不是源码根因。具体归因至少结合源码工作量、生成 IR/指令、内存流量、timeline 或隔离实验之一。

| 类型 | 证据 | 优先候选 |
|---|---|---|
| GM/MTE | 大量 GM 往返、小 burst、实际时间接近或远高于搬运下界 | 融合、连续搬运、驻留、L2 策略、减少物化 |
| Vector | VEC 时间主导且计算/重排工作量能解释 | 寄存器复用、减少 cast/遍历、SIMD 数据流、fast math |
| Scalar/调度 | 动态索引、分支、循环控制或固定段主导 | 外提不变量、仿射任务、合并工作、编译期通用特化 |
| 并行度 | 任务少于核或尾差显著 | inner flatten、调整核数/粒度、makespan 映射 |
| Cube | Cube 时间主导或 MMAD 利用不足 | M/N/K tile、L1/L0C、K 流水、输出路径 |
| Launch | kernel 很短，固定启动/多 kernel 占主要时间 | 合并 kernel/dispatch；同时保留整链口径 |
| 混合 | 不同 case 或阶段主瓶颈不同 | 按通用结构分支，逐 case 裁决 |

逻辑带宽是源码字节/耗时，不等于物理 HBM 利用率。L2 命中、原子读改写、重复消费和写分配会改变物理流量。

## 候选排序

按以下顺序检查，但以当前瓶颈和收益上界决定实际排序：

1. 消除跨维度重复计算、重复归约或多余 kernel；
2. 消除 GM 往返、完整中间物化和重复读；
3. 常驻权重、状态、LUT、索引或广播参数；
4. 展平独立任务、改善核间尾差和固定段摊销；
5. 搜索有来源的 tile/core 与物理 Vector/Cube 路线；
6. 在 MTE/Compute 均有可见时间且每核迭代充足时测试流水；
7. 有生成指令差异时再测试局部算术/lowering 写法。

连续两个候选收益低于约 3% 且仍有明显目标差距时，停止等价微调，重新检查算法复用、数据流、live state、任务映射和历史已验证结构。需要超过 1.2× 时，不连续优先实施只能解释低个位数收益的候选，除非它是高收益路线的必要前置。

## 执行域与数值

- 规则连续逐元素通常优先比较 `SimdVF`；离散索引、复杂分支、原子或依赖线程语义时使用 `SimtVF`。同仓已验证路径优先于抽象偏好。
- `T.Kernel` 保持一维 block 网格；多轴任务由调度结构表达。
- fp16/bf16 reduction、norm、softmax 统计和 GEMM 累加默认从 fp32 状态开始。
- 内部低精度、HF32、硬件除法、fast math 和近似指令可以实验，但不得改变 reference、原容差或公开语义；定向精度和全量回归均通过后才保留。
- 尾块不得读取未初始化 lane；完整寄存器访问必须清零、填充、钳位或正确 mask。
- JIT/host 分支只能依据通用性质：dtype、对齐、容量、任务数、连续性、尾块或硬件上限；不得按公开 case ID 或精确测试 shape 投机。

## 历史复用

先检查同算子 git 历史、已有调优目录和同仓实现。复用时核对当前接口与 case、TileLang/后端/设备版本、数据驻留与布局、中间物化与同步、tile/core/stage、尾块、launcher 参数、原性能口径和源码 SHA。

兼容历史候选仍只进入候选池，不直接成为 B*。模板文件存在、名称相同或历史标为 verified 都不能替代当前重验。

## 精度与性能门禁

- 每次候选先验证实际激活的定向 shape；最终只运行一次现有相关完整测试套。
- 禁止扩大容差、降低 reference 精度、跳过失败 shape 或把子集通过称为全量通过。
- 区分数值失败、编译失败、kernel 执行失败、profiler 失败和主动 `NotImplementedError`。
- 基线与候选保持输入、dtype、输出语义、设备、频率、warmup、repeat、并发和计时层级一致。
- 晋升同时报告几何平均或用户指定指标、最差 case 和所有退化 case；平均值不能掩盖单 case。
- 没有当前设备有效实测时只称候选或设计，不宣称收益。
