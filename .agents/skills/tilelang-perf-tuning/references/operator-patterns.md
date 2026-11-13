# 算子族模式

只读取与目标算子匹配的小节。模式用于产生候选，不是可直接复制的模板。

## Reduction / Norm / Softmax

- 低精度输入从 fp32 统计/累加状态开始；稳定 softmax 使用 `max` 与 `sum(exp(x-max))`。
- 若归约轴可驻留，比较精确轴长与 power-of-two padding；padding 会放大 DMA、UB 和多版本 footprint。
- 每行归约后逐元素写回时，广播参数单版本常驻，统计量尽量寄存器/UB 复用；比较重读原行与保存中间量。
- 短归约且外层记录多时，必须测试 `rows_per_tile > 1` 的连续多记录 DMA；`rows_per_tile` 与 stages 分开 A/B。
- 非归约连续轴宽时，增大输出 tile 以摊销 task/DMA/地址准备，同时重算任务波和核数。
- 行归约与列归约的数据连续性不同，应独立建模和路由；不要用一个通用 tile 掩盖两类路径。
- 固定小状态多轮迭代时比较状态跨轮常驻、只保存反向需要的快照、减少真实依赖之外的 barrier。

## GEMM / Quant / LSTM

- GEMM 优先从当前仓库已运行的 Cube 路径适配；确认 A/B 布局、`transpose_B`、L1/L0 分配和输出路径。
- L0C 默认 fp32，首个有效 K tile 清零，后续累加；嵌套 K 子块只能在全局首块清零。
- 联合搜索 M/N/K tile、L1/L0 footprint、独立输出 tile 数和 K 流水，不单独追求更大 tile。
- 权重跨时间步、batch 或输出 tile 重用时优先比较转换后权重/L1/UB 驻留与预计算输入贡献。
- Quant 同时建模 packed load、解包/反量化、scale 广播、ND→NZ/布局转换和 MMAD；在 producer 内融合反量化可避免完整权重物化。
- 原生低精度输入/残差路径和 fast math 必须经过原容差；递归状态和长链累积通常比独立输入贡献更敏感。
- 小 M/B 时检查 padded MMAD 浪费和 batch 并行度；分组提高局部效率但可能损失任务数。

## Scan / Recurrent / Stateful

- 先标出必须串行的状态轴和可并行的 batch/channel/feature 轴；不要尝试展平真实依赖。
- 状态、carry、循环不变量和广播参数尽量跨步常驻；按状态大小、live register 和代码体积选择寄存器或 UB。
- 比较 loop order 对状态往返、连续访问和任务数的影响；改变 loop order 后重新建模尾部和同步。
- 标量广播可能比算术本身昂贵；核对 broadcast load、vdup/vmuls/vaxpy 等当前 lowering 的实际指令成本。
- 多核切状态轴只有在跨核归约/边界同步成本可控时进入候选；先量化每步同步次数和最坏延迟。
- 对长序列给出逐状态最小操作数和依赖链下界，区分算法限制与代码生成吞吐差距。

## Attention / Sparse Attention

- Q/K/V 与 online softmax 的物理数据流一起建模：Q 驻留、K/V tile、max/sum 状态、输出累加和 gather 路径。
- softmax max/sum 与 GEMM accumulator 默认 fp32；检查 causal/all-masked、尾块和输出 dtype。
- GQA/MLA、Dk/Dv、topK、S1 等会改变 Cube padding、随机 gather 流量和任务数，按实际路径分组。
- 热循环中的除法/取模、sub-block 路由和索引表生成优先外提或改为仿射任务映射。
- 随机索引不支持批量 DMA 时，分别裁决标量/SIMT、indexed Vector、可界定连续窗口和 materialized gather；不能仅因某个 gather API 缺失就宣布收敛。
- split-K、多 kernel partial/merge 和 Cube/Vector forwarding 要计入全部 workspace、launch、同步和整链时间。
- 小 G/M 的 Cube padding 可能比有用 FLOP 大得多；同时报告有用 FLOP 和实际 padded 计算下界。

## Elementwise / Broadcast / RoPE

- 连续逐元素优先 GM→UB→Vector→UB→GM，常量和复用表移出 Persistent 热循环。
- Broadcast 比较 resident input、跨行/跨 token 复用与 shape-generic 尾部；避免每 task 重载小表。
- RoPE 的 half-split 与 interleaved 使用不同物理重排；核对 sin/cos 是 half dim 还是 full dim，减少重复装载。
- 令牌/行对齐分块可把热循环地址变成仿射并消除除法/取模；先量化这些标量操作在生成代码中的成本。
- interleaved/短记录优先比较连续 load + register select/interleave 与 indexed gather，不先写完整 planar 临时布局。
- fp16/bf16 超越函数和旋转统计按原精度要求选择 fp32 或经验证 fast path，输出最后转换。

## Gather / Scatter / Layout / Slice

- 将顺序的 index/output 流与随机 source/destination 流分开；顺序流尽量用 DMA 宽事务，随机流再选择 SIMT、indexed Vector 或窗口重排。
- 对 stride-2/4、interleave/deinterleave、transpose 等规则映射优先验证连续 load + register reorder；检查是否存在重叠 load、随路扩宽和多余中转。
- Scatter 必须区分唯一目标、重复目标和原子语义；缺少竞争的测试不能证明普通 store 正确。
- Slice/Transpose 根据最大连续后缀和 stride 建模二维 copy、整行 copy、register reorder 和 gather fallback。
- 输出超过 L2 或物理流量已接近 HBM 下界时，不用不可能的目标驱动无证据微调；仍需裁决可减少物理流量的路线。
- dtype 改变 lane 数、load 对齐和 register pack；int8/int64 通常需要独立成本链，不能直接继承 fp16 路线。

## 通用验收

任何模式都必须回到当前源码和 case：确认 API/lowering、容量、精度、实际激活分支和同口径性能。模式命中不代表收益，更不允许继承历史加速比。
