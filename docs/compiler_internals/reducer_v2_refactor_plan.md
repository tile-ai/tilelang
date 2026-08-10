# Reducer v2 重构设计方案

- 状态：v2 correctness baseline 与 conservative LocalComplete fast path 已在 `refactor/reducer-v2` 分支实现并进入验证；subgroup fast path 尚未实现
- 更新日期：2026-08-06
- 相关讨论：[RFC #2897](https://github.com/tile-ai/tilelang/issues/2897)
- 当前止血修复：[PR #2881](https://github.com/tile-ai/tilelang/pull/2881)

## 1. 结论先行

这次重构的核心不是“让 `FinalizeReducer` 推导出更准确的线程数”，而是重新定义 reducer 的语义边界：

> reducer 表示一个 deferred reduction epoch；`ReducerUpdateOp` 决定一个逻辑 contribution 出现几次，layout 只决定数据存在哪里、如何通信，不能改变 contribution multiplicity。

已确定采用以下设计：

1. v2 只接受 first-class `ReducerInitOp`、`ReducerUpdateOp` 和 `FinalizeReducerOp`；不支持旧的 `T.clear`/`T.fill`、任意 reducer `BufferStore` 或 `acc[i] += value` 写法。
2. `T.reducer_init` 必须显式出现；`FinalizeReducerOp` 必须写入独立 destination，不支持 in-place finalize。
3. reducer 在高层 IR 中使用不可普通读写的 `local.reducer` 虚拟 storage scope，而不是 `local.fragment`。
4. correctness baseline 将 reducer 物化为每个 participant 一份完整 logical output array 的普通 `local` partial；每个 update site 独立消除由其 `T.Parallel` layout 产生的重复执行，finalize 再做 participant-wide collective。
5. `PartitionLoop` 不再接收 reducer buffer 列表。`ReducerUpdateOp` lowering 产生通用的 `parallel multiplicity = once` statement marker，loop partition 只负责将该 marker 实现为 `REP == 0`。
6. `ThreadReduceStep` / `ReduceOwnershipPlan` 只用于可选 fast path。只有一个 epoch 内所有 update site 的 normalized group signature 完全兼容时，才使用 subgroup reduction；分析失败必须回退到 correctness baseline。
7. 第一版 participant domain 只支持 compiler-known contiguous `Range`。其他 domain 只有通用 fallback 或明确拒绝两种结果。
8. 第一版只支持 one allocation / one epoch 与内建 associative + commutative reductions。
9. 在进入 subgroup 设计前，先允许一个不需要通信的 LocalComplete plan：只有 update indices 与 parallel logical vars 逐维相等、loop 可安全复制、destination layout 已知时，才用 destination layout 物化 compact partial 并跳过 collective。

这个方案刻意把“先保证语义正确”和“再恢复 subgroup 性能”分开。前半部分不依赖复杂 ownership plan 合并，因此可以分阶段落地，也容易验证和回滚。

### 1.1 当前原型实现范围

`refactor/reducer-v2` 已经落地第一版 correctness baseline：

- 统一的 reduction identity/combine/codegen algebra；
- `local.reducer`、`ReducerInitOp`、`ReducerUpdateOp` 和 out-of-place `FinalizeReducerOp`；
- one allocation / one epoch lifecycle、普通 access、index bounds 和 replica-invariance verifier；
- 每 participant 一份完整 logical output array 的普通 `local` materialization；
- update-scoped parallel multiplicity marker 与 `REP == 0` lowering；
- compiler-known contiguous participant `Range` 检查，以及 CUDA/ROCm scalar collective emitter；
- direct-ownership LocalComplete planner、compact layout-shaped partial 和 collective-free finalize；
- sum/max/min/bitand/bitor/bitxor、logical seed 和 batch hint 的 scalar fallback；
- repo 内 reducer tests、example 和 docs 的 clean-break v2 迁移。

当前没有实现 Phase 5 subgroup fast path、general affine/Fragment ownership、batched finalize fast path、multiple epochs、自定义 combine 或 discontiguous participant set。非 GPU target 会明确拒绝 finalize；ROCm emitter 已通过启用 `USE_ROCM` 与 HIP stub 的完整编译验证，尚未做真实 ROCm 设备上的数值验证。

## 2. 为什么现有抽象无法继续扩展

本章对应以下 issue / PR：

- [Issue #2408](https://github.com/tile-ai/tilelang/issues/2408)：fully replicated reducer 在 `T.Parallel` 中重复计入 contribution，产生 silent wrong-code。
- [RFC #2897](https://github.com/tile-ai/tilelang/issues/2897)：将 reducer 重构为 ownership-safe deferred reduction 的长期设计讨论。
- [PR #2881](https://github.com/tile-ai/tilelang/pull/2881)：当前止血方案，只让每个 `T.Parallel` logical iteration 的 canonical replica 更新 fully replicated reducer。

### 2.1 最小 wrong-code 例子

Issue #2408 可以缩减为下面这个 `BD=8, threads=128` 的例子：

下面使用的是当前 v1 语法，只用于解释 root cause，不属于 v2 支持的 API。

```python
BD = 8
THREADS = 128

with T.Kernel(1, threads=THREADS):
    x_frag = T.alloc_fragment((BD,), T.float32)
    T.copy(x, x_frag)

    total = T.alloc_reducer(
        (1,), T.float32, op="sum", replication="all"
    )
    T.clear(total)
    for j in T.Parallel(BD):
        total[0] += x_frag[j]
    T.finalize_reducer(total)
```

layout inference 得到的关键关系是：

```text
x_frag: thread = rep * 8 + j, replicate = 16
total:  thread = rep,         replicate = 128
```

`x_frag[j]` 的一个 logical element 在 128 个线程中有 16 个 physical replicas。旧 lowering 生成的代码等价于：

```cpp
x_frag[0] = x[threadIdx.x & 7];
total[0] = 0.0f;
total[0] += x_frag[0];
total[0] = tl::AllReduce<tl::SumOp, 128>::run(total[0], workspace);
```

于是每个 `x[j]` 被 16 个线程分别加入 partial，最终得到：

```text
sum(thread = 0..127, x[thread % 8])
  = 16 * sum(j = 0..7, x[j])
```

当 `x = [1, 2, ..., 8]` 时，正确答案是 `36`，旧实现返回 `576`。

PR #2881 的短期修复变为：

```cpp
x_frag[0] = x[threadIdx.x & 7];
total[0] = 0.0f;
if ((threadIdx.x >> 3) == 0) {
  total[0] += x_frag[0];
}
total[0] = tl::AllReduce<tl::SumOp, 128>::run(total[0], workspace);
```

前 8 个 canonical threads 各贡献一个不同的 `x[j]`，其他 120 个 partial 保持 sum identity，因此 participant-wide `AllReduce<128>` 得到正确的 `36`。

这个修复也说明问题不在 `AllReduce<128>` 本身：baseline 可以继续让 128 个 participant 进入 collective；真正缺失的是 update site 上“一个 logical contribution 只能生效一次”的显式语义。

### 2.2 Reducer 仍然有必要

reducer 本身仍然有必要：它允许 kernel 在多个 tile/pipeline iteration 中先做 thread-local accumulation，最后只执行一次 cross-thread collective。用 `T.reduce` 替代会把 collective 放进每个 iteration；先物化完整 reduction domain 再统一 reduce 又可能消耗不可接受的 shared memory/register。因此需要重构的是 reducer 的语义表达，而不是删除 deferred reduction 能力。

### 2.3 当前抽象混合了哪些概念

当前 reducer 大致经过以下路径：

```text
alloc_reducer(local.fragment + reducer_info)
  -> T.clear/T.fill 充当 epoch 起点
  -> 普通 BufferStore 充当 update
  -> LayoutReducer 强制生成 Fragment layout
  -> ParallelOp 对 reducer layout 做特殊豁免
  -> FinalizeReducer 从 ReplicateExtent 推导 AllReduce width
```

这里混合了四类本应独立的信息：

| 信息 | 当前载体 | 问题 |
|---|---|---|
| 逻辑 combine 语义 | block annotation 中的 reducer op | update 本身仍是任意 store，无法可靠验证 |
| contribution multiplicity | 没有显式表示 | layout replication 会静默改变贡献次数 |
| partial 物理存储 | `FullyReplicated Fragment` | replication 通常暗示值等价，但 partial 在 finalize 前本来就允许不一致 |
| collective participant domain | reducer `ReplicateExtent()` | 丢失 update site、线程 offset、barrier domain 等信息 |

因此，`total` 被标成 fully replicated，并不代表每个线程的 `total[0]` 在 accumulation 期间相同。更准确地说，它们是尚未合并的 partial states。继续把这种状态塞进普通 Fragment 语义，会迫使 layout inference、race verifier、loop partition 和 finalize codegen 都增加 reducer 特例。

PR #2881 的 `fully_replicated_reducer_buffers` 是合理的止血方案，但它仍然依赖“识别写入了哪个特殊 buffer”。v2 应让 statement 自己表达 multiplicity，而不是让通用 loop lowering 反向猜测 buffer 的业务语义。

## 3. 目标与非目标

### 3.1 目标

- 每个动态逻辑 contribution 恰好进入 reducer 一次。
- update 可以位于多个 `T.Parallel`、多个 pipeline iteration 和受支持的条件分支中。
- correctness 不依赖成功合并多个 update site 的 ownership plan。
- reducer 在 finalize 前不可被普通读取或任意写入。
- identity、可选 seed、participant domain 和 final result layout 各自有明确语义。
- CUDA 和 ROCm 共用高层 correctness planning，只在 collective emission 层分叉。
- 所有优化分析失败时要么回退到正确实现，要么给出明确的 unsupported diagnostic，不能 silent wrong-code。
- v2 pipeline 中不存在 legacy syntax canonicalization 或双 lowering 路径。
- 删除 reducer 对 `LayoutReducer`、`ParallelOp`、`VerifyParallelLoop` 和 `PartitionLoop` 的 blanket exceptions。

### 3.2 初版非目标

- 自定义任意 lambda/结构化 monoid。
- 非交换 reduction 的稳定顺序语义。
- bitwise reproducible floating-point sum。
- 任意不连续 participant set。
- 一个 reducer allocation 上的任意多 epoch/reset。
- 第一版就得到最小寄存器占用或所有现有 subgroup 性能。

第一版只支持编译器内建的 associative + commutative combine。浮点 sum 延续现有 GPU reduction 的 reassociation 约定。

## 4. 精确语义模型

### 4.1 Reducer 是一个 epoch，不是普通 buffer

一个 reducer epoch 由以下信息定义：

```text
ReducerEpoch = {
  logical_shape,
  dtype,
  combine_op,
  identity,
  participant_domain,
  updates*,
  finalize,
}
```

生命周期为：

```text
Allocated --init--> Active --update*--> Active --finalize--> Finalized
```

第一版要求一个 allocation 只有一个 epoch。第二次 init、reset 或 finalize 后重新 update 都直接拒绝；multiple epochs 必须留给后续独立设计。

### 4.2 什么叫“一个逻辑 contribution”

一个 update 的动态逻辑实例由以下坐标共同决定：

- reducer epoch；
- update site；
- 外层 serial/pipeline loop iteration；
- 语义上的 `T.Parallel` logical indices；
- 没有被 layout 证明为 replica 的物理 participant lane。

`T.Parallel` lowering 可能让同一个 logical index 在多个 physical threads 上执行。layout 中的 `REP` 维度表示这些执行是同一个逻辑实例的物理副本，因此 reducer update 只能选择其中一个 representative。

需要特别说明：

- update 位于 `T.Parallel` 外时，不会自动把整个 CTA 去重；每个 active physical participant 的执行仍是独立 contribution。
- update 位于 `T.Parallel` 内时，只去除该 loop layout 明确产生的 replica 维度。
- 如果 update 的 predicate 或 value 显式依赖 `REP` 对应的物理 thread identity，且编译器无法证明 replica-invariant，初版应拒绝，而不是猜测用户意图。

### 4.3 四种 layout/domain 必须分开

| 概念 | 含义 | 是否用户可见 |
|---|---|---|
| logical reducer shape | update 使用的逻辑输出 index 空间 | 是 |
| partial storage | finalize 前每个 participant 保存 partial 的方式 | 否，compiler policy |
| final result layout | finalize 后结果由哪些线程/slot 持有 | 通过 destination layout 表达 |
| participant domain | 哪些物理线程初始化 partial、参与 collective 和 barrier | compiler-known execution domain |

baseline 中 partial storage 可以很保守：每个 participant 都持有完整 logical shape 的普通 `local` array。它可能增加寄存器压力，但语义简单，并且与当前 fully replicated reducer 的最坏空间开销相当。后续再用 segmented/sparse storage planner 缩减。

## 5. 已确定的用户 API

### 5.1 唯一支持的形态

```python
acc = T.alloc_reducer((M,), T.float32, op="sum")
T.reducer_init(acc)

for k_tile in T.Pipelined(...):
    for i, j in T.Parallel(M, K_TILE):
        T.reducer_update(acc[i], A_frag[i, j] * x_frag[j])

result = T.alloc_fragment((M,), T.float32)
T.finalize_reducer(acc, result)
T.copy(result, out)
```

`T.reducer_init` 必须由用户显式写出，因为 runtime initialization 必须发生在明确的 execution domain 中。`alloc_reducer` 只是 allocation builder，本身没有可执行语句位置。第一版不提供自动插入 init/finalize 的 structured sugar。

`T.finalize_reducer(acc, result)` 的 destination 也是必填参数。partial handle `acc` 在 finalize 后仍不可读取；所有结果消费都通过普通 destination buffer 完成。

### 5.2 Update 中不再写 `+=` 语义

combine op 属于 reducer definition，而不是每个 update：

```python
sum_acc = T.alloc_reducer(..., op="sum")
T.reducer_update(sum_acc[i], value)

max_acc = T.alloc_reducer(..., op="max")
T.reducer_update(max_acc[i], T.abs(value))
```

旧的 `acc[i] += x`、`acc[i] -= x`、`acc[i] = max(...)` 和 `acc[i] = min(...)` 都不做 canonicalization，直接诊断为 unsupported reducer access。若要给 sum reducer 贡献负值，必须显式写：

```python
T.reducer_update(sum_acc[i], -value)
```

`T.reducer_update` 的 indexed target 只是一种 point-region descriptor；除此之外的
任意 reducer `BufferLoad`、`BufferStore`、`T.clear` 或 `T.fill` 都是非法操作。

### 5.3 Identity 与 seed

identity 由 `combine_op + dtype` 决定，不再由用户传给 `T.fill`：

| combine | identity |
|---|---|
| sum / bitwise-or / bitwise-xor | `0` |
| max | lowest representable value |
| min | highest representable value |
| bitwise-and | all bits set |

如果用户需要逻辑 seed，必须单独表示，例如：

```python
acc = T.alloc_reducer(..., op="sum", seed=1.0)
```

seed 在 collective 之后对每个逻辑结果组合一次。它不能替代每线程 partial identity，否则会被 participant 数量放大。

`T.reducer_init` 始终使用 compiler-defined identity。`T.clear`/`T.fill` 不承担 reducer initialization 或 clamp semantics。

`batch`、vector width 和 workspace policy 都是 codegen hints，不应出现在语义 API 的核心位置。无效 hint 必须回退，不能改变参与 reduction 的元素集合。

## 6. IR 设计

### 6.1 `ReducerInfo`

建议删除独立的 `ReducerOpType`，复用并整理 `T.reduce` 的 `ReduceType`/algebra helper：

```text
ReducerInfo {
  ReduceType combine_type
  Optional<PrimExpr> seed
}
```

shape 和 dtype 可从 reducer `Buffer` handle 获取，不必重复保存。abs-sum/abs-max 应规范化为 contribution transform 加基础 combine，而不是继续扩展一套 reducer enum。

C++ analysis 中使用 `Buffer`、`Var`、`Range` 等 ObjectRef handle，并用 `ObjectPtrHash` / `ObjectPtrEqual` 或 `.same_as()` 做 identity 判断；不在跨函数数据结构中保存裸 `BufferNode*`。

### 6.2 虚拟 reducer storage

高层 allocation 使用：

```text
scope = "local.reducer"
```

该 scope 的 buffer 是 opaque handle：

- 只允许出现在 reducer init/update/finalize op 中；
- 不参与普通 Fragment layout propagation；
- 不允许普通 `BufferLoad`、`BufferStore`、`access_ptr` 或 alias；
- 必须在 backend codegen 前被 materialize 成真实 storage。

baseline materialization 将其映射为每线程普通 `local` buffer，而不是 `FullyReplicated Fragment`。这样 partial values 不需要满足 Fragment replication 的值等价假设。

### 6.3 First-class ops

推荐节点字段如下，最终 C++ 名称可在实现时调整：

```text
ReducerInitOp {
  Buffer reducer
}

ReducerUpdateOp {
  Buffer reducer
  Array<PrimExpr> logical_indices
  PrimExpr contribution
}

FinalizeReducerOp {
  Buffer reducer
  Buffer destination
  Map<String, ObjectRef> codegen_hints
}
```

约束：

- op type 从 allocation 的 `ReducerInfo` 解析；下游 planned op 可以复制该 handle，但 verifier 必须检查 identity 一致。
- update 是 effectful read-modify-write；effect/scheduling analysis 不能把多个 update 任意重排或删除。
- control-flow predicate 由外层 TIR 表示，不作为 update 字段重复保存。
- destination 必须存在，并且不能与 reducer partial handle alias。
- update contribution 可以读取普通 buffer，但不能写状态或包含 opaque effectful call。

建议三个 op 都注册为 opaque effect，并提供准确的 access regions：

| op | reducer access | destination access |
|---|---|---|
| init | full-region write | none |
| update | indexed read-write | none |
| finalize | full-region read | write |

这样 pipeline/scheduler 能看见依赖关系，但不会把 reducer handle 当成普通 Fragment access source。

### 6.4 Analysis-only plan

layout inference 后构造一个不暴露给用户的 epoch plan：

```text
ReducerEpochPlan {
  Buffer reducer
  ReducerInfo info
  Range participant_range
  std::vector<ReducerUpdateSite> updates
  PartialStoragePlan storage
  FinalizePlan finalize
}

ReducerUpdateSite {
  Span span
  Array<PrimExpr> logical_indices
  ParallelMultiplicity multiplicity
  Optional<ThreadGroupSignature> fast_path_signature
}
```

这些结构使用 IR handle 作为 identity key。建议把 planning 与 storage materialization 实现为同一个 PrimFunc pass 的两个阶段，使完整 plan 保持为 pass 内部 C++ structs。只有随后 `LowerTileOp` 确实需要的少数字段才写回 planned op；不要为了调试便利而过早扩大 public IR surface。

## 7. `T.Parallel` 与 update multiplicity

### 7.1 通用 marker，而不是 reducer buffer 参数

`ReducerUpdateOp::Lower` 先生成普通 combine store，但在 statement 外包一层通用语义 marker：

```text
AttrStmt(
  node = Integer(0),
  attr_key = attr::kParallelMultiplicity,
  value = IntImm(ParallelMultiplicity::kOncePerLogicalIteration),
  body = partial[index] = combine(partial[index], value),
)
```

`PartitionLoop` 已经通过 `InverseWithLevel` 得到当前 physical thread 对应的 logical indices 和 `REP`。它对 marker 做如下转换：

```cpp
if (rep == 0) {
  partial[index] = combine(partial[index], value);
}
```

转换后删除 marker。若当前 loop 的 replicate extent 为 1，则只删除 marker，不生成条件。

这样 `PartitionLoop` 的职责仍是“实现 parallel multiplicity”，而不是“知道哪些 buffer 是 reducer”。最终可以删除：

- `fully_replicated_reducer_buffers` 参数；
- `ReducerStoreGuarder` 的 buffer identity lookup；
- `LowerTileOp` 对 reducer store 的反向 var remap 扫描。

update 位于 `T.Parallel` 外时，没有 loop-created replica，marker 直接被 cleanup pass 去除，每个 active physical participant 正常贡献一次。

### 7.2 为什么 fragment 不需要特殊 update

普通 fragment work 和 reducer effect 有不同的 multiplicity：

```python
for i in T.Parallel(...):
    y_frag[i] = f(x_frag[i])
    T.reducer_update(acc[0], x_frag[i])
```

lowering 应为：

```cpp
// 每个 physical replica 都要填充自己的 fragment slot。
y_frag[...] = f(x_frag[...]);

// 同一个 logical i 只贡献一次。
if (rep == 0) {
  acc_partial[0] = combine(acc_partial[0], x_frag[...]);
}
```

被选中的 representative 是否能读到正确 `x_frag[i]`，由现有 fragment/layout compatibility proof 保证。若 contribution 中的 fragment read 与 loop layout 不兼容，就应在 layout inference 阶段报错；不需要为 fragment store 整体加 guard，也不能把整个 loop body 放到 `rep == 0` 下。

### 7.3 Predicate 规则

update 可以位于依赖 logical indices 的条件中：

```python
if j < valid_n:
    T.reducer_update(acc[i], value)
```

要求 predicate 在同一个 replica equivalence class 内不变。planner 应证明：

```text
predicate(logical_indices, rep = 0)
  == predicate(logical_indices, rep = r)
```

无法证明时，初版拒绝并指出 update span。finalize 不允许位于 representative guard 或 participant-divergent control flow 中，因为 collective/barrier 必须由完整 participant domain 到达。

## 8. Participant domain

### 8.1 明确表示

初版 participant domain 使用一个连续 `Range(min, extent)`：

```text
ParticipantDomain {
  Range thread_range
}
```

来源是 reducer epoch 所在的真实 execution scope，包括 producer/consumer warp specialization 产生的 non-zero thread offset，而不是 result Fragment layout。

初始化、partial storage、finalize 和 workspace addressing 必须使用同一个 epoch domain。每个 update site 的 execution domain 必须是 epoch domain 的子集；域外 participant 保持 identity。若多个 update site 无法嵌入同一个可支持的连续 domain，初版拒绝。

### 8.2 三个容易混淆的线程数量

| 值 | 含义 | baseline |
|---|---|---|
| reduction width | 一次 collective 实际组合多少 lanes | participant extent |
| barrier arrive count | 有多少线程到达对应 barrier | participant extent |
| workspace stride | 不同 batch/output 的 workspace 间距 | 由 participant storage policy 决定 |

subgroup fast path 中 reduction width 可以变小，但 barrier arrive count 不一定跟着变小。API 和 codegen helper 必须分别传递这些值，不能继续用一个 template argument 的来源隐式推断其他值。

对于当前 XOR butterfly 不支持的非 power-of-two width，允许的行为只有：

1. 使用通用 shared-memory reduction fallback；或
2. 在 capability check 中明确拒绝。

不能为了使用某个 fast path 而改变 contribution 集合。

## 9. Compiler pipeline

建议的长期 pass 顺序：

```mermaid
flowchart LR
    A[Frontend TIR] --> B[VerifyReducerEpochs]
    B --> C[VerifyParallelLoop]
    C --> D[LayoutInference]
    D --> E[PlanAndMaterializeReducers]
    E --> F[LowerTileOp / PartitionLoop]
    F --> G[VerifyReducerLowered]
    G --> H[Backend codegen]
```

frontend 直接生成 `local.reducer` allocation 和三个 first-class ops，不存在 legacy normalization stage。

### 9.1 `VerifyReducerEpochs`

使用结构化 control-flow state analysis 检查 lifecycle 和 effect legality。它不做 layout 决策。

这是 correctness pass，必须始终运行，不能受 optional data-race-check 配置控制。

除生命周期外，它还要直接拒绝 reducer 上的 `T.clear`、`T.fill`、普通 load/store、in-place finalize、缺失 destination 和第二个 epoch。

### 9.2 `LayoutInference`

- reducer handle 本身不作为 Fragment layout source。
- `ReducerUpdateOp` 的 contribution 中的普通 fragment reads 仍参与 loop layout inference。
- logical reducer indices 用于边界和 output projection 分析，但不要求 reducer partial 有 Fragment layout。
- `FinalizeReducerOp` 只对 destination 参与正常 layout inference。

### 9.3 `PlanAndMaterializeReducers`

建议初版实现成同一个 PrimFunc pass 内的 analyze + rewrite 两个阶段，避免为了在 pass 间传递完整 plan 而过早增加 public IR node。

Analyze 阶段：

- 从 init/finalize execution scope 确定 participant range。
- 要求 participant domain 可表示为 compiler-known contiguous `Range`。允许在新增 lanes 可执行 init/finalize 并始终提供 identity 时，安全扩大到一个已知连续 superset；否则拒绝。
- 为每个 update site 获取 enclosing parallel layout。
- 验证 replica-invariant predicate/value requirements。
- 默认选择 canonical unique-contribution plan。
- 可选计算并比较 `ThreadGroupSignature`。
- 固定选择每 participant 一份完整 logical output array 的 partial storage，并决定 finalize strategy，但不直接发 backend code string。

Rewrite 阶段：

- 根据 epoch plan 为每个 `local.reducer` 创建真实的普通 `local` partial buffer。
- 重写 allocation 以及 init/update/finalize 中的 reducer handle。
- 只把 LowerTileOp 必需的 participant range、selected strategy 和 group fields 写入 planned op；完整 epoch plan 留在 pass 内部。
- 此时不展开 update，也不发 collective，保证带 logical loop indices 的 update 仍能经过正常 loop partition。

### 9.4 `LowerTileOp` / `PartitionLoop`

- init lowering 为每个 participant 发出 local identity initialization。
- update lowering 生成 combine store 与 generic multiplicity marker。
- loop partition 将 marker 转成 `REP == 0` guard。
- 普通 fragment reads/writes 维持原有 lowering。
- finalize lowering 消费已经确定的 epoch plan；高层 plan 为 CUDA/ROCm 共用，最后的 collective emitter 可以 target-specific。
- pass 尾部删除位于 `T.Parallel` 外、因而不需要去重的 marker。

### 9.5 `VerifyReducerLowered`

- 确认不存在 `local.reducer` allocation。
- 确认不存在 reducer init/update/finalize op。
- 确认不存在未消费的 multiplicity marker。
- 确认 collective 不在 representative/update predicate 内。
- 违反任一条件都在进入 backend codegen 前报 compiler invariant error。

## 10. Correctness baseline

### 10.1 Initialization

每个 participant 的每个 partial slot 初始化为 identity：

```cpp
for (logical_output : reducer_shape) {
  partial[logical_output] = identity;
}
```

不能对 init 使用 `REP == 0`，因为未执行 init 的 participant 仍会进入最终 collective，它们必须提供 identity。

### 10.2 Update

每个 update site 独立 lowering：

```cpp
if (site_rep == 0 && site_predicate) {
  partial[index] = combine(partial[index], contribution);
}
```

多个 update site 即使分别来自 8-thread、16-thread 或完全不同的 loop layout，也不需要合并 ownership plan。每个 site 只保证自己的 logical contributions 唯一即可。

### 10.3 Finalize

对每个 logical output：

```cpp
value = participant_wide_allreduce(partial[index]);
value = combine_seed_once_if_present(value);
write_to_destination_according_to_result_layout(value);
```

没有贡献的 participant 持有 identity，因此不会改变结果。collective 必须位于所有 update predicates 和 replica guards 之外。

### 10.4 最小错误示例

对于 `BD=8, threads=128`：

- loop layout 的 replica extent 是 16；
- 只有 `rep == 0` 的 8 个线程分别贡献 `x[0] ... x[7]`；
- 其余 120 个 partial 为 0；
- `AllReduce<128>` 得到 `sum(x)`，而不是 `16 * sum(x)`。

这里 `128` 是 correctness baseline 的 participant-wide width，不再从 reducer result replication 猜出来。

## 11. Optional subgroup fast path

baseline 正确后，再为每个 update site 生成 normalized signature：

```text
ThreadGroupSignature {
  Range participant_range
  normalized group_id(thread)
  normalized lane_id(thread)
  Array<ThreadReduceStep> reduction_steps
  logical_output_projection
  normalized uniform predicate
}
```

signature 不能从 reducer result layout 反推。建议从 update site 的 enclosing loop layout 与 `logical_indices` projection 出发：

1. 找出被 reducer output indices 保留的 spatial axes，以及被投影掉的 reduction axes；
2. 规范化 loop layout 中 thread 到 group/lane 的映射；
3. 用 contribution 中的 fragment reads 做 layout compatibility proof，而不是把任意一个 source fragment 当作整个 epoch 的 ownership source；
4. 任一步无法规范化，就不给该 site 生成 signature，并回退 canonical baseline。

epoch 只有在以下条件全部满足时才能采用 subgroup plan：

- 每个 update site 都成功生成 signature；
- participant range 相同；
- group/lane partition 完全相同；
- reduction steps 的 extent、scale、lower factor 完全相同；
- logical output projection 相同；
- predicate compatibility 可证明；
- backend 支持对应 width、offset、barrier 和 workspace policy。

第一版只做 exact normalized equality，不尝试求多个 plan 的并、交、最大公约数或等价变换。

结果：

| planner outcome | update | finalize |
|---|---|---|
| CanonicalUnique | `REP == 0` | participant-wide collective |
| CompatibleGroups | 保留每个 group 的等价 partial | group-local collective |
| LocalOnly | 普通 local combine | copy/seed，无 collective |
| UnsupportedDomain | 不生成代码 | source diagnostic |

`ReduceOwnershipPlan` 可以复用来构造 signature 的局部字段，但不能把第一个 update site 的 plan 直接保存到 reducer 并假设它代表整个 epoch。

## 12. Verifier 规则

### 12.1 Lifecycle

- update 必须被 init dominate，并被 finalize post-dominate。
- init/finalize 的 participant domain 必须一致。
- finalize 后禁止 update。
- 初版禁止 missing finalize、double finalize 和 reset/multiple epochs。
- init/finalize 不能只出现在条件的一侧。
- finalize 必须在 participant-uniform control flow 中。

### 12.2 Access legality

- 在任何状态下都禁止普通 reducer load/store；reducer handle 只能作为三个 first-class ops 的参数。
- Finalized 结果通过 destination 读取，不再读取 partial handle。
- 禁止 reducer `access_ptr`、match buffer、alias、return 或传给未知 extern。
- `T.clear`、`T.fill`、in-place finalize 和缺失 destination 都直接拒绝。

### 12.3 Update legality

- logical index rank 与 shape 一致且可证明不越界，或由标准 safe-access 机制保护。
- contribution dtype 可安全转换到 accumulator dtype。
- reducer op 必须来自 allocation 上的同一个 `ReducerInfo`，且属于第一版内建 commutative op set。
- contribution 无 side effect。
- replica predicate/value compatibility 可证明。
- 允许 update 位于 serial/pipeline loops 和普通 logical predicate 中。

### 12.4 Control-flow implementation 建议

初版可以用 conservative structured analysis，不必立刻构建完整 CFG：

- `SeqStmt` 按顺序传递 reducer state；
- `IfThenElse` 分别分析两个分支并要求出口 state 相同；
- update 不改变 state，允许只出现在一个分支；
- init/finalize 导致分支出口 state 不同则拒绝；
- loop body 出口 state 必须等于入口 state，除非后续明确支持 loop-local 完整 epoch。

## 13. 分阶段实施计划

下面的 phase 是开发与 review 顺序，不代表要向用户发布 v1/v2 双路径。Phase 2 和 Phase 3 可以先落不可达或测试专用的基础设施，但完成 Phase 4 cutover 时只保留 v2 public contract；整个过程不引入 runtime compatibility switch。

### Phase 0：止血并建立回归基线

状态：PR #2881 已覆盖 replicated `T.Parallel` update 的主要 wrong-code。

还需要：

- 修复或禁用 #2623 的 batched finalize wrong-code 路径。
- 为 #2408、mixed update widths、ordinary fragment write + reducer update 建立 numerical tests。
- 将 `fully_replicated_reducer_buffers` 明确标记为临时参数。

退出条件：已知路径要么正确，要么明确拒绝，不再 silent wrong-code。

### Phase 1：统一 reduction algebra

建议先做一个纯重构 PR：

- `FinalizeReducer` 改用 `ReduceType`。
- 抽取共享的 `IdentityFor(dtype, type)`、`MakeCombine(type, lhs, rhs)`、backend reducer name/capability helpers。
- 删除 sum/max/min 的重复 enum/string table。

退出条件：`T.reduce` 和当前 reducer implementation 的行为、codegen、测试结果不变；该 PR 不引入兼容层。

### Phase 2：引入 first-class reducer IR

- 新增 `ReducerInitOp`、`ReducerUpdateOp` 和新版 `FinalizeReducerOp`。
- 新增 Python `T.reducer_init` / `T.reducer_update` / out-of-place `T.finalize_reducer` API。
- 新增 `local.reducer` 虚拟 scope。
- 添加 IR printer/FFI/reflection/access-region 支持。
- 增加 mandatory `VerifyReducerEpochs`，直接拒绝旧式 reducer load/store、`T.clear`/`T.fill`、in-place finalize 和 multiple epochs。
- 第一版只注册内建 associative + commutative `ReduceType`。

退出条件：新 API 能通过 parser、IR round-trip 和 verifier tests；所有生命周期与非法 access diagnostics 都有 source span。

### Phase 3：canonical ownership-safe baseline

- 实现 `PlanAndMaterializeReducers`。
- participant domain 只接受 compiler-known contiguous `Range`；可证明安全时扩大到连续 superset，否则拒绝。
- 每个 participant 固定物化一份完整 logical output array 的普通 `local` storage。
- update lowering 产生通用 parallel multiplicity marker。
- `PartitionLoop` 消费 marker 并实现 `REP == 0`。
- baseline 始终使用 participant-wide collective。
- finalize 必须写入独立 destination，其 layout 走普通 layout inference。
- CUDA/ROCm 共享 epoch plan，backend 只负责 capability 与 emitter。
- `VerifyReducerLowered` 在 backend codegen 前检查虚拟对象和 marker 均已消失。

退出条件：多 update site、不同 loop layout、pipeline、独立 destination 和 offset contiguous Range 的 numerical tests 全部正确。

### Phase 4：全仓切换并删除 v1

- 将 repo 内 tests、examples 和 docs 一次性改为显式 init/update/out-of-place finalize。
- 删除 `replication=` 参数、旧 `T.finalize_reducer(reducer)` 签名以及 reducer `T.clear` 文档。
- 删除 `ReducerRepType` 和 reducer-specific `LayoutReducer` 逻辑。
- 删除 `ParallelOp` 跳过 fully replicated reducer 的特殊逻辑。
- 删除 `VerifyParallelLoop` 的 reducer store blanket exemption。
- 删除 `fully_replicated_reducer_buffers`、`ReducerStoreGuarder` 和相关反向 var-remap 扫描。
- 禁止 core IR 中出现 arbitrary reducer `BufferStore`。

退出条件：全仓搜索不到 v1 reducer API 或 metadata consumer；只有 v2 pipeline 能构造 reducer。

### Phase 4.5：conservative LocalComplete optimization

- 复用 finalize destination Fragment layout 作为 storage layout 与 ownership certificate。
- 第一版只接受 `Parallel(M) -> output[M]` 的逐维 identity mapping；inner serial loops 可以继续向同一 output 累加。
- 只在 loop 可安全复制时用 destination layout 覆盖原 Parallel layout；ordinary stores、Fragment/local loads、非 pure calls 或 layout conflicts 全部 fallback。
- materialized partial shape 改为 destination layout 的 `OutputShape()`。
- update 在所有 destination replicas 上执行，并使用独立的 partition-required marker；不能复用会产生 `REP == 0` 的 multiplicity marker。
- finalize 直接把 local partial slots 写到 destination physical slots，不生成 AllReduce、barrier 或 workspace。
- 带 reducer effect marker 的 loop 在 logical layout 阶段禁止普通 vectorization；physical
  indices 物化后只能由 reducer-aware vectorizer 证明 target lanes 独立，或显式构造
  vector partial 与 horizontal combine。证明失败时保留 scalar loop-carried combine。

退出条件：M8/M32/M128 independent-output cases 生成 compact local storage 且没有 collective；seed、inner serial reduction 与 fallback numerical tests 全部通过；关闭证明路径时仍得到 canonical baseline。

### Phase 5：exact-signature subgroup optimization

- 实现 `ThreadGroupSignature` normalization/equality。
- 复用 `ThreadReduceStep` 生成 reduction steps。
- 只有全 epoch exact-compatible 时选择 subgroup plan。
- debug mode 可强制 canonical baseline，并比较两种 lowering 的 numerical output/codegen invariants。
- `batch`、vector width 和 workspace policy 只作为不改变语义的 auto/codegen strategy。

退出条件：fast path 关闭时所有程序仍正确；开启后性能恢复且结果与 baseline 一致。

## 14. 建议的 PR 拆分

| PR | 主要内容 | 是否改变语义 | 可独立回滚 |
|---|---|---:|---:|
| A | shared reduction algebra | 否 | 是 |
| B | first-class IR/API + mandatory verifier | 新增 v2 contract | 是 |
| C | full local storage + multiplicity marker | 实现 canonical baseline | 是 |
| D | contiguous participant plan + out-of-place CUDA/ROCm finalize | 完成 v2 correctness | 是 |
| E | migrate in-tree users + delete v1 implementation | clean-break cutover | 整体回滚 |
| F | conservative LocalComplete compact-storage fast path | 仅性能 | 是，可强制 baseline |
| G | exact-signature subgroup fast path | 仅性能 | 是，可强制 baseline |

### 14.1 预计代码落点

下面是建议的职责划分，不要求最终文件名完全一致：

| 位置 | 主要改动 |
|---|---|
| `src/op/reduce.h` 及共享 helper | 统一 `ReduceType`、identity、combine 和 capability contract |
| `src/op/deferred_reducer.{h,cc}` | reducer init/update/finalize op、reflection、access regions、target registry |
| `src/transform/verify_reducer.cc` | lifecycle、access、control-flow 和 alias verifier |
| `src/transform/plan_reducer.cc` | epoch planning、participant range、storage materialization、group signature |
| `src/op/parallel.{h,cc}` | 将 update target 与普通 Fragment access 分开，只分析 contribution inputs |
| `src/transform/loop_partition.{h,cc}` | 消费 generic multiplicity marker，删除 reducer buffer 参数 |
| `src/transform/lower_tile_op.cc` | planned reducer op lowering 与最终 invariant cleanup |
| `src/backend/common/op/{reduce,finalize_reducer}.h` | 共享 collective contract，分离 width/barrier/workspace 参数 |
| `src/{cuda,rocm}/op/finalize_reducer.cc` | target capability 与最终 emitter |
| `tilelang/language/{allocate,reduce_op}.py` | 唯一 v2 API 与非语义 codegen hints |
| 各 target `pipeline.py` | 插入 mandatory verify/plan stages |
| `testing/python/{language,transform}` | numerical、IR、diagnostic 与 differential tests |

Phase 5 引入 fast path 后，可以保留一个只用于测试和 debug 的内部 strategy override：

```text
tl.reducer_strategy = "canonical" | "auto"
```

它只能在 canonical baseline 与优化路径之间选择，不得恢复 v1 语义，也不应成为长期用户 API。

## 15. 测试与验收矩阵

### 15.1 Numerical correctness

- `BD=8, threads=128` 返回 36 而不是 576。
- #2408 非整除 GEMV tile width。
- 同一 reducer 的 8-thread 与 16-thread 两个 update sites。
- 多个 exact-compatible update sites。
- ordinary replicated fragment write 与 reducer update 位于同一 loop。
- predicated update、空 contribution domain 和边界 tile。
- 多个 pipeline iterations，且只 finalize 一次。
- scalar 与多元素 output shape。
- sum/max/min/bitwise builtins 和支持的 accumulator dtypes。
- seed 恰好应用一次。
- scalar/batched/vectorized finalize 结果一致。

### 15.2 Participant domains

- full CTA。
- warp-only。
- non-zero offset contiguous consumer range。
- partial CTA 下多个完整 warp groups。
- unsupported non-power-of-two/discontiguous domain 给出 diagnostic 或走通用 fallback。
- CUDA 与 ROCm 对共享 plan 产生等价语义。

### 15.3 Structural IR/codegen checks

- frontend 直接生成三个 first-class ops，不存在 reducer `BufferStore` canonicalization stage。
- 每个 reducer 恰好有一个显式 init 和一个带独立 destination 的 finalize。
- update marker 只包住 reducer combine，不包住 ordinary fragment work。
- marker 在 backend codegen 前全部消失。
- final collective 不位于 `REP == 0` 或 update predicate 内。
- reduction width 与 barrier arrive count 独立断言。
- `local.reducer` 在 backend codegen 前全部 materialize。
- fast path 分析失败时确定性回退 baseline。
- LocalComplete independent-output case 使用 layout-shaped compact partial，且不存在 AllReduce、named barrier 或 workspace。
- LocalComplete inner serial reduction 不被普通 vectorization 拆成同址并行 stores；可选
  packed 路径必须显式构造 vector partial 与 horizontal combine，并在证明失败时回退。

### 15.4 Diagnostics

- missing/double init。
- read before finalize。
- update after finalize。
- missing/double finalize。
- second epoch/reset。
- `T.clear`/`T.fill` 或任意 reducer load/store。
- in-place finalize、missing destination 或 destination alias。
- unsupported/non-commutative op。
- invalid index/dtype。
- effectful contribution。
- divergent finalize。
- replica-dependent predicate/value。
- alias/access pointer/escape。
- incompatible participant scopes。

### 15.5 Differential testing

为同一个 kernel 提供两种 lowering：

```text
canonical baseline vs. subgroup fast path
```

随机生成小 shape、update site 数量、loop extent、predicate 和 participant offset，比较：

- 编译是否接受/拒绝一致；
- numerical result；
- sanitizer/race behavior；
- barrier/workspace invariants。

## 16. 风险与控制手段

| 风险 | 控制手段 |
|---|---|
| baseline 的完整 per-thread output array 增加寄存器压力 | 第一版明确接受该成本并建立性能基准；后续 segmented storage 只能作为不改变语义的优化 |
| new op 穿过 layout/pipeline pass 时被错误重排 | opaque effect + access regions + structural tests |
| warp specialization participant scope 不一致 | epoch verifier 比较显式 `Range`，不从 result layout 猜 |
| 非连续 participant 被错误扩大 | 只有新增 lanes 可执行完整 init/finalize 且恒为 identity 时才允许 contiguous-superset fallback，否则拒绝 |
| subgroup signature 误判相等 | 第一版只接受 canonical form 的 exact equality；debug 强制 baseline |
| clean break 影响现有 kernel | Phase 4 同一批迁移全部 in-tree users，提供 release note，但不保留 compiler compatibility path |
| generic multiplicity marker 泄漏到后端 | dedicated cleanup + final invariant check |

## 17. 已拍板的 v2 边界

### 17.1 不支持 legacy 写法

v2 只支持 `alloc_reducer`、显式 `reducer_init`、first-class `reducer_update` 和带独立 destination 的 `finalize_reducer`。

不实现 store canonicalizer、in-place adapter、deprecation window 或 v1/v2 lowering switch。旧式 API 由调用方直接迁移。

### 17.2 Init 必须显式

core IR 与用户 API 都必须显式出现 `T.reducer_init`。第一版不提供隐藏 init 的 structured sugar。

理由：identity initialization 需要确定 runtime execution/participant domain，不能只靠 allocation metadata 表示。

### 17.3 第一版 canonical partial storage

canonical fallback 是每 participant 一份完整 logical output array 的普通 `local`
storage。LocalComplete 只是在独立证明成立时替换物理表示；证明失败必须回到这个
fallback。

理由：最容易证明正确，也完全摆脱 `FullyReplicated Fragment` 的值语义冲突。寄存器优化放到独立 storage planner。

### 17.4 第一版 participant set

只支持 compiler-known contiguous `Range`。能证明新增 lanes 始终提供 identity 且安全到达 collective 时，可以扩大到 contiguous superset；其他情况拒绝。

理由：当前 AllReduce/barrier/workspace 都天然依赖连续 range。先让 contract 明确，再扩展 mask/discontiguous sets。

### 17.5 第一版 reducer 能力边界

只支持 one allocation / one epoch 与 compiler 内建的 associative + commutative reductions。第二次 init/reset 和自定义或非交换 combine 都拒绝。

先稳定 one allocation / one epoch 与内建 commutative reductions，再分别写 RFC，避免 verifier、identity ABI 和 backend capability 同时膨胀。

## 18. 最终应满足的不变量

重构完成后，以下断言应始终成立：

1. `ReducerUpdateOp` 是 contribution multiplicity 的唯一来源。
2. layout replication 不能静默改变 logical contribution count。
3. reducer partial 在 finalize 前不是普通 Fragment value。
4. 每个 update site 的 correctness 可以独立证明，不依赖另一个 site 的 ownership plan。
5. participant range 来自 execution domain，不来自 result layout。
6. barrier participant count 与 reduction group width 是独立量。
7. subgroup optimization 失败时存在正确 baseline。
8. `PartitionLoop` 不包含 reducer buffer 特例。
9. 每个 reducer 恰好有一个显式 init、一个 epoch 和一个带独立 destination 的 finalize。
10. reducer handle 永远不存在普通 load/store、clear/fill 或 in-place finalize。
11. 第一版 participant domain 必须解析为已知连续 `Range` 或安全的连续 superset。
12. backend codegen 看不到 `local.reducer`、first-class reducer op 或 multiplicity marker。

本设计可以压缩成下面这句：

> first-class update 定义“贡献了什么、贡献几次”；epoch plan 定义“哪些线程参与”；storage/layout/codegen 只负责“把这个语义高效实现出来”。
