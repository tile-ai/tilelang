# Reducer v2 当前实现状态

- 分支：`refactor/reducer-v2`
- 记录日期：2026-08-10
- 当前阶段：correctness baseline、conservative LocalComplete 和受限 projected partial groups 已实现
- 长期设计：[Reducer v2 重构设计方案](./reducer_v2_refactor_plan.md)
- 相关讨论：[Issue #2408](https://github.com/tile-ai/tilelang/issues/2408)、[RFC #2897](https://github.com/tile-ai/tilelang/issues/2897)、[PR #2881](https://github.com/tile-ai/tilelang/pull/2881)

## 1. 当前结论

这个分支已经完成了 reducer v2 的第一版 correctness baseline，并在其上加入两个可回退
的 physical plans：LocalComplete 和受限 projected partial groups。

当前实现遵循下面几个已经确定的原则：

1. reducer 是一个 deferred reduction epoch，不是普通 Fragment。
2. 用户必须显式写 `T.reducer_init`、`T.reducer_update` 和 `T.finalize_reducer`。
3. 无法证明更强 ownership 时，每个 participant 都分配一份完整 logical output array，作为 canonical local partial state。
4. `T.Parallel` layout 产生的重复执行在 update site 消除，不能在 finalize 阶段猜测 contribution multiplicity。
5. finalize 是 out-of-place 的，所有 participants 先完成 collective，只有 destination owner 写入 Fragment。
6. 每个 reducer allocation 当前只允许一个 epoch；同一个 kernel 可以有多个相互独立的 reducer allocations。
7. 当前只依赖 compiler-known contiguous participant `Range`，不支持 discontiguous participant set。
8. compiler-known scalar/row reduction 可以选择 projected partial group；其他情况仍回退 canonical baseline。
9. 对 `Parallel(M) -> output[M]` 这类可直接证明的独立输出，planner 可以选择 LocalComplete plan：每个 destination replica 独立计算自己的 local slots，不执行 collective。

因此，目前的优先级是：

```text
语义可描述
  -> 编译器可以验证
  -> 所有无法证明安全的情况明确拒绝
  -> 先用通用 lowering 保证正确
  -> 将来再增加可回退的优化路径
```

## 2. 用户可见的 v2 contract

### 2.1 基本写法

```python
with T.Kernel(1, threads=128):
    total = T.alloc_reducer((1,), T.float32, op="sum")
    T.reducer_init(total)

    for i in T.Parallel(8):
        T.reducer_update(total[0], value[i])

    result = T.alloc_fragment((1,), T.float32)
    T.finalize_reducer(total, result)
```

三个操作的语义分别是：

| 操作 | 语义 |
|---|---|
| `T.reducer_init(acc)` | 用 reduce identity 初始化 planner 选择的 physical partial storage |
| `T.reducer_update(acc[indices], value)` | 向一个 logical output element 贡献恰好一次 `value` |
| `T.finalize_reducer(acc, dst)` | 按 physical plan 完成 reduction，并写入独立 destination Fragment |

### 2.2 allocation metadata

```python
T.alloc_reducer(shape, dtype, op="sum", seed=None)
```

它生成的不是 `local.fragment`，而是不可普通访问的虚拟 scope：

```text
local.reducer
```

allocation 所在 SBlock 同时记录：

```text
reducer_info = {
  reducer.data: {
    "op": ...,
    "seed": ...,
  }
}
```

当前 public reducer API 支持：

- `sum`
- `max`
- `min`
- `bitand`
- `bitor`
- `bitxor`

bitwise reductions 要求 accumulator dtype 是 integer 或 bool。

### 2.3 seed 的语义

`seed` 不是每个 participant 的初值，否则会在 collective 中被重复计算。

当前 lowering 是：

```text
participant partial 从 identity 开始
  -> participant-wide collective
  -> collective result 与 seed combine
  -> destination owner store
```

所以 seed 在 logical reduction algebra 中只合并一次。

LocalComplete plan 没有 cross-thread collective，但同样在 update 全部完成后才
combine seed。每个 physical destination replica 各自执行一次，随后普通 Fragment
ownership 只消费对应 replica；因此每个 logical result 观察到的 seed 仍然恰好一次。

### 2.4 明确不支持的 legacy 写法

下面这些写法不再做 canonicalization，直接拒绝：

```python
acc = T.alloc_reducer(..., replication="all")
T.clear(acc)
acc[i] += value
T.finalize_reducer(acc)  # 缺少独立 destination
```

同样禁止：

- reducer `BufferLoad`
- reducer `BufferStore`
- `T.fill(reducer, ...)`
- reducer `access_ptr`
- in-place finalize
- finalize 后继续 update
- 第二次 init/reset

## 3. 编译流水线

当前 reducer 的主要编译阶段如下：

```text
Frontend
  local.reducer allocation
  reducer_info metadata
  ReducerInit/Update/Finalize calls
        |
        v
LegalizeNegativeIndex
        |
        v
VerifyReducerEpochs
  lifecycle / access / bounds / side-effect checks
        |
        v
普通高层 transforms
  warp specialization / pipeline planning / software pipeline
        |
        v
LayoutInference
  只为 Fragment、Shared 和 T.Parallel 推导 layout
  reducer handle 本身不参与 Fragment layout inference
        |
        v
PlanAndMaterializeReducers
  再次验证 epoch
  检查 replica invariance
  local.reducer -> full local partial array
  为三个 first-class calls 写入 planned annotations
        |
        v
LowerTileOp
  init -> identity stores
  update -> combine store + optional multiplicity marker
  finalize -> target-specific collective lowering
        |
        v
PartitionLoop
  multiplicity marker -> REP == 0 guard
        |
        v
VerifyReducerLowered
  确认 reducer-only IR 已全部被消费
        |
        v
CUDA / ROCm codegen
```

### 3.1 为什么验证要执行两次

第一次 `VerifyReducerEpochs` 尽早检查 frontend contract，给用户更直接的 source diagnostic。

第二次验证位于 `PlanAndMaterializeReducers` 内部。原因是第一次验证之后还有 warp specialization、pipeline planning、software pipeline 和 layout inference。这些 pass 可能改变 control-flow nesting 或 participant scope，因此不能假设 frontend 时合法的 lifecycle 到 materialization 时仍然合法。

### 3.2 为什么 materialization 位于 LayoutInference 之后

`local.reducer` 故意不拥有 Fragment layout。LayoutInference 只需要知道：

- `T.Parallel` logical iteration 如何映射到 physical threads；
- contribution 使用的 Fragment values 如何映射；
- destination Fragment 的 owner 和 local index 是什么。

等这些信息稳定后，planner 才把 reducer 物化成普通 `local` full array。这样 reducer storage 不会反过来影响 loop layout，也不会再出现“先给 reducer 推一个 fully replicated Fragment layout，然后把这个 layout 当成 reduction semantics”的问题。

## 4. 新增的 first-class IR

实现位于：

- `src/op/deferred_reducer.h`
- `src/op/deferred_reducer.cc`

### 4.1 `ReducerInfo`

`ReducerInfo` 保存 allocation-level 的 reduction definition：

```text
combine_type
optional seed
```

update site 不能自行更换 reduce op。init、所有 updates 和 finalize 都从同一个 allocation metadata 获得 combine type。

### 4.2 `ReducerInitOp`

高层 op 持有 reducer handle。planner 写入 combine type 后，它 lower 成对完整 local partial array 的 serial initialization：

```cpp
for each logical output index:
  partial[index] = identity(op, dtype);
```

### 4.3 `ReducerUpdateOp`

高层 op 显式保存：

```text
reducer
logical_indices
contribution
combine_type
parallel_once
```

普通 lowering 是：

```cpp
partial[index] = combine(partial[index], contribution);
```

如果 update 位于 `T.Parallel` 中，planner 会设置 `parallel_once=true`，lowering 再为这个 effect statement 包一层通用 multiplicity marker。

### 4.4 `FinalizeReducerOp`

finalize 同时持有 reducer 和 destination：

```text
reducer
destination
combine_type
optional seed
batch hint
```

C++ IR 层会检查 reducer 和 destination 都是完整 region，并要求 dtype 一致。target-independent op 最终通过 registry 分发到 CUDA 或 ROCm emitter。

### 4.5 access regions

三个 ops 都注册成 opaque tile operations，并显式声明 access region：

| op | reducer access | destination access |
|---|---|---|
| init | write | 无 |
| update | read/write | 无 |
| finalize | read | write |

这些 access regions 让通用 compiler infrastructure 仍然能看到 effect ordering，同时避免把 reducer update 伪装成普通 Fragment `BufferStore`。

## 5. Reduction algebra 统一

`src/op/reduce.h` 和 `src/op/reduce.cc` 新增了共享 helpers：

- `IsBuiltinCommutativeReduceType`
- `ReduceTypeName`
- `MakeReduceIdentity`
- `MakeReduceCombine`
- `ReduceCodegenName`

普通 `T.reduce` backend 和 reducer v2 共用这些定义。这样 identity、scalar combine 和 GPU runtime reducer 名称不再各自维护一套 switch。

这里同时修正了 floating-family 判断，使 `bfloat16`、float8、float6 和 float4 不会被错误地当成非浮点类型。

普通 `T.reduce` 仍然保留自己的 NaN-propagating max/min 特殊 lowering；reducer v2 第一版没有暴露 NaN propagation policy。

## 6. Verifier 当前检查什么

实现位于 `src/transform/reducer/verify_reducer_epochs.cc`。

### 6.1 metadata 与 allocation

collector 要求：

- 每个 `local.reducer` allocation 都有 reducer metadata；
- 每份 reducer metadata 都对应一个 `local.reducer` allocation；
- 同一个 storage `Var` 不能重复 allocation 或重复定义 metadata；
- metadata 不能包含 legacy `rep` 字段；
- reduce op 必须属于 compiler 内建 commutative set。

### 6.2 lifecycle state machine

每个 allocation 使用下面的状态机：

```text
Allocated
  -- T.reducer_init --> Active
  -- T.reducer_update --> illegal
  -- T.finalize_reducer --> illegal

Active
  -- T.reducer_update --> Active
  -- T.finalize_reducer --> Finalized
  -- T.reducer_init --> illegal

Finalized
  -- any reducer operation --> illegal
```

函数结束时，每个 reducer 必须处于 `Finalized`。

当前 init 和 finalize 必须位于 participant-uniform top-level control flow。把它们放在 `For` 或 `IfThenElse` 内都会被保守拒绝。update 可以位于 serial/pipeline loop 和普通 logical predicate 中。

### 6.3 普通 access

verifier 会拒绝 reducer 的：

- `BufferStore`
- `BufferLoad`
- `tvm_access_ptr`
- 通过 contribution 间接读取 reducer state

contribution 允许读取普通 state，但不能写 state或调用 opaque effect。

### 6.4 logical output bounds

update 的 index rank 必须等于 reducer rank，每个 index 必须能被 `arith::Analyzer` 证明位于：

```text
[0, reducer.shape[dimension])
```

分析会绑定 enclosing `For` ranges、thread extents 和 `IfThenElse` constraints。当前无法证明安全的动态 index 会被拒绝，而不是冒险生成代码。

### 6.5 replica invariance

只有当 `T.Parallel` layout 的 `ReplicateExtent > 1` 时，update 才需要消除 physical replicas。

在这种情况下，planner 要求：

- logical indices 不依赖 physical `threadIdx`；
- contribution 不依赖 physical `threadIdx`；
- enclosing predicate 和 serial-loop bounds 不依赖 physical thread；
- contribution 不读取普通 thread-private `local`/`local.var` value。

canonical baseline 中 Fragment loads 可以保留，因为 Parallel layout inference 已验证
logical Fragment access 与 loop layout compatible。Global/Shared load 也可以保留，
只要地址表达式本身不依赖 physical replica。LocalComplete 第一版会改写 loop layout，
因此只接受可安全重复的 Global/Shared loads；带普通 Fragment/local load 的 loop
保守 fallback。

对 ordinary local value 的规则目前是保守的：即使用户知道所有 threads 写入了同一个值，只要 compiler 没有对应证明，它仍会拒绝 replicated update。

### 6.6 lowered invariant

`VerifyReducerLowered` 在 backend codegen 前确认不存在：

- `local.reducer` allocation；
- `reducer_info` metadata；
- `ReducerInitOp`、`ReducerUpdateOp`、`FinalizeReducerOp`；
- 未消费的 parallel multiplicity / partition-required marker。

它的作用是防止新 IR 被某个遗漏的 pipeline/backend 静默忽略。

## 7. Physical materialization plans

### 7.1 FullParticipant baseline

当前 baseline 为每个 participant 分配完整 logical output shape：

```text
high-level:
  local.reducer partial[M]

materialized:
  local partial[M] per participant
```

例如 reducer shape 是 `(128,)`、CTA 有 256 threads，则概念上每个 thread 都拥有自己的 `partial[128]`。

这不是性能最优方案，但语义非常直接：

1. init 将本 participant 的全部 128 个元素设成 identity；
2. update 只更新本 participant 被分配到的 logical contributions；
3. finalize 对每个 logical output index 合并所有 participant partials。

这种表示完全不依赖 `ThreadReduceStep` 或某一个 update site 的 ownership plan，因此 mixed update layouts 不会让 reducer 自身获得互相冲突的 Fragment layout。

代价也很明确：

- local/register footprint 较大；
- init 和 finalize 都要遍历完整 output array；
- 大 output shape 可能增加 register pressure 或 local-memory spill；
- finalize 当前会生成较多 scalar collectives。

这些是当前明确接受的 correctness-first tradeoff。

### 7.2 LocalComplete fast path

当 finalize destination 已有确定的 Fragment layout，planner 会尝试把它同时用作
partial storage layout 和 ownership certificate。第一版只在下面条件全部成立时选择
LocalComplete：

- 每个 update 都位于已知 `T.Parallel` root 中；
- reducer logical shape、destination layout input shape 和 parallel loop shape 一致；
- update indices 与 parallel logical vars 逐维相等；
- participant/thread/replicate/output extents 都是 compiler-known；
- loop 中没有 ordinary BufferStore、thread-private Fragment/local load 或非 pure call；
- 同一 parallel root 上的所有 LocalComplete candidates 请求完全相同的 layout。

planner 随后把 parallel root 改成 destination layout。这样 `M < threads` 时并不是只让
一个 physical owner 计算后再广播，而是让 destination 的每个 physical replica 都执行
相同 logical contribution：

```text
logical reducer[M]
  -> destination layout: local slots S, replicate R
  -> materialized partial[S] per participant
  -> every replica updates its own partial[S]
  -> direct local finalize, no AllReduce/barrier/workspace
```

例如 `M=8, threads=128` 时，常见 layout 是一个 local slot、16 个 replicas；
materialized reducer 从每线程 `partial[8]` 缩为 `partial[1]`。而
`Parallel(M, K) -> output[M]` 不是 direct identity mapping，不能走 LocalComplete；
planner 会继续尝试下面的 projected partial group，证明失败时才走 FullParticipant。

materialization 后，init/update/finalize call 的 access region 都立即改写成 compact
physical buffer 上的合法 region；update 原本的 logical indices 单独保存在 planned
annotation 中，供 `ReducerUpdateOp` 映射到 physical slots。这样
`PlanAndMaterializeReducers` 与 `LowerTileOp` 之间不存在 shape/region 不一致的临时 IR。

这条路径故意不处理 Fragment contribution、permuted/affine output mapping 或 mixed
side effects；任一证明失败都只影响性能，不改变合法程序语义。

### 7.3 Projected partial groups

LocalComplete 失败后，planner 会逐个 update site 尝试识别一个 compiler-known reduction
axis。目前覆盖 scalar `Parallel(K) -> output[0]`、显式
`Parallel(M, K) -> output[M]`，以及 layout inference 融合后的等价 contiguous Range。

planner 从 source loop layout 投影掉 reduction axis，得到 compact partial layout，并提取
compile-time thread reduction steps。第一版只接受 power-of-two logical width，且要求
destination Fragment 能表示 projected logical outputs。拥有相同 source layout、partial
layout 和 thread steps 的 sites 共享一份 partial；不兼容的 sites 使用独立 partial，无法
证明的 sites 则共同使用一份 canonical FullParticipant partial。

finalize 先把 destination 初始化为 identity，然后依次归约并 combine 每个 physical
partial group，最后只应用一次 seed。projected collective 在完整 participant scope 中
uniformly 执行；partial layout image 之外的线程提供 identity，避免把 warp shuffle 或
barrier 放进不完整的 partition predicate。

## 8. Parallel effect markers

### 8.1 marker 的作用

`ReducerUpdateOp` 位于 `T.Parallel` 时会 lower 成：

```text
AttrStmt(
  key = "tl.parallel_multiplicity",
  value = 1,
  body = partial[index] = combine(...)
)
```

这是一个通用 effect multiplicity contract，含义是：

> 对当前 logical parallel iteration，这个 effect statement 必须执行一次。

它不包含 reducer buffer 列表，也不要求 `PartitionLoop` 理解 reducer 类型。

### 8.2 `PartitionLoop` 如何消费 marker

`PartitionLoop` 对 loop layout 做 `InverseWithLevel` 后，可以得到 logical indices 和 `REP`。它只把被 marker 包住的 statement 改成：

```cpp
if (REP == 0) {
  partial[index] = combine(partial[index], contribution);
}
```

同一个 `T.Parallel` 中的普通 Fragment load/store 不会被这个 guard 包住。

### 8.3 Physical partial partition marker

LocalComplete 和 projected update 使用另一个 marker：

```text
tl.parallel_partition_required
```

它要求 `LowerTileOp` 必须按 planner 已接受的 loop layout 做 physical thread
partition，但不额外套用 generic `REP == 0` guard。partition 后 marker 会被移除。带这
两类 reducer marker 的 loop 都禁止普通 vectorization，以免把对同一个 partial slot 的
loop-carried dependency 错误改写为并行 stores。

这解决了 PR #2881 临时参数的问题：

```cpp
PartitionLoop(..., fully_replicated_reducer_buffers)
```

该参数和 `ReducerStoreGuarder` 已经删除。loop partition 现在只消费一个局部、显式、与 buffer identity 无关的 multiplicity marker。

### 8.4 为什么 marker 只在必要时生效

如果 loop layout 的 `ReplicateExtent == 1`，每个 logical iteration 本来就只执行一次，不需要额外 guard。

如果存在 replicas，marker 才会选择 `REP == 0` 的 canonical physical execution。这样既避免 contribution 重复，也不会错误地把整个 finalize collective 放进 `REP == 0` 条件。

## 9. Finalize correctness baseline

共享 lowering 位于：

```text
src/backend/common/op/finalize_reducer.h
```

CUDA 和 ROCm 只提供 target-specific collective emitter 和 capability information。

### 9.1 当前 participant domain

当前 finalizer 使用 `LowerArgs.thread_bounds` 作为 participant range，并要求它是 compiler-known constant contiguous `Range`：

```text
[participant_min, participant_min + participant_extent)
```

当前实际常见路径是完整 CTA range。non-power-of-two collective width 会通过现有 `CheckAllReduceWidth` 明确拒绝。

### 9.2 每个 epoch 的 lowering

1. 按 destination Fragment layout 把 logical result 初始化为 identity；
2. 依次处理每个 physical partial group；
3. canonical group 在完整 participant range 上执行 scalar AllReduce；
4. projected group 从 partial layout inverse 恢复 logical output，layout image 外的线程提供
   identity，并由所有 participants uniformly 到达 collective；
5. 根据 destination Fragment inverse layout 判断当前 thread 是否拥有结果，再把 group
   result combine 到 destination；
6. 所有 groups 完成后，按 logical output 恰好 combine 一次 seed。

概念代码是：

```cpp
for (logical_index : full_output_shape) {
  reduced = AllReduce(partial[logical_index]);
  reduced = combine_seed_if_present(reduced);

  if (destination_owner(logical_index, threadIdx.x)) {
    destination[local_index(logical_index)] = reduced;
  }
}
```

### 9.3 `batch` 当前只是 hint

`T.finalize_reducer(..., batch=N)` 的 `batch` 当前不影响语义。第一版无论 hint 是多少都回退到 scalar collectives。

这样先避免旧 batched finalize 中 layout batch size、workspace stride 和 barrier count 混在一起导致的 wrong-code。真正的 batched fast path 以后必须证明与 scalar baseline 等价后再加入。

### 9.4 target 状态

| target | 当前状态 |
|---|---|
| CUDA | scalar AllReduce emitter 已实现并做数值/codegen 测试 |
| ROCm | scalar AllReduce emitter 已实现并通过 ROCm + HIP stub 编译 |
| CPU | pipeline 会验证 reducer，但 finalize 没有 target emitter，会明确失败 |
| Metal | pipeline 会验证 reducer，但 finalize 没有 target emitter，会明确失败 |
| WebGPU | pipeline 会验证 reducer，但 finalize 没有 target emitter，会明确失败 |

ROCm 尚未在真实设备上做数值验证。

## 10. `BD=8, threads=128` 现在如何 lowering

对应 Issue #2408 的典型输入是：

```python
src = T.alloc_fragment((8,), T.float32)
T.copy(A, src)

total = T.alloc_reducer((1,), T.float32, op="sum")
T.reducer_init(total)

for i in T.Parallel(8):
    T.reducer_update(total[0], src[i])

result = T.alloc_fragment((1,), T.float32)
T.finalize_reducer(total, result)
```

代表性的 CUDA 结构变成：

```cpp
float total[1];
total[0] = 0.0f;

// 128/8 physical replicas 中，只保留 canonical replica 的 contribution。
if ((threadIdx.x >> 3) == 0) {
  total[0] += src[layout_dependent_local_index];
}

// Collective 位于 replica guard 外，所有 128 participants 都会到达。
float reduced =
    tl::AllReduce<tl::SumOp, 128, 1, 0,
                  tl::NamedBarrier<128>>::run(total[0], workspace);

result[0] = reduced;
```

这里最重要的是区分两件事：

- `REP == 0` 决定一个 logical contribution 只出现一次；
- `AllReduce<..., 128>` 决定当前 correctness baseline 用哪些 participants 合并 partials。

前者属于 update multiplicity，后者属于 finalize communication。它们不再从同一个 reducer Fragment layout 推导。

## 11. 删除的 v1 machinery

这个分支选择 clean break，没有保留 legacy compatibility lowering。

已删除：

- `src/op/finalize_reducer.h`
- `src/op/finalize_reducer.cc`
- `src/transform/layout_reducer.h`
- `src/transform/layout_reducer.cc`
- `LayoutReducer` Python transform wrapper
- `ReducerRepType`
- ParallelOp 中的 reducer metadata 特例
- vectorize planner 中的 all-replicated reducer 特例
- race verifier 中跳过 reducer stores 的特例
- `PartitionLoop` 的 `fully_replicated_reducer_buffers` 参数
- `ReducerStoreGuarder`

删除这些特例后，职责变成：

| 组件 | 现在负责什么 |
|---|---|
| reducer verifier/planner | lifecycle、contribution multiplicity、storage materialization |
| Fragment LayoutInference | 普通数据的 physical placement |
| ParallelOp | logical loop 到 physical threads 的 layout |
| PartitionLoop | 实现 generic multiplicity marker |
| finalize backend | 对已确定 participant range 执行 collective |

## 12. Python API、example 和文档迁移

### 12.1 Python API

修改了：

- `tilelang/language/allocate.py`
- `tilelang/language/reduce_op.py`
- `tilelang/language/common.py`
- `tilelang/transform/__init__.py`

`T.reducer_init` 和 `T.reducer_update` 已加入 language exports，`T.finalize_reducer` 改成强制接收 destination。

frontend 会尽早检查：

- reducer scope；
- destination 不 alias reducer；
- destination 是 `local.fragment`；
- reducer/destination dtype 一致；
- reducer/destination rank 一致；
- `batch >= 1`；
- bitwise reducer dtype 合法。

C++ verifier/backend 仍然会再次检查关键 invariant，不能只依赖 Python frontend。

### 12.2 in-tree migration

已经迁移：

- `examples/gemv/example_gemv.py` 中的 `gemv_alloc_reducer`
- `testing/python/language/test_tilelang_language_reduce.py` 中的 reducer tests
- `docs/programming_guides/instructions.md`

GEMV 现在显式 init，在 `T.Parallel` 内调用 update，并 finalize 到独立 result Fragment。

## 13. 测试与验证状态

### 13.1 新增专项测试

文件：

```text
testing/python/language/test_tilelang_language_reducer_v2.py
```

当前包含 13 个测试，覆盖：

- frontend IR 中存在 `local.reducer` 和三个 first-class calls；
- CUDA source 中 reducer-only IR 已消失；
- `BD=8, threads=128` 生成正确的 update replica guard；
- 多 reducer allocations；
- predicated update；
- seed 不重复计入；
- sum/max 数值正确性；
- bitand/bitor/bitxor 数值正确性；
- legacy `replication=` 拒绝；
- finalize 缺 destination 拒绝；
- direct clear/load/store 拒绝；
- missing init/finalize 和 double init 拒绝；
- output index 越界或无法证明安全时拒绝；
- replicated update 使用 physical `threadIdx` 时拒绝；
- replicated update 读取 ordinary thread-private local value 时拒绝。

### 13.2 已通过的检查

```text
pre-commit: passed
default/CUDA CMake build: passed
ROCm + HIP stub CMake build: passed
focused reducer/reduce pytest: 41 passed
broad reducer/reduce run: 104 passed outside the known ordinary reduce cases
GEMV 128 x 128 numerical check: passed
```

### 13.3 当前 broad test 中另外暴露的问题

普通 `T.reduce` 的下面三个既有 batch/layout 组合会触发原有 backend assertion：

- `sum-float16-64x128-f2f-t256-b4`
- `sum-bfloat16-64x128-f2f-t256-b4`
- `min-float16-128x128-f2f-t256-b8`

diagnostic 分别属于：

```text
ReduceOp: batch exceeds per-thread output element count N
```

这些 case：

- 不使用 `local.reducer`；
- 不产生 reducer metadata；
- `PlanAndMaterializeReducers` 会直接返回原函数；
- 失败位置是原有 ordinary `T.reduce` batched lowering guard。

因此这个分支没有顺便修改它们。它们应该作为独立的 ordinary reduce/layout batch 问题处理。

## 14. 当前明确暂缓的内容

### 14.1 Generalized projected planning

当前 projected partial group 只识别一个 compiler-known reduction axis、contiguous/fused
row-major mapping 和 power-of-two logical width。任意 affine/permuted mapping、多个
reduction axes、symbolic thread split 或需要合并近似 signatures 的情况仍然回退
FullParticipant。后续扩展仍必须保持“证明失败只影响性能”的原则。

### 14.2 Batched finalize fast path

`batch` 目前只是 non-semantic hint。没有 `run_batch` lowering，也没有 batched workspace/barrier planning。

### 14.3 Multiple epochs

每个 allocation 只允许：

```text
one init -> zero or more updates -> one finalize
```

不支持 finalize 后 reset，也不支持在 loop 内重复完整 epoch。

### 14.4 Custom combine

只接受 compiler 内建 associative + commutative reductions。没有用户自定义 reducer lambda，也没有非交换 reduction。

### 14.5 复杂 participant domain

没有 discontiguous participant set、动态 participant width 或任意 predicate-defined group。

### 14.6 性能优化

当前已实现 conservative LocalComplete fast path。B300、128 threads、关闭 warp
specialization/TMA、50 ms warmup + 200 ms measurement 的 benchmark 结果为：

| case | legacy `replication="none"` | v2 LocalComplete | legacy / v2 |
|---|---:|---:|---:|
| M8, 512 blocks | 7.168 us | 6.528 us | 1.098x |
| M32, 512 blocks | 7.168 us | 6.528 us | 1.098x |
| M128, 256 blocks | 7.168 us | 6.464 us | 1.109x |

三个 v2 kernel 都从 full logical array 缩为每线程一个 local slot，且生成代码中
AllReduce/named barrier 都为 0。fallback regression 中，M1xK128 与 M32xK128 的
v2 latency 分别为 7.168 us 和 11.200 us，与 legacy 的 7.168 us 和 11.168 us
基本一致，并继续各自保留一个 AllReduce site。

仍未尝试：

- general affine/permuted/Fragment ownership compact storage；
- 合并多个 output collectives；
- 合并或批量执行多个 projected group collectives；
- 在 FullParticipant fallback 中跳过无用 full-array initialization；
- 根据 destination layout 批量 store；
- 针对 large output shape 控制 register pressure。

## 15. 建议的代码阅读顺序

如果要从源码理解当前实现，建议按下面的顺序：

1. `tilelang/language/allocate.py`
   - 看 `alloc_reducer` 如何生成 `local.reducer` 和 metadata。
2. `tilelang/language/reduce_op.py`
   - 看三个 public first-class calls 如何构造 access regions。
3. `src/op/deferred_reducer.h`
   - 看 IR node fields 和 target registry contract。
4. `src/op/deferred_reducer.cc`
   - 看 init/update 的基本 lowering 和 multiplicity marker。
5. `src/transform/reducer/verify_reducer_epochs.cc`
   - 看 metadata collector 和 lifecycle verifier。
6. `src/transform/reducer/reducer.cc`
   - 看 physical planner、replica checks 和 materializer。
7. `tilelang/cuda/pipeline.py`
   - 看三个 reducer passes 位于整个 compiler pipeline 的位置。
8. `src/transform/lower_tile_op.cc`
   - 看 marker 如何影响 loop partition/vectorize decision。
9. `src/transform/loop_partition.cc`
   - 看 `REP == 0` 与 partition-required 两类 marker 如何分别消费。
10. `src/backend/common/op/finalize_reducer.h`
- 对照 FullParticipant collective 与 LocalComplete direct-local finalize。
11. `src/cuda/op/finalize_reducer.cc` 和 `src/rocm/op/finalize_reducer.cc`
    - 看 target-specific AllReduce emitter。
12. `testing/python/language/test_tilelang_language_reducer_v2.py`
    - 用正例和 diagnostics 对照前面的 contract。

## 16. 当前工作区状态

所有实现目前位于：

```text
refactor/reducer-v2
```

该分支用于持续评审 correctness baseline、LocalComplete 和第一版 projected partial
group planning。

这个状态可以概括为：

```text
已完成：
  clean-break API
  first-class reducer IR
  lifecycle/access verifier
  replica-safe update lowering
  full local-array materialization
  contiguous full-participant finalize baseline
  direct-ownership LocalComplete plan
  compact per-layout local partial storage
  collective-free finalize for independent outputs
  constrained projected partial groups
  mixed projected/canonical multi-partial finalize
  uniform collective participation with identity outside the layout image
  CUDA/ROCm scalar emitters
  in-tree migration 与回归测试

暂缓：
  generalized affine/multi-axis projected planning
  batched finalize optimization
  general affine/Fragment ownership plans
  multiple epochs
  custom combine
  discontiguous participant domain
```
