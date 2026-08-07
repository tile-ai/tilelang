# Reducer v2 提案：Ownership-safe deferred reduction

- 状态：Draft，征求设计反馈
- 日期：2026-08-07
- 相关问题：[Issue #2408](https://github.com/tile-ai/tilelang/issues/2408)
- 长期讨论：[RFC #2897](https://github.com/tile-ai/tilelang/issues/2897)
- 临时修复：[PR #2881](https://github.com/tile-ai/tilelang/pull/2881)
- 原型实现：[Draft PR #2901](https://github.com/tile-ai/tilelang/pull/2901)
- 详细实现计划：[Reducer v2 重构设计方案](./reducer_v2_refactor_plan.md)

## 术语约定

| 术语 | 含义 |
|---|---|
| v1 reducer | 本文对当前 TileLang reducer 实现的称呼：reducer 表现为带有特殊 layout/replication 属性的 Fragment buffer，通过 `T.clear`、普通 read-modify-write `BufferStore`（如 `acc[i] += value`）和 `T.finalize_reducer` 使用 |
| logical output | reducer shape 中由 logical indices 指定的一个结果元素 |
| logical contribution | 一个动态 logical reducer update instance 提供给 logical output 的值；v2 由 `T.reducer_update` 显式表示，v1 只能从普通 read-modify-write store 猜测 |
| 贡献次数 | 一个 logical contribution 被 combine 到对应 logical result 中的次数；reducer 语义要求它恰好为 1，不能随 physical layout 的 replica 数量变化 |
| participant | reducer epoch execution scope 中的一个 physical thread；它必须初始化自己的 partial，并按 selected plan 到达 finalize/collective，但不一定产生非 identity contribution |
| replica | layout 为同一个 logical parallel iteration 产生的 physical execution 副本 |
| partial | finalize 前由一个 participant 保存、尚未完成全局合并的 accumulator state |

## 摘要

本文提议把 TileLang reducer 定义为一个 first-class deferred reduction epoch，而不是
带有特殊 layout 的普通 Fragment buffer。

核心语义是：

> `T.reducer_update` 定义 logical contribution 及其贡献次数；physical layout
> 只决定 contribution 存在哪里、由哪些 participant 通信，不能改变 contribution
> 出现的次数。

提案采用 correctness-first 的两层实现：

1. 编译器始终可以使用 FullParticipant baseline：每个 participant 持有一份完整
   logical output partial，每个 `T.Parallel` logical iteration 只选择一个
   representative 更新，最后执行 participant-wide collective。
2. 只有 ownership proof 成立时，编译器才选择 compact LocalComplete 或未来的
   subgroup fast path；证明失败只影响性能，不能静默产生错误结果。

第一版是 clean break，不兼容旧式 `T.clear`、普通 reducer `BufferStore`、
`acc[i] += value` 或 in-place finalize。

## 1. 设计 reducer 的动机

### 1.1 为什么需要设计 `alloc_reducer`

先看一个只使用普通 `T.reduce_sum` 的 tiled GEMV。由于 `K` 维无法放进一个 tile，
kernel 在每个 `k_tile` 中物化当前 tile 的乘积，立即 reduce 出 `tile_sum`，再把它
累加到跨 tiles 存活的 `y_frag`：

```python
# 为突出 reduction，省略输入、输出的函数签名。
with T.Kernel(..., threads=128):
    A_frag = T.alloc_fragment((BLOCK_M, BLOCK_K), T.float32)
    x_frag = T.alloc_fragment((BLOCK_K,), T.float32)
    products = T.alloc_fragment((BLOCK_M, BLOCK_K), T.float32)
    tile_sum = T.alloc_fragment((BLOCK_M,), T.float32)
    y_frag = T.alloc_fragment((BLOCK_M,), T.float32)
    T.clear(y_frag)

    for k_tile in range(T.ceildiv(K, BLOCK_K)):
        T.copy(A[..., k_tile * BLOCK_K], A_frag)
        T.copy(x[k_tile * BLOCK_K], x_frag)

        for i, k in T.Parallel(BLOCK_M, BLOCK_K):
            products[i, k] = A_frag[i, k] * x_frag[k]

        T.reduce_sum(products, tile_sum, dim=1, clear=True)
        for i in T.Parallel(BLOCK_M):
            y_frag[i] += tile_sum[i]

    T.copy(y_frag, y)
```

把这个 kernel 实例化为 `BLOCK_M=32`、`K=512`、`BLOCK_K=128`、
`threads=128` 和 float32。下面只保留 v1 compiler 生成 CUDA 中与 reduction 直接
相关的部分；load、地址计算和最终 store 均已省略：

```cpp
float y_frag[4];
float products[32];
float2 tile_sum_pack[4];
float tile_sum[4];

for (int k_tile = 0; k_tile < 4; ++k_tile) {
  // Producer: materialize products for the current tile.
  // products[...] = A_frag[...] * x_frag[...];

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    // Part 1: participant-local fold.
    tile_sum_pack[i] = make_float2(0.0f, 0.0f);
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      tile_sum_pack[i] =
          tl::add2(tile_sum_pack[i],
                   *(float2*)(products + (i * 8) + (rv * 2)));
    }
    tile_sum[i] = tile_sum_pack[i].x + tile_sum_pack[i].y;

    // Part 2: cross-participant combine.
    tile_sum[i] =
        tl::AllReduce<tl::SumOp, 16, 1, 0,
                      tl::NamedBarrier<128>>::run(tile_sum[i]);
  }

  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    y_frag[i] += tile_sum[i];
  }
}
```

这里可以看到，一次 `T.reduce_sum` 实际包含两件不同的事情：

1. **Participant-local fold**：把当前 participant 持有的多个 `products` 合并成
   `tile_sum` partial；这只是 local arithmetic，不需要线程间通信。
2. **Cross-participant combine**：通过 subgroup `AllReduce` 合并不同 participants
   的 partials；这一步包含真正的 cross-thread communication 和 synchronization。

从中间 IR 的角度看，一个动态 `T.reduce_sum` instance 大致展开成下面的整个
`SeqStmt`。这里使用 pseudo-TIR，`out_slot` 表示当前 participant 持有的 output
slot，`rv` 表示该 output 在当前 participant 内的 reduction values：

```text
# BEGIN lowering of:
#   T.reduce_sum(products, tile_sum, dim=1, clear=True)

for out_slot in serial(4):
    local_pack = (0.0f, 0.0f)

    # Part 1: participant-local fold
    for rv in unrolled(4):
        local_pack += load_float2(
            products,
            offset=out_slot * 8 + rv * 2,
        )
    local_partial = local_pack.x + local_pack.y

    # Part 2: cross-participant combine
    tile_sum[out_slot] = allreduce_sum(
        local_partial,
        participants=16,
    )

# END lowering of T.reduce_sum
```

也就是说，从初始化 `local_pack`、折叠 participant-local values，到执行 subgroup
`AllReduce` 并写回 `tile_sum`，这一整块都是同一个 `T.reduce_sum` 调用的 lowering，
并不是调用点外另外插入的一次 reduction。

当前 `T.reduce_sum` 把这两步绑定在同一个调用中：它看到当前 tile 的完整
`products` fragment 后，立即执行完整的 local-fold-and-combine，调用返回后
`tile_sum` 才能被累加到 `y_frag`。因此若 `K` 被分为 `N_K` 个 tiles，每个 logical
output 就会经历 `N_K` 次 cross-participant combine。生成代码还需要同时保存
`products[32]`、`tile_sum_pack[4]`、`tile_sum[4]` 和跨 tiles 存活的 `y_frag[4]`。

最初设计 `alloc_reducer`，正是为了把 cross-thread combine 移出这个 tiled loop：
各 participant 先持续累加自己的 local partial，所有 tiles 处理完成后再统一合并。
同一个 kernel 可以用下面的 v1 reducer 风格表示；该语法只用于解释历史动机，不
属于本提案支持的 v2 API：

```python
# 计算 y[i] = sum_k(A[i, k] * x[k])
with T.Kernel(..., threads=128):
    A_frag = T.alloc_fragment((BLOCK_M, BLOCK_K), T.float32)
    x_frag = T.alloc_fragment((BLOCK_K,), T.float32)
    y_reducer = T.alloc_reducer(
        (BLOCK_M,),
        T.float32,
        op="sum",
        replication="all",
    )
    T.clear(y_reducer)

    for k_tile in range(T.ceildiv(K, BLOCK_K)):
        # A_frag 和 x_frag 只包含当前 K tile。
        T.copy(A[..., k_tile * BLOCK_K], A_frag)
        T.copy(x[k_tile * BLOCK_K], x_frag)

        for i, k in T.Parallel(BLOCK_M, BLOCK_K):
            y_reducer[i] += A_frag[i, k] * x_frag[k]

    T.finalize_reducer(y_reducer)
    T.copy(y_reducer, y)
```

同样参数下，v1 `alloc_reducer` 生成 CUDA 中与 reduction 直接相关的部分是：

```cpp
float y_reducer[32];
extern __shared__ __align__(1024) float workspace[];

// reducer initialization is omitted.
for (int k_tile = 0; k_tile < 4; ++k_tile) {
  // Participant-local accumulation is fused into reducer updates.
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    y_reducer[i] += A_frag[i] * x_frag[0];
  }
}

// T.finalize_reducer: one cross-participant combine per logical output.
for (int i = 0; i < 32; ++i) {
  __syncthreads();
  y_reducer[i] =
      tl::AllReduce<tl::SumOp, 128, 1, 0,
                    tl::NamedBarrier<128>>::run(y_reducer[i],
                                                &(workspace[0]));
}
```

`alloc_reducer` 没有消除 reduction 的任何一部分，而是重新安排了它们的位置：
participant-local fold 被融合进跨 tiles 的 reducer updates；32 个 logical outputs
的 cross-participant combine 则全部延后到 `T.finalize_reducer`。

两个版本计算相同的 `y[i] = sum_k(A[i, k] * x[k])`，但 collective 的位置不同：

1. `T.reduce_sum` 版本在每个 `k_tile` 内立即完成所有 logical outputs 的
   cross-thread combine，然后把结果累加到 `y_frag`，因此每个 output 经历
   `N_K` 次 combine phases。
2. v1 `alloc_reducer` 版本在循环内只更新 participant-local partial，直到
   `T.finalize_reducer` 才执行每个 logical output 的 cross-thread combine，因此
   每个 output 只经历一次 final combine phase。

CUDA 源码中的 local array 数量本身不能证明硬件寄存器数量，因为后端还会做
scalarization、liveness analysis 和 register coalescing。对上述两个 cubins 执行
`cuobjdump --dump-resource-usage`，得到下表。该 snapshot 使用 TileLang v1 commit
`4b218139d7cd`、CUDA 13.1 和 `sm_103`；绝对数值不应直接推广到其他 toolchain 或
GPU。表中的 register 数据来自完整 cubin，而不是上面经过裁剪的 CUDA 片段：

| 写法 | 每个 logical output 的 combine phases | registers/thread |
|---|---:|---:|
| 每 tile `T.reduce_sum` | 4 | 168 |
| v1 `alloc_reducer` | 1 | 76 |

因此在这个 `BLOCK_M=32` 实例中，`T.reduce_sum` 除了让每个 logical output 多经历
三次 combine phases，还多用了 92 registers/thread。生成代码中的
`products[32]`、`tile_sum_pack[4]`、`tile_sum[4]` 与 `y_frag[4]` 解释了额外 live
state 的来源。不过，这不是 `alloc_reducer` 对所有 shape 的性能保证：v1 的
`replication="all"` 已经让每个线程持有完整的 `y_reducer[32]`，当 logical output
继续增大时，这部分 register pressure 也可能成为主导。因此 reducer v2 仍需通过
ownership planning 决定 compact storage，而不能把 fully replicated storage 当作
默认优化方案。

如果没有 reducer，又想把 collective 移出循环，就需要物化所有
`A[i, k] * x[k]` 或所有 tile partials，再在循环结束后统一 reduce。这会增加
register/shared-memory footprint 和数据搬运，也不利于 software pipeline。

`alloc_reducer` 能同时避免 per-tile collectives 和完整中间结果物化，是因为：

- 每个 participant 在 register/local storage 中保存自己的 `y[i]` partial；
- 每个 `k_tile` 只把新 contribution 累加到这些 partials；
- partial state 跨越整个 tile loop 存活；
- 所有 tiles 处理完成后，`T.finalize_reducer` 才执行必要的 cross-thread combine。

我们把这种“跨多个 tile/update 持有 participant-local partial，最后才统一完成
跨 participant reduction”的实现方式称为 **deferred reduction**。

### 1.2 另一个实用的案例: FlashAttention：把 row-sum 拆成 local accumulation 与 final reduction

[FlashAttention-2](https://arxiv.org/abs/2307.08691) 的 online softmax 是一个很
直观的例子。下面只保留 online softmax 中的 row-sum state `l`；计算 row-max、
`alpha` 和当前 KV tile 的 `p` 的代码均省略。

直接使用 `T.reduce_sum` 时，每个 KV tile 都立即完成一次 row-sum：

```python
p = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
alpha = T.alloc_fragment((BLOCK_M,), T.float32)
tile_l = T.alloc_fragment((BLOCK_M,), T.float32)
l = T.alloc_fragment((BLOCK_M,), T.float32)
T.clear(l)

for kv_tile in range(T.ceildiv(N, BLOCK_N)):
    # Omitted: compute row-max, alpha, and p for this KV tile.
    T.reduce_sum(p, tile_l, dim=1, clear=True)

    for i in T.Parallel(BLOCK_M):
        l[i] = alpha[i] * l[i] + tile_l[i]
```

如果有 `N_KV` 个 KV tiles，这个版本就执行 `N_KV` 次 cross-participant row-sum。
使用 v1 `alloc_reducer` 时，可以先让每个 participant 跨 tiles 保存自己的 row-sum
partial：

```python
p = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
alpha = T.alloc_fragment((BLOCK_M,), T.float32)
l_reducer = T.alloc_reducer(
    (BLOCK_M,),
    T.float32,
    op="sum",
    replication="all",
)
T.clear(l_reducer)

for kv_tile in range(T.ceildiv(N, BLOCK_N)):
    # Omitted: compute row-max, alpha, and p for this KV tile.

    # alpha is row-wise and identical for all participants of the same row.
    for i in T.serial(BLOCK_M):
        l_reducer[i] *= alpha[i]

    for i, j in T.Parallel(BLOCK_M, BLOCK_N):
        l_reducer[i] += p[i, j]

T.finalize_reducer(l_reducer)
```

第二个版本在 tile loop 内只做 participant-local rescale 和 accumulation，最后才执行
cross-participant combine。因此每个 row 从 `N_KV` 次 row-sum collectives 变成一次
final combine；row-max 仍然留在每个 tile 内，不会被 defer。

这里的 `l_reducer[i] *= alpha[i]` 是 reducer state transform。当前 reducer v2
第一版尚不支持它，所以这个例子只用于说明进一步扩展 deferred reduction 的价值，
不是第一版已经承诺支持的 API。

### 1.3 为什么不能只用 `alloc_fragment + T.Parallel`

1.1 说明了为什么需要把 cross-thread combine 延后到 tile loop 之后。一个自然的
问题是：既然 partial 最终保存在 registers 中，为什么不直接使用普通 Fragment？

#### Fragment 能保存 partial，但没有 reduction contract（规定各线程保存的中间累加值如何合并成最终结果的语义约定）

例如，用户可以写出下面的代码：

```python
# 这段代码缺少 cross-thread combine，并不是正确的 GEMV 实现。
partial = T.alloc_fragment((BLOCK_M,), T.float32)
T.clear(partial)

for k_tile in range(T.ceildiv(K, BLOCK_K)):
    T.copy(A[..., k_tile * BLOCK_K], A_frag)
    T.copy(x[k_tile * BLOCK_K], x_frag)

    for i, k in T.Parallel(BLOCK_M, BLOCK_K):
        partial[i] += A_frag[i, k] * x_frag[k]

T.copy(partial, y)
```

这并不只是抽象层面的担忧。使用 TileLang v1 commit `4b218139d7cd`，并取
`BLOCK_M=32`、`K=512`、`BLOCK_K=128`、`threads=128` 时，parallel-loop verifier
首先给出 warning：

```text
Data race detected: `partial(i,)` is written by multiple threads in loop (kk,)
```

warning 的含义是：在 logical IR 中，不同 parallel `kk` iterations 都对同一个
`partial[i]` 执行普通 RMW，但程序没有声明这些 updates 是 reduction
contributions。编译仍会继续，LayoutInference 随后把 iteration space 分配到线程。
令：

```text
row_group = threadIdx.x // 16   # 0..7
k_group   = threadIdx.x % 16    # 0..15
```

得到的关键 layout 是：

| Fragment | 一个线程持有的 logical elements | 每线程元素数 | replica 数量 |
|---|---|---:|---:|
| `A_frag` | `i = 8 * local_i + row_group`，`k = 8 * k_group + local_k` | 32 | 1 |
| `x_frag` | `k = 8 * k_group + local_k` | 8 | 8 |
| `partial` | `i = 8 * local_i + row_group` | 4 | 16 |

其中 `local_i=0..3`、`local_k=0..7`。对固定的 logical row `i`，`partial[i]`
沿 `k_group=0..15` 被复制成 16 份 thread-local partials；每份只累加自己负责的
K-slice。最终 CUDA 的关键部分是：

```cpp
float partial[4];
float A_frag[32];
float x_frag[8];

// Every thread initializes its own four partials.
for (int i = 0; i < 4; ++i) {
  partial[i] = 0.0f;
}

for (int k_tile = 0; k_tile < 4; ++k_tile) {
  // Loads into A_frag and x_frag are omitted.
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    partial[i >> 3] += A_frag[i] * x_frag[i & 7];
  }
}

// T.copy selects k_group == 0; it does not reduce the 16 replicas.
if ((threadIdx.x % 16) == 0) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    y[i * 8 + threadIdx.x / 16] = partial[i];
  }
}
```

生成代码中没有 `AllReduce`。每份 replica 只累加
`4 tiles * 8 values/tile = 32` 个 values，最后 `T.copy` 选择 `k_group=0` 的一份
写回，其余 15 份被丢弃。使用全 1 输入时，结果可以直接看出错误：

```text
actual y[i]   = 32
expected y[i] = 512
```

这里需要区分 logical warning 与最终的 physical execution：LayoutInference 之后，
各线程更新的是自己的 registers，并没有两个线程同时写同一个物理寄存器；真正的
问题是 16 份 partials 缺少 cross-thread combine。verifier 能指出普通
`T.Parallel` updates 存在冲突，却不能凭空决定应该使用哪种 reduction op 或如何
finalize。

这个具体结果说明，Fragment 确实可以保存某个 physical thread 的 local partial，
但这段程序没有声明多个 physical partials 与最终 `y[i]` 之间的关系：

- `partial[i]` 是最终 logical output，还是某个 participant 的中间状态？
- 多个 participants 持有的 `partial[i]` 应该按 sum、max 还是其他 op 合并？
- reduction identity 是什么，哪些 threads 属于 participant domain？
- combine 应该何时发生，结果采用什么 layout？

`T.Parallel` 只定义 logical iteration space；loop/layout inference 再决定这些
iterations 如何映射到 physical threads。`alloc_fragment` 只定义 local storage 及其
layout，最后的 `T.copy` 也只是 copy，不会自动把多个 thread-local partials reduce
成一个 logical result。

用户当然可以在每个 tile 后显式调用 `T.reduce_sum`，但这会回到 1.1 中每个 tile
立即执行 collective 的版本；也可以先物化额外的 reduction axis，但这又失去了
避免中间结果物化的收益。因此这里确实需要一个区别于普通 Fragment 的 reduction
abstraction。

#### v1 `alloc_reducer` 补充了 contract，但 update 仍然隐式

v1 `alloc_reducer` 正是为了补上这部分语义。相比普通 Fragment，它至少告诉编译器：

- 这是 reducer state，而不是普通 local buffer；
- logical output shape 和 reduction op 是什么；
- `T.clear` 建立一个 accumulation epoch；
- `T.finalize_reducer` 必须合并 physical partials。

但是 v1 把 reducer 实现为一种带特殊 layout/replication 属性的 Fragment buffer，
循环中的 update 仍然只是普通 read-modify-write：

```python
acc[i] += value

# 等价的普通 buffer 操作：
acc[i] = acc[i] + value
```

因此 compiler 仍需从 `BufferLoad + BufferStore` pattern、loop layout 和 Fragment
layout 反向猜测：

- 这个 RMW 是一个 logical reducer contribution，还是普通 state transform？
- 哪些动态 logical iterations 应该各贡献一次？
- layout replicas 是冗余 execution，还是需要在 finalize 中合并的不同 partials？
- 每个 logical contribution 的贡献次数是否恰好为 1？
- 哪些 participants 必须初始化 identity、进入 collective 和 barrier？
- logical seed 应该在每个 partial 中使用，还是在最终结果中只使用一次？

所以这一节并不是说 Fragment 的 storage abstraction 有问题，而是说 Fragment 不足以
承担 logical reduction semantics。v1 `alloc_reducer` 识别了这个缺口，却仍把最关键的
update semantics 隐藏在普通 RMW store 中；reducer v2 才进一步使用显式
`reducer_init`、`reducer_update` 和 `finalize_reducer` 建立完整 contract。

## 2. 当前 v1 `alloc_reducer` 为什么会重复累加

先不引入 update site 等抽象概念，直接看触发原始问题的 kernel。这里的 legacy
语法只用于解释 v1 的问题，不属于本提案支持的 API：

```python
@T.prim_func
def reducer_v1_example(
    A: T.Tensor((8,), T.float32),
    B: T.Tensor((1,), T.float32),
):
    with T.Kernel(1, threads=128):
        x_frag = T.alloc_fragment((8,), T.float32)
        T.copy(A, x_frag)

        total = T.alloc_reducer(
            1,
            T.float32,
            op="sum",
            replication="all",
        )
        T.clear(total)
        for i in T.Parallel(8):
            total[0] += x_frag[i]

        T.finalize_reducer(total)
        T.copy(total, B)
```

用户想要计算：

```text
B[0] = A[0] + A[1] + ... + A[7]
```

例如输入 `A = [1, 2, ..., 8]` 时，正确结果应为 `36`，但 v1 会得到 `576`。

为了看清问题，可以先把 reducer 换成一次普通的 `T.reduce_sum`：

```python
x_frag = T.alloc_fragment((8,), T.float32)
total = T.alloc_fragment((1,), T.float32)
T.copy(A, x_frag)
T.reduce_sum(x_frag, total)
T.copy(total, B)
```

这段代码在相同的 128-thread kernel 中会正确得到 `36`。下面比较两种写法经过同一版
v1 compiler 后的 layout 和 CUDA。

### 2.1 两种写法得到的 buffer layout 其实相同

省略恒为 0 的 local index 后，两种写法中的 `x_frag` 和 `total` 都得到以下 layout：

```text
x_frag[i]:
  thread         = rep * 8 + i
  replicate_size = 16
  thread_range   = [0, 128)

total[0]:
  thread         = rep
  replicate_size = 128
  thread_range   = [0, 128)
```

因此 `x_frag` 的物理映射也是相同的：

| `x_frag` 元素 | 持有它的线程 | 每个线程读到的值 |
|---|---|---|
| `x_frag[0]` | `0, 8, 16, ..., 120` | `A[0]` |
| `x_frag[3]` | `3, 11, 19, ..., 123` | `A[3]` |
| `x_frag[7]` | `7, 15, 23, ..., 127` | `A[7]` |

每个 `x_frag[i]` 都有 16 个 physical replicas，而 scalar `total[0]` 是 fully
replicated 的，每个线程都有一个 physical copy。仅看这两个 buffer 的 layout，无法
区分哪个 kernel 会算对。

v1 reducer 中显式 `T.Parallel(8)` 的 loop layout 也与 `x_frag` 相同：

```text
i   = threadIdx.x % 8
rep = threadIdx.x // 8
```

所以 `total[0] += x_frag[i]` 会被全部 128 个线程执行，每个逻辑 `i` 执行 16 次。

### 2.2 真正的区别在生成的归约宽度

下面是 legacy checkout 实际生成的关键 CUDA。`T.reduce_sum` 版本是：

```cpp
float x_frag[1];
float total[1];
x_frag[0] = A[((int)threadIdx.x) & 7];
total[0] = 0.0f;
total[0] = total[0] + x_frag[0];
total[0] =
    tl::AllReduce<tl::SumOp, 8, 1, 0, tl::NamedBarrier<128>>::run(total[0]);
if (((int)threadIdx.x) == 0) {
  B[0] = total[0];
}
```

v1 `alloc_reducer` 版本只有一处关键差别：

```cpp
float x_frag[1];
float total[1];
x_frag[0] = A[((int)threadIdx.x) & 7];
total[0] = 0.0f;
total[0] = total[0] + x_frag[0];
total[0] =
    tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(
        total[0], &workspace[0]);
if (((int)threadIdx.x) == 0) {
  B[0] = total[0];
}
```

`AllReduce` 中紧跟 `SumOp` 的模板参数才是这里的归约线程范围。`NamedBarrier<128>`
中的 `128` 表示 barrier policy 的到达线程数，不能把它误认为归约宽度。

`T.reduce_sum` 选择宽度 8。128 个线程被看成 16 组，每组 8 个线程；每一组都恰好
持有 `A[0]..A[7]`，因此独立算出一份 `sum(A)`：

```text
threads   0..7   -> A[0] + A[1] + ... + A[7] = 36
threads  8..15   -> A[0] + A[1] + ... + A[7] = 36
...
threads 120..127 -> A[0] + A[1] + ... + A[7] = 36
```

这些组之间不会再次求和。因为每组都得到相同结果，`total[0]` 的 128 个 physical
copies 最终确实相等，thread 0 写回其中一份即可。

v1 reducer 却选择宽度 128，把全部 128 个 per-thread partial 放进同一次归约；这在
数学上等价于把上述 16 组结果再次相加，最终得到：

```text
result = 16 * (A[0] + A[1] + ... + A[7]) = 576
```

实际执行结果与这个分析一致：

| 写法 | 归约宽度 | 输出 |
|---|---:|---:|
| `T.reduce_sum` | 8 | `36` |
| v1 `alloc_reducer` | 128 | `576` |

### 2.3 这个对比揭示了两个关键发现

#### 发现一：相同 layout 在 Fragment 和 reducer 中表示了不同含义

两种写法中的 `total` 都具有下面的物理 layout：

```text
thread         = rep
replicate_size = 128
thread_range   = [0, 128)
```

但是这个 layout 在两种场景中的含义并不相同：

| | 普通 Fragment `total` | v1 reducer `total` |
|---|---|---|
| replica 表示什么 | 已经算完的同一个 logical value 的物理副本 | 每个 participant 保存的一份待合并 partial |
| finalize/copy 前的实际内容 | 128 份相等的 `sum(A)` | thread `t` 中是 `A[t % 8]`，只有每 8 个线程后才重复 |
| 后续操作 | 任取一个有效副本写回，不再合并 replicas | `replication="all"` 要求 finalize 合并全部 128 份 partial |

也就是说，普通 Fragment 的 fully replicated 描述的是 value replication：这些
physical copies 在逻辑上是同一个已经完成的值。v1 reducer 的
`replication="all"` 描述的却更接近 partial participation：每个线程先保存自己的
中间累加值，之后还要参加 collective。

v1 使用同一个 Fragment layout 同时承载了这两种不同的语义。layout 本身只能说明值
放在哪些线程，不能说明这些 copies 是“已经相等的最终值”，还是“尚未合并、允许不同
的 partial”。因此，看到相同的 fully replicated layout，却生成不同代码，并不是
layout 推导前后矛盾，而是 `replication="all"` 在 reducer 中被赋予了普通 Fragment
layout 无法表达的额外含义。

#### 发现二：`T.reduce_sum` 看得到 source layout，finalize 只看得到 `total`

`T.reduce_sum(x_frag, total)` 在一个 op 中同时看得到：

- source logical extent 是 8；
- `x_frag` layout 是 `thread = rep * 8 + i`；
- `i` 是 reduction axis，`rep` 是 16 份重复执行；
- destination 是 `total[0]`。

因此它可以直接得到：

```text
沿 i 归约       -> reduction width = 8
保留 rep 维度   -> 16 个 replica groups 各自得到一份 sum(A)
```

这不仅正确，也更高效：生成的 `AllReduce<..., 8, ...>` 不跨 replica groups 通信，
这个实例中也不需要 128-thread reduction 使用的 shared-memory workspace。

v1 reducer 把 update 和 finalize 分开以后，这些信息没有保存在 reducer IR 中：

1. `total[0] += x_frag[i]` 只是普通 read-modify-write store，source layout 和 loop
   layout 不会成为 `total` 的语义属性。
2. `T.finalize_reducer(total)` 只看得到 `total` 的 fully replicated layout 和 sum
   op，不再看得到 `x_frag`，也不知道 partial 来自哪个 logical `i`、哪个 loop
   replica。
3. 仅凭 `replicate_size=128`，finalize 无法判断应该做 16 组独立的 8-way reduction，
   还是一次 128-way reduction；对更一般的多个 update sites，这个信息更无法从
   `total` layout 反推。

所以这不是简单地让 finalize 根据 `total` layout 选择一个更小的归约宽度就能解决的
问题；必要信息在 finalize 之前已经丢失。v1 只能把 128 份 partial 都当作不同贡献
进行合并。

从用户语义看，每个逻辑 `i` 只应让 `A[i]` 参与求和一次。本文把这称为一次 logical
contribution（逻辑贡献），它的贡献次数必须是 1；layout 产生 16 个 replicas 不能把
贡献次数改成 16。这正是 v1 没有显式保存、而 v2 需要在 update site 保留下来的信息。

### 2.4 从止血补丁到根因：reducer 不能继续伪装成 Fragment

发现重复 contribution 后，最直接的修复并不是立刻重写 reducer。`PartitionLoop`
仍然持有 `T.Parallel(8)` 的 loop layout，能够得到 `rep`，因此可以只让一个 canonical
replica 执行 store：

```cpp
float partial = 0.0f;
int i = threadIdx.x % 8;
int rep = threadIdx.x / 8;
if (rep == 0) {
  partial += A[i];
}
float result = AllReduceSum<128>(partial);
```

这就是 [PR #2881](https://github.com/tile-ai/tilelang/pull/2881) 采用的止血方向。它对
这个例子是正确的：前 8 个线程各贡献一个 `A[i]`，其余线程保持 identity，但全部
128 个线程仍然进入 finalize collective。

#### 为什么一个局部修复会变成 compiler-wide 特例

困难在于，v1 没有 `ReducerUpdateOp`。`PartitionLoop` 看到的只是普通
`BufferStore`，无法区分下面两种 store：

```python
y_frag[i] = value       # 每个 replica 都可能需要更新自己的 Fragment copy
total[0] += value       # 同一个 logical contribution 只能选择一个 replica
```

为了只 guard 第二种 store，短期实现必须把特殊 reducer buffer 的集合一路传给通用
loop lowering：

```cpp
PartitionLoop(..., fully_replicated_reducer_buffers)
```

然后根据 buffer identity 找到 reducer store，再在它外面添加 `rep == 0`。这个参数
本身可以修复当前问题，但也暴露出 v1 reducer 已经不是一个能被普通 Fragment pipeline
自然处理的对象。当前抽象大致经过以下路径：

```text
alloc_reducer(local.fragment + reducer_info)
  -> T.clear/T.fill 充当 epoch 起点
  -> 普通 BufferStore 充当 reducer update
  -> LayoutReducer 强制生成 Fragment layout
  -> ParallelOp / VerifyParallelLoop 对 reducer access 做特殊处理
  -> LowerTileOp / PartitionLoop 按 reducer buffer identity 添加 guard
  -> FinalizeReducer 从 reducer ReplicateExtent 推导 AllReduce width
```

每一层看到的都只是 reducer 语义的一小部分，于是每一层都需要额外补丁：

| compiler stage | 为什么普通 Fragment 规则不够 | 需要的 reducer 特例 |
|---|---|---|
| frontend / IR | `alloc_reducer` 返回可普通读写的 buffer，update 与任意 store 无法区分 | 从 `T.clear`、RMW pattern 和 block annotation 猜 lifecycle 与 combine |
| `LayoutReducer` / LayoutInference | partial 在 finalize 前允许不同，不满足 replicated Fragment 的值等价假设 | 强制生成 reducer layout，并避免普通 layout propagation 冲突 |
| `ParallelOp` / `VerifyParallelLoop` | 普通 replicated Fragment store 与 reducer contribution 的执行次数规则不同 | 对已识别的 reducer buffer 放宽或改写普通 ownership/race 规则 |
| `LowerTileOp` / `PartitionLoop` | loop lowering 不知道哪个副作用只能执行一次 | 传入 `fully_replicated_reducer_buffers`，按 buffer identity 给 store 加 guard |
| `FinalizeReducer` | `total` layout 不包含 source、update site 或 loop replica 信息 | 把 `ReplicateExtent()` 同时当成 participant domain 和 reduction width |

而且每加入一种合法写法，特例还会继续扩散。例如多个 update sites 可能拥有不同 loop
layouts；一个 `T.Parallel` 中可能同时包含普通 Fragment store 和 reducer update；
contribution 还可能来自 Fragment/local value。仅仅知道“这个 buffer 是 reducer”并不能
回答每个 update site 应该使用哪个 representative，也不能恢复前面已经丢失的 source
layout，从而生成 `T.reduce_sum` 那样的窄归约。

#### 根因不是某个 pass 推导得不够聪明

这些补丁共同指向同一个根因：v1 把一个 deferred reduction epoch 伪装成了普通
Fragment buffer。

- Fragment layout 描述一个 logical value 放在哪些线程和 local slots；replicas
  通常表示这个值的等价 physical copies。
- reducer partial 是一段尚未结束的计算状态；不同 participant 的值本来就允许不同，
  并且还带有 init、update、combine 和 finalize 的时序语义。
- 普通 `BufferStore` 只表示一次内存写入，不表示它对应哪个 logical contribution，
  更不表示这个 contribution 在 physical replicas 中应生效几次。
- finalize 时只剩下 `total` layout，再复杂的分析也无法唯一恢复多个 update sites
  曾经拥有的 source/loop layouts。

因此，继续把 reducer 当作 Fragment，会让 correctness 依赖越来越多 pass 对某个特殊
buffer 的共同识别。某个补丁遗漏一次识别，就可能静默改变 contribution 的次数。

#### 对应的长期解法

v2 不是单纯换一套 API 名字，而是针对上述信息丢失逐项改变 IR 边界：

| v1 的根本问题 | v2 的对应设计 |
|---|---|
| reducer handle 是普通 Fragment，可被任意 load/store | 使用不可普通读写的 `local.reducer` logical handle；规划完成后才物化为普通 local partial storage |
| init、update 和 finalize 隐藏在 `T.clear`、RMW store 与 annotation 中 | 使用 first-class `ReducerInitOp`、`ReducerUpdateOp` 和 out-of-place `FinalizeReducerOp` |
| contribution 次数没有显式表示 | 每个 update site 保留自己的 loop layout，并产生“每个逻辑迭代一次”的通用 marker |
| `PartitionLoop` 必须识别特殊 buffer | `PartitionLoop` 只实现 marker 的 `rep == 0` 语义，不再接收 reducer buffer 列表 |
| partial storage、result layout 和 participant domain 来自同一个 Fragment layout | planner 分别决定 local partial storage、final destination layout 和 execution-scope participant `Range` |
| ownership 推导失败可能影响 correctness | 始终保留 FullParticipant baseline；ownership proof 只启用可回退的 fast path |

换句话说，解决方案不是让 `FinalizeReducer` 根据 `total` layout 猜得更准，而是在
update 仍然可见时保留 contribution 语义，在 finalize 前由 planner 统一选择 physical
storage 与 communication plan。下面的 v2 写法正是这个推导的直接结果。

### 2.5 v2 显式表示 reducer update

v2 使用 `T.reducer_update` 把这个意图直接写进 IR：

```python
acc = T.alloc_reducer((1,), T.float32, op="sum")
T.reducer_init(acc)

for i in T.Parallel(8):
    T.reducer_update(acc[0], x_frag[i])

result = T.alloc_fragment((1,), T.float32)
T.finalize_reducer(acc, result)
```

这里的 `acc[0]` 是 update target descriptor，不是读取 reducer 当前值。frontend 会把
它转换为指向 logical output 0 的 read-write point region；只有出现在
`T.reducer_update` 的第一个参数中才合法。

最保守的 FullParticipant lowering 可以选择每组 replica 中 `rep == 0` 的线程执行
update：

```cpp
float partial = 0.0f;
int i = threadIdx.x % 8;
int rep = threadIdx.x / 8;
if (rep == 0) {
  partial += A[i];
}
float result = AllReduceSum<128>(partial);
```

现在只有线程 `0..7` 分别累加 `A[0]..A[7]`。其余 120 个线程的 partial 保持 sum
identity `0`，但仍然一起进入 `AllReduce<128>`。因此结果正好是 `sum(A)`。

这个 FullParticipant baseline 优先保证正确性，并没有复现 `T.reduce_sum` 的 8-way
fast path。关键区别在于，v2 planner 做决定时仍然看得到每个
`T.reducer_update`、它所在的 loop layout 和 contribution input layout；后续优化可以
据此证明并选择更窄的 reduction 或完全 local 的方案。证明失败时则回到上面的
baseline，不再要求只持有 `total` layout 的 finalize 猜测已经丢失的信息。

把本例和开头的术语对应起来：

| 术语 | 本例中的含义 |
|---|---|
| logical contribution | 某个逻辑迭代 `i` 提供的 `A[i]` |
| 贡献次数 | 每个 `A[i]` 必须且只能被合并 1 次 |
| replica | 执行同一个 `i` 的 16 个物理线程副本 |
| participant | 必须初始化 partial 并进入 finalize 的全部 128 个线程 |
| partial | 每个 participant 在 finalize 前持有的私有 `total[0]` |

这个例子说明了 v2 的核心边界：layout 决定一个逻辑迭代如何映射到物理线程，
`T.reducer_update` 决定它对最终结果贡献几次。前者不能改变后者。

## 3. 问题陈述

### 3.1 最小错误结果示例

上面的 `reducer_v1_example` 就是 Issue #2408 中产生错误结果的最小示例。输入
`A = [1, 2, ..., 8]` 时，正确结果是 `36`，旧实现会返回 `576`。

这个例子确定了三个关键事实：

1. 问题不在 `AllReduce<128>` 本身。完整 participant domain 继续使用
   128-thread collective 可以得到正确结果。
2. `total` 被标记为 fully replicated，不代表 accumulation 中 128 个
   `total[0]` partials 已经相等。
3. 真正缺失的是 update site 的贡献次数语义：同一个 logical `i` 的
   16 个 physical replicas 只能有一个执行 combine，其余 participant partials
   必须保持 identity。

PR #2881 通过识别特殊 reducer buffer，并让 `REP == 0` 的 canonical replica 执行
store，临时补上了这个贡献次数约束。v2 的目标是让 first-class update
statement 自己表达该 contract，而不是让通用 loop lowering 从 buffer identity
反向猜测。

### 3.2 v1 混合了四种独立概念

| 概念 | v1 中的近似载体 | 问题 |
|---|---|---|
| logical combine | reducer annotation | update 仍是任意 store，难以验证 |
| 贡献次数 | 没有显式表示 | loop replication 会改变贡献次数 |
| partial storage | FullyReplicated Fragment | partial 本来允许在线程间不同，违背 Fragment value 假设 |
| participant domain | result/reducer replicate extent | update scope、thread offset 和 barrier domain 被丢失 |

`fully replicated` 只能描述一个最终 logical value 的 physical copies，不能证明
accumulation 中所有 copies 已经相等。partial state 和普通 Fragment value 必须分开。

## 4. 目标与非目标

### 4.1 目标

- 每个动态 logical contribution 恰好参与一次 reduction。
- update 可以跨多个 serial/pipeline iteration 和多个 update site。
- correctness 不依赖成功合并复杂 ownership plans。
- reducer 在 finalize 前不可被普通读取、写入、alias 或 escape。
- identity、seed、participant domain、partial storage 和 destination layout 有独立语义。
- 优化分析失败时确定性 fallback，或者给出明确 unsupported diagnostic。
- CUDA 和 ROCm 共享高层 plan。
- 删除通用 compiler passes 中基于 reducer buffer identity 的特殊分支。

### 4.2 第一版非目标

- legacy reducer 语法兼容或自动 canonicalization。
- 一个 allocation 上的 multiple epochs/reset。
- 用户自定义 combine lambda。
- 非交换 reduction 的稳定顺序。
- bitwise reproducible floating-point sum。
- 任意 discontiguous 或动态 participant set。
- 带 per-iteration uniform state rescale 的 online-softmax reducer，或其他
  structured reducer state。
- 第一版即恢复所有 subgroup 性能。
- 第一版即得到最小 register footprint。

第一版只支持 compiler 内建的 associative + commutative reductions。浮点 reduction
沿用当前 GPU lowering 允许 reassociation 的约定。

## 5. 提议的用户语义

### 5.1 API

```python
acc = T.alloc_reducer(
    logical_shape,
    dtype,
    op="sum",
    seed=None,
)
T.reducer_init(acc)

T.reducer_update(acc[logical_indices], contribution)

dst = T.alloc_fragment(logical_shape, dtype)
T.finalize_reducer(acc, dst)
```

`T.alloc_reducer` 只创建一个 logical reducer handle。它不是可读写 Fragment，也不
隐式执行 initialization。

`acc[logical_indices]` 在 `T.reducer_update` 的第一个参数位置只是一个 logical update
target descriptor。frontend 会立即把它规范化为 read-write point region，并不会执行
普通 reducer load；将 `acc[...]` 赋给变量、参与表达式或传给其他 op 仍然是非法访问。

三个 first-class operations 的 contract 是：

| operation | contract |
|---|---|
| `T.reducer_init(acc)` | 在 epoch participant domain 中，用 algebra identity 初始化 planner 选择的 physical partial |
| `T.reducer_update(acc[indices], value)` | 为指定 logical output 产生一个 contribution |
| `T.finalize_reducer(acc, dst)` | 完成 plan 所需的通信、应用 seed，并写入独立 destination |

### 5.2 精确的 reduction 语义

对每个 logical output `o`：

```text
result[o] = combine(
    optional_seed,
    combine_all(contribution(u) for u in logical_updates_of(o)),
)
```

没有 seed 时，从 combine identity 开始。动态 logical update instance 由下面的信息
共同确定：

- reducer epoch；
- update site；
- 外层 serial/pipeline iteration；
- `T.Parallel` logical indices；
- 没有被 loop layout 证明为 replica 的 physical execution。

layout 中的 `REP` 表示同一个 logical parallel iteration 的 physical replicas，
所以不能让每个 replica 都自动成为新的 contribution。

update 位于 `T.Parallel` 外时则没有 loop-created replica 可以消除：每个 active
participant 的执行都是独立 logical contribution。

### 5.3 Lifecycle

第一版 lifecycle 为：

```text
Allocated --init--> Active --update*--> Active --finalize--> Finalized
```

每个 allocation 必须恰好有：

- 一个显式 init；
- 零个或多个 update；
- 一个 out-of-place finalize。

double init、update after finalize、double finalize、reset 或 loop-local repeated epoch
都直接拒绝。

### 5.4 Identity 与 seed

identity 由 `combine op + dtype` 唯一决定：

| combine | identity |
|---|---|
| sum / bitwise-or / bitwise-xor | `0` |
| max | dtype 的最低值 |
| min | dtype 的最高值 |
| bitwise-and | 全 bits 为 1 |

seed 是 logical reduction 的一个额外输入，必须在 participant partial 合并后应用
一次。不能用 seed 初始化每个 participant，否则 sum seed 会被 participant 数量
放大。

### 5.5 Clean break

下面的 v1 写法不再支持：

```python
acc = T.alloc_reducer(..., replication="all")
T.clear(acc)
acc[i] += value
T.finalize_reducer(acc)
```

禁止 reducer 普通 load/store、`T.clear`、`T.fill`、`access_ptr`、alias 和
in-place finalize。sum 的负 contribution 写成：

```python
T.reducer_update(acc[i], -value)
```

而不是为 reducer update 引入 `-=` 语义。

## 6. Compiler design

### 6.1 IR 与 pipeline

高层 reducer allocation 使用虚拟 storage scope：

```text
local.reducer
```

它只能出现在三个 first-class operations 中，并且必须在 backend codegen 前被
materialize。

建议 pipeline 为：

```mermaid
flowchart LR
    A[Frontend TIR] --> B[VerifyReducerEpochs]
    B --> C[LayoutInference]
    C --> D[PlanAndMaterializeReducers]
    D --> E[LowerTileOp / PartitionLoop]
    E --> F[VerifyReducerLowered]
    F --> G[Backend codegen]
```

各阶段职责是：

1. `VerifyReducerEpochs` 只验证 lifecycle、access、effect 和 control flow。
2. `LayoutInference` 正常推导 contribution inputs、parallel loops 和 destination，
   不给 reducer handle 强行分配 Fragment layout。
3. `PlanAndMaterializeReducers` 选择 physical plan，并把 `local.reducer` 改写为
   真实 `local` storage。
4. `LowerTileOp / PartitionLoop` 实现 update 的贡献次数和 physical indexing。
5. `VerifyReducerLowered` 保证虚拟 scope、first-class ops 和 effect markers 已全部
   被消费。

### 6.2 Canonical FullParticipant baseline

当没有更强 ownership proof 时，每个 participant 分配一份完整 logical output
array：

```text
partial[logical_shape] per participant
```

init 将所有 slots 设为 identity。对于 `T.Parallel` 中的每个 update site，只有该
logical iteration 的 canonical replica 更新：

```cpp
if (REP == 0) {
  partial[index] = combine(partial[index], contribution);
}
```

finalize 对每个 logical output 做 participant-wide collective，再 combine seed，并
按照 destination layout 写回。

这个 baseline 的重要性质是：不同 update sites 可以拥有完全不同的 loop layouts。
每个 site 只负责消除自己的 replicas，不需要先求出一个覆盖整个 epoch 的复杂
ownership plan。

### 6.3 通用贡献次数标记

`ReducerUpdateOp` lowering 不把 reducer buffer 列表传给 `PartitionLoop`，而是在
combine statement 外附加通用 marker：

```text
tl.parallel_multiplicity = once_per_logical_iteration
```

`PartitionLoop` 已经能从 loop layout inverse 得到 `REP`，因此只需把 marker
lower 成 `REP == 0` guard。普通 Fragment work 不被 guard。

这使 loop partition 只理解副作用的执行次数，不需要理解 reducer 的业务语义。

### 6.4 Conservative LocalComplete fast path

对于：

```python
for i in T.Parallel(M):
    T.reducer_update(acc[i], A[i])
```

如果 destination layout 已知，planner 可以证明每个 destination replica 都能独立
构造自己 local slots 的完整结果。此时可以：

- 用 destination layout 物化 compact partial；
- 让所有 destination replicas 执行 update，不使用 `REP == 0`；
- finalize 直接复制 local physical slots；
- 不生成 AllReduce、barrier 或 workspace。

例如 `M=8, threads=128` 时，destination 可能是每线程一个 local slot、16 个
replicas。reducer storage 可以从每 participant 8 个 slots 缩成 1 个 slot。

第一版只有以下条件全部满足才选择 LocalComplete：

- 每个 update 位于已知 `T.Parallel` root 中；
- reducer shape、destination input shape 和 parallel loop shape 相同；
- update indices 与 parallel logical variables 逐维相等；
- thread、replicate 和 physical output extents 都是 compile-time known；
- loop 可以安全复制：没有普通 store、thread-private Fragment/local load 或
  non-pure call；
- 同一 parallel root 上的 LocalComplete candidates 请求完全相同的 layout。

需要强调：LocalComplete 不是“只选一个 physical owner”。它允许多个 destination
replicas 各自得到一份完整结果，所以不需要通信。

下面的模式不能使用第一版 LocalComplete：

```python
# k 被分配给多个 physical participants，每个 participant 只有 partial K。
for i, k in T.Parallel(M, K):
    T.reducer_update(acc[i], A[i, k])
```

它回退到 FullParticipant collective。下面的形式则可以：

```python
for i in T.Parallel(M):
    for k in T.serial(K):
        T.reducer_update(acc[i], A[i, k])
```

因为负责 `i` 的每个 destination replica 都执行完整 serial `K`。

Fragment/local contribution 也暂不进入 LocalComplete，因为其值可能取决于原有
physical layout。未来只有在 source Fragment ownership 与 destination ownership
兼容性可证明时才放开。

### 6.5 Future subgroup fast path

`Parallel(M, K) -> output[M]` 确实存在更高效的 subgroup reduction 机会，但它是
独立优化，不应成为 correctness 的前置条件。

未来可以从每个 update site 生成 normalized `ThreadGroupSignature`，描述：

- participant range；
- logical output 到 group 的 projection；
- group/lane mapping；
- reduction steps；
- uniform predicate；
- contribution Fragment compatibility。

只有一个 epoch 内所有 sites 的 signatures exact-compatible，且 backend 支持对应
width/barrier/workspace policy 时，才选择 subgroup plan。否则回退 baseline。

第一版不尝试对多个 signatures 求并、交或“近似兼容”。

## 7. Participant domain

participant domain 表示哪些 physical threads：

- 初始化 partial；
- 提供 contribution 或 identity；
- 到达 collective/barrier；
- 使用对应 workspace addressing。

第一版只支持 compiler-known contiguous：

```text
Range(thread_min, thread_extent)
```

该 range 来源于真实 execution scope，而不是 result Fragment replicate extent。
warp specialization 产生的 non-zero offset 也必须保留。

下面三个量不能混为一谈：

| quantity | meaning |
|---|---|
| reduction width | 一次 collective 实际组合多少 lanes |
| barrier arrive count | 有多少 threads 到达 barrier |
| workspace layout/stride | partials 如何在 workspace 排布 |

subgroup plan 中 reduction width 可以小于整个 participant range，但 barrier policy
不一定相同。backend contract 必须显式传递它们。

## 8. Correctness invariants

所有 lowering 必须维持以下不变量：

1. 每个 reducer allocation 恰好一个 init、一个 epoch 和一个 finalize。
2. `ReducerUpdateOp` 是 logical contribution 贡献次数的唯一来源。
3. layout replication 不能改变 logical contribution count。
4. partial state 在 finalize 前不是普通 Fragment value。
5. 每个 update site 的 correctness 可以独立证明。
6. participant range 来自 execution scope，不从 result layout 反推。
7. seed 对每个 logical output 恰好 combine 一次。
8. collective 位于 update predicates 和 representative guards 之外。
9. optimization proof 失败必须 fallback，不能生成近似代码。
10. backend codegen 看不到 `local.reducer`、first-class reducer ops 或未消费 marker。

## 9. 预期收益

| 维度 | 收益 |
|---|---|
| 用户语义 | 显式表达 init、logical contribution 和 finalize，不再依赖特殊 `+=` pattern |
| 正确性 | layout replication 不能静默放大 contribution；非法 lifecycle 和 access 可以提前诊断 |
| 性能 | 保留跨 tile accumulation、单次 finalize 的价值；ownership 明确时可完全消除 collective |
| 编译器设计 | logical reduction、participant domain、partial storage 和 result layout 分层规划 |
| 后端复用 | CUDA/ROCm 共用 epoch plan，只在 collective capability/emission 层分叉 |
| 可维护性 | 删除 layout inference、race verifier 和 `PartitionLoop` 中按 reducer buffer 猜语义的特例 |
| 可扩展性 | subgroup、segmented storage、batched finalize 都可以作为有 baseline 的独立优化加入 |

这里最重要的收益不是某个固定的代码生成技巧，而是建立一个优化可以失败的语义
边界。编译器不需要为了性能分析的完备性牺牲 correctness。

## 10. Tradeoffs

### 10.1 Correctness baseline 的成本

FullParticipant 为每个 participant 分配完整 logical output array，可能导致：

- register pressure 增加；
- large output spill 到 local memory；
- init/finalize 遍历较多 slots；
- 多 output 时生成较多 scalar collectives。

这是第一版有意接受的成本。它提供清晰、可验证的 reference lowering，也为后续
segmented/sparse storage planner 和 batched finalize 提供正确对照。

对于 footprint 过大的 case，初版可以明确拒绝，或者由策略阈值限制，而不是偷偷
采用无法证明的 ownership。

### 10.2 LocalComplete 的限制

LocalComplete 当前只覆盖 direct identity ownership。它不处理：

- affine/permuted output mapping；
- Fragment/local contribution；
- mixed side effects；
- 一个 Parallel root 上互相冲突的 destination layouts；
- subgroup communication。

这些限制是 proof boundary，不是用户语义限制。合法程序仍可进入 baseline。

### 10.3 Floating-point ordering

sum 的 reduction tree 可能随 physical plan 改变，因此不承诺 bitwise reproducible
floating-point result。不同 plan 必须满足同一 associative/commutative algebra
contract 和既有数值容差。

## 11. Alternatives considered

### 11.1 删除 reducer，统一使用 `T.reduce`

无法满足跨 tile/pipeline iteration 保存 partial、最后只通信一次的需求；或者需要
先物化完整 reduction domain，产生额外 storage 和带宽成本。

### 11.2 继续把 reducer 当 FullyReplicated Fragment

这会继续混淆“final value replicas 应该相等”和“partial states 尚未合并”。每个
compiler pass 都需要 reducer-specific exception，且 layout replication 仍可能改变
副作用的执行次数。

### 11.3 由用户指定 `replication="all" / "none"`

用户很难在 source level 正确描述所有 update sites、participant offsets 和
destination ownership。replication 是 physical policy，不应改变 logical reduction
语义，也不应成为 correctness knob。

### 11.4 从 finalize result layout 推导 reduction width

result layout 只描述结果放在哪里，不描述 contributions 由哪些 threads 产生。它
无法恢复 update site 的贡献次数、execution scope 或 barrier domain。

### 11.5 必须先求出完整 ownership plan 才允许 reducer

多个 update sites 可以来自不同、复杂的 layouts。把 plan merge 设为 correctness
前置条件会让编译器难以扩展，也容易在第一个 site 的 plan 被误当成整个 epoch plan
时产生错误结果。ownership plan 应只决定 fast path。

## 12. Prototype status and preliminary evidence

Draft PR #2901 已实现：

- first-class init/update/finalize 和 `local.reducer`；
- lifecycle、ordinary access、bounds、effect 和 replica-invariance verification；
- FullParticipant baseline 与通用贡献次数标记；
- compiler-known contiguous participant range；
- CUDA/ROCm shared high-level finalize contract；
- conservative LocalComplete compact-storage fast path；
- sum/max/min/bitwise builtins 和 logical seed。

subgroup fast path、multiple epochs、custom combine、general Fragment ownership 和
discontiguous participant set 尚未实现。

在 B300、128 threads 的初步 microbenchmark 中，LocalComplete cases 的结果为：

| case | legacy | v2 LocalComplete | legacy / v2 |
|---|---:|---:|---:|
| M8, 512 blocks | 7.168 us | 6.528 us | 1.098x |
| M32, 512 blocks | 7.168 us | 6.528 us | 1.098x |
| M128, 256 blocks | 7.168 us | 6.464 us | 1.109x |

三个 v2 kernels 都使用每 participant 一个 local slot，并且没有 AllReduce、named
barrier 或 workspace。FullParticipant fallback microbench 与 legacy 基本持平。

这些数据只说明 direct-ownership optimization 没有引入明显固定开销，不代表完整
workload 的最终性能结论。仍需要覆盖 dtype、output footprint、participant range、
occupancy 和 real kernels 的系统评估。

## 13. Validation plan

### 13.1 Numerical correctness

- `BD=8, threads=128` 返回 `36` 而不是 `576`。
- 多个 update sites 使用不同 loop layouts。
- predicated update、边界 tile 和空 contribution domain。
- 多个 pipeline iterations，只 finalize 一次。
- scalar 与多元素 logical output。
- sum/max/min/bitwise 和支持的 dtypes。
- non-zero seed 恰好应用一次。
- LocalComplete 的 inner serial reduction。

### 13.2 Planner decisions

- `Parallel(M) -> output[M]` 选择 LocalComplete。
- `Parallel(M, K) -> output[M]` 回退 FullParticipant。
- Fragment/local contribution 不错误选择 LocalComplete。
- 多个 reducers 共享 Parallel 时，相同 destination layout 可共同 fast path。
- 同一个 Parallel 上 destination layout 冲突时确定性 fallback。
- 不同 thread counts、replicate extents 和每线程多 physical slots。

### 13.3 Structural checks

- update marker 只包 reducer combine，不包普通 Fragment work。
- final collective 不位于 `REP == 0` 或 update predicate 内。
- LocalComplete codegen 没有 collective/barrier/workspace。
- `local.reducer`、first-class ops 和 markers 在 backend 前全部消失。
- 强制 baseline 与 auto strategy 得到一致的数值结果。

### 13.4 Performance

- 与 legacy 和强制 FullParticipant baseline 对比。
- 分别扫描 output size、reduction size、threads、blocks、dtype 和 combine op。
- 记录 register usage、spill、occupancy、workspace 与 latency。
- 对真实 GEMV/row-reduction/pipelined kernels 做 end-to-end evaluation。

## 14. Rollout

建议按以下顺序推进：

1. 固定 shared reduction algebra 和 first-class IR contract。
2. 合入 verifier 与 FullParticipant correctness baseline。
3. 全仓迁移到显式 init/update/out-of-place finalize，并删除 v1。
4. 单独合入 conservative LocalComplete fast path。
5. 在 baseline 可强制启用的前提下，研究 exact-signature subgroup fast path。

每个 performance optimization 都必须能够独立关闭并回到同一个 canonical
semantics，不能重新引入 v1 compatibility path。

## 15. 希望 reviewers 重点反馈的问题

1. first-class epoch 与 clean-break API 是否足够清晰？
2. 每 participant 完整 logical array 是否适合作为 canonical correctness baseline？
   是否需要在第一版加入静态 footprint 上限？
3. participant domain 第一版限制为 compiler-known contiguous `Range` 是否可接受？
4. LocalComplete 的 direct identity proof boundary 是否足够保守、容易 review？
5. Fragment ownership compatibility 应该是下一步，还是优先做 subgroup signature？
6. backend contract 中 reduction width、barrier count 和 workspace policy 是否还需要
   更明确的独立类型？
7. 哪些真实 kernels 最适合作为性能与表达能力的验收 workload？
8. FlashAttention 的 deferred row-sum 是否值得后续引入 first-class uniform state
   transform，还是更适合实现成 compiler-known structured monoid？

## 16. 提议采纳的决定

本 proposal 请求确认以下设计方向：

1. reducer 是 first-class deferred reduction epoch，不是普通 Fragment buffer。
2. v2 只支持显式 init/update/out-of-place finalize，不保留 legacy lowering。
3. FullParticipant 是所有合法 reducer 的 canonical correctness baseline。
4. participant domain 第一版只支持 compiler-known contiguous `Range`。
5. LocalComplete、subgroup 和 storage compression 都是可回退的 physical plans。
6. layout 负责 physical placement/communication，不能定义 logical contribution
   的贡献次数。

一句话概括：

> reducer 负责表达“哪些 logical contributions 在一个 epoch 中被组合”；planner
> 负责选择“如何存储和通信”。两者分开之后，correctness 不再依赖某一种 layout
> inference 恰好成功。
