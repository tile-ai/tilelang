# SunMMIO TileLang Kernel 快速入门

[English](sunmmio_tilelang_getting_started.md) | [完整用户手册](sunmmio_tilelang_user_guide_zh_cn.md)

## SunMMIO A4E 架构总览

### 架构特征

![SunMMIO NPU 架构图](imgs/sunmmio_a4e_architecture.png)

*SunMMIO NPU 的整体硬件架构，包括 2D mesh、多类计算单元和片上存储层次。*

SunMMIO NPU 的硬件架构特征为：

- 芯片上有 16 个 Core，组成 2D mesh。
- 每个 Core 上有 Tensor Core、Vector Unit、ODMA。
- Core 之间可以通过 HLink / VLink 做广播和数据交换。
- 片上 SRAM 分成 `ASRAM / WSRAM / RSRAM`。

### 编写 SunMMIO TileLang Kernel 时的 Mental Model

![SunMMIO TileLang Mental Model](imgs/sunmmio_kernel_mental_model.png)

*编写 SunMMIO TileLang kernel 时的 Mental Model，包括分片、执行和通信三层视角。*

给 SunMMIO 写 TileLang kernel 时，需要有如下 Mental Model：

1. 同样一个 kernel，运行在不同的 Core 上，处理不同的数据
2. Host 上的数据被 sharding 到不同 Core 各自的 DRAM 上，每个 Core 只拥有一部分数据
3. 存在某些情况，一个 Core 需要其他 Core 的数据，即需要核间通信（`all_gather`、`broadcast`、`put`）

另外，考虑到 Tensor Core、Vector Core 的硬件特性，还需要：

4. 将并行的标量运算表达为 tile 级别的运算
5. 理解 DRAM Tensor 的默认 layout，并在需要对接固定格式或调试时显式指定 layout

## 一个例子

下面用一个分布式矩阵乘的例子描述写 SunMMIO kernel 时最关键的几件事：

- `MeshTensor` / `Placement`
- `T.Kernel`
- 核间通信
- `layout`

```python
import tilelang.language as T
from tilelang.layout import make_zz_layout

def matmul_persistent(M, N, K, block_M, block_N, block_K, num_stages, dtype=T.bfloat16, accum_dtype=T.float32):
    placement = T.placement.full_shard(row_dim=0, col_dim=1)
    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), placement, dtype, layout=A_layout),
        B: T.MeshTensor((K, N), placement, dtype, layout=B_layout),
        C: T.MeshTensor((M, N), placement, accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as _cid:
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape

            A_shared_dist = T.alloc_shared((block_M, block_K * T.mesh_ncols()), dtype)
            B_shared_dist = T.alloc_shared((block_K * T.mesh_nrows(), block_N), dtype)
            C_shared = T.alloc_shared((block_M, block_N), accum_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_shared)
                    for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=num_stages):
                        T.comm.all_gather(
                            A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K],
                            A_shared_dist,
                            direction="horizontal",
                            axis=-1,
                        )
                        T.comm.all_gather(
                            B[k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N],
                            B_shared_dist,
                            direction="vertical",
                            axis=0,
                        )
                        T.gemm(A_shared_dist, B_shared_dist, C_shared)

                    T.copy(C_shared, C[bx * block_M, by * block_N])

    return main
```

这个 factory 返回 `PrimFunc`。由于 `target="auto"` 不会探测该后端，需要显式指定 SunMMIO target：

```python
import tilelang

func = matmul_persistent(1024, 1024, 1024, 32, 32, 128, 3)
kernel = tilelang.compile(func, target="sunmmio")
```

后面几节都围绕这个例子展开。

## Sharding（Tensor 在 Core Mesh 上的分片）

### 1. Sharding 机制

![Sharding 方法举例](imgs/sunmmio_mesh_sharding.png)

*Tensor 在 Core Mesh 上的 Sharding 方法示意，说明逻辑维度如何映射到 mesh 方向。*

我们用 `T.MeshTensor` 和 `T.placement` 定义全局 tensor 在 mesh 上怎么分。

最常见的几种写法是：

- `T.placement.full_shard(row_dim, col_dim)`：沿 mesh 的行、列方向切分指定 tensor 维度
- `T.placement.row_shard(dim)`：沿行方向切分，在列方向复制
- `T.placement.col_shard(dim)`：沿列方向切分，在行方向复制
- `T.placement.replicated()`：在所有 Core 上复制
- `T.placement.mesh_as_line(dim)`：把 mesh 按 row-major 顺序视为一维并切分指定维度

在这个 GEMM 例子里：

```python
placement = T.placement.full_shard(row_dim=0, col_dim=1)

A: T.MeshTensor((M, K), placement, dtype, layout=A_layout)
B: T.MeshTensor((K, N), placement, dtype, layout=B_layout)
C: T.MeshTensor((M, N), placement, accum_dtype, layout=C_layout)
```

这里的 `T.placement.full_shard(0, 1)` 表达：

- 第 0 维沿 mesh 的 row 方向切分
- 第 1 维沿 mesh 的 column 方向切分

### 2. 写 kernel 时怎么用

使用 `MeshTensor` 时，`global_shape` 是完整逻辑 shape，`local_shape` 是每个 Core 分配到的本地 slot shape。

因此常见写法是：

```python
# A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, ...)
sharded_M, sharded_K = A.local_shape
_, sharded_N = B.local_shape
```

后面的循环边界都基于这些局部 shape。

### 3. 这个例子里的分片含义

这个 GEMM 例子里，单个 Core 不持有完整的 `A` 和 `B`，而是 `A` 和 `B` 的 shard。

- `A` 在列方向上需要后续做横向收集
- `B` 在行方向上需要后续做纵向收集
- `C` 是每个 Core 负责写回的输出分块

这也是后面 `all_gather` 会出现的原因。

### 4. 另外一个具体 Sharding 例子

假设有一个 shape 为 `(64, 32)` 的 tensor：

```python
C: T.MeshTensor(
    (64, 32),
    T.placement.full_shard(row_dim=0, col_dim=1),
    (4, 4),
    dtype,
)
```

这里假设 mesh 是 `4 x 4`。

- `row_dim=0` 表示第 0 维沿 mesh 的行方向切分
- `col_dim=1` 表示第 1 维沿 mesh 的列方向切分

所以：

- 第 0 维 `64` 被 4 行 Core 切开，每行拿到 `16`
- 第 1 维 `32` 被 4 列 Core 切开，每列拿到 `8`

每个 Core 看到的 local shard shape 是：

```python
(16, 8)
```

如果把 Core 坐标记成 `(row, col)`，那么：

- `(0, 0)` 拿到 `C[0:16, 0:8]`
- `(0, 1)` 拿到 `C[0:16, 8:16]`
- `(0, 2)` 拿到 `C[0:16, 16:24]`
- `(0, 3)` 拿到 `C[0:16, 24:32]`
- `(1, 0)` 拿到 `C[16:32, 0:8]`
- `(1, 1)` 拿到 `C[16:32, 8:16]`
- `...`
- `(3, 3)` 拿到 `C[48:64, 24:32]`

![full_shard(0, 1) 示例](imgs/sunmmio_full_shard_example.png)

*`T.placement.full_shard(0, 1)` 的具体示例，展示 `(64, 32)` tensor 在 `4 x 4` mesh 上的实际切分结果。*

这就是 `MeshTensor` 的基本工作方式：先定义全局 shape，再定义 mesh 上的切分规则，然后每个 Core 自动看到自己的 local shard。

## Kernel 启动

### 1. `T.Kernel()` 在做什么

这个例子里：

```python
with T.Kernel() as _cid:
    ...
```

`T.Kernel()` 使用目标的符号化 mesh，在整个 mesh 上启动 kernel，每个 Core 执行同一份代码。

`_cid` 是当前 Core 的线性 id。这个例子里没有直接使用 `_cid`，因为数据分配已经由 `MeshTensor` 完成了。

### 2. 为什么这里只写一个 `T.Kernel`

这个例子没有再额外写一层“把不同 Core 分配到不同 tile”的循环。原因是分片已经提前体现在 `MeshTensor` 上了。

所以这里的执行顺序是：

1. `T.Kernel()` 把同一份 kernel 发到所有 Core
2. 每个 Core 通过 `MeshTensor` 自动拿到自己的 shard
3. 每个 Core 在自己的 shard 上继续做 block 级循环

### 3. 这个例子里的 block 循环

```python
for bx in T.serial(T.ceildiv(sharded_M, block_M)):
    for by in T.serial(T.ceildiv(sharded_N, block_N)):
        ...
```

这里的 `bx`、`by` 是当前 Core 内部的 block 坐标。它们由当前 Core 上的数据切片大小 `sharded_M`、`sharded_N` 决定。

## 核间通信

### 1. 这个例子为什么需要核间通信

GEMM 里每个 Core 只拿到 `A` 和 `B` 的局部 shard。要让当前 Core 完成一轮 `T.gemm`，还需要把同一行或同一列上的其他 shard 收集过来。

这个例子使用的是 `all_gather`。

### 2. A 的 `all_gather`

```python
T.comm.all_gather(
    A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K],
    A_shared_dist,
    direction="horizontal",
    axis=-1,
)
```

这里的含义是：

- 当前 Core 先取出自己持有的 `A` 子块
- 沿着 mesh 的水平方向做收集
- 收集后的结果沿最后一维拼接到 `A_shared_dist`

所以 `A_shared_dist` 的 shape 是：

```python
(block_M, block_K * T.mesh_ncols())
```

### 3. B 的 `all_gather`

```python
T.comm.all_gather(
    B[k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N],
    B_shared_dist,
    direction="vertical",
    axis=0,
)
```

这里的含义是：

- 当前 Core 先取出自己持有的 `B` 子块
- 沿着 mesh 的垂直方向做收集
- 收集后的结果沿第 0 维拼接到 `B_shared_dist`

所以 `B_shared_dist` 的 shape 是：

```python
(block_K * T.mesh_nrows(), block_N)
```

### 4. 通信之后即可开始 `T.gemm`

通信结束后：

- `A_shared_dist` 拥有一整行需要参与 GEMM 的 `A` 数据
- `B_shared_dist` 拥有一整列需要参与 GEMM 的 `B` 数据

这时就可以直接做：

```python
T.gemm(A_shared_dist, B_shared_dist, C_shared)
```

这个例子里，通信和计算的配合关系非常清楚：

1. 先从 `MeshTensor` 里切出当前 Core 的局部 block
2. 做 `all_gather`
3. 用 gather 后的 shared buffer 做 GEMM

### 5. 前端可写的核间通信语句

当前前端主要可以写下面几类通信语句：

- `T.comm.broadcast(src, dst, src_core, direction=...)`
- `T.comm.put(src, dst, src_core, dst_core)`
- `T.comm.all_gather(send_buffer, recv_buffer, direction=..., axis=...)`
- `T.comm.all_reduce(buffer, out, reduce_type, direction=..., clear=...)`

它们的用途分别是：

- `broadcast`：一个源 Core 向一行、一列或整个 mesh 广播数据
- `put`：一个源 Core 向一个目标 Core 发送数据
- `all_gather`：多个 Core 的局部数据收集到一个更大的接收 buffer
- `all_reduce`：对多个 Core 上相同位置的元素做归约，并把结果写到本地 shape 相同的输出 buffer；它不会归约本地 buffer 内的某个维度

## Layout

### 1. Layout 是什么

layout 描述的是逻辑排布到物理排布的映射。逻辑排布指用户看到的 tensor 下标，例如 `A[i, j]`；物理排布指这份 tensor 在 DRAM 或片上 buffer 里的实际存储顺序。

同样的逻辑 shape，可以有不同的物理排布。常见的几种含义如下：

- row major：行优先。最后一维连续。
- col major：列优先。第一维连续。
- ZZ：按 block 切分后，再按块组织存储。SunMMIO 的 GEMM 场景里经常用这个。

当前前端已经有的 SunMMIO 相关 layout constructor 包括：

- `make_row_major`
- `make_aligned_row_major`
- `make_zz_layout`
- `make_zn_layout`
- `make_zzz_layout`
- `make_nzz_layout`

其中：

- `make_row_major` 是普通 row major
- `make_aligned_row_major` 是带对齐约束的 row major
- `make_zz_layout` 是 blockwise row-major
- `make_zn_layout` 是 blockwise col-major
- `make_zzz_layout` 是 clustered row-major
- `make_nzz_layout` 是 clustered col-major

### 2. 什么时候需要显式 layout

对 SunMMIO 来说，layout 会影响：

- Tensor Core / Vector Core 访问是否合法；
- DMA 访存是否对齐；
- all-gather / broadcast 后的数据排列是否匹配后续算子。

用户主要关心 **DRAM** 中 Tensor 的 layout，Shared Memory 中 Tensor 的 layout 由编译器推导。`MeshTensor` 不显式传入 `layout` 时，普通 dtype 的 rank-1 tensor 默认使用 1024-byte 对齐的 row-major，rank 大于等于 2 的 tensor 默认在最后两维使用 32x32 分块的 ZZ layout。

大多数普通 kernel 可以使用默认 layout。需要对接外部固定格式、选择非默认分块维度或调试 layout 时，再显式传入 layout。本文示例显式构造 ZZ layout，是为了完整展示该接口。

### 3. 显式 layout 的常见写法

这个例子里的 layout 写法是：

```python
from tilelang.layout import make_zz_layout

A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

A: T.MeshTensor((M, K), placement, dtype, layout=A_layout)
```

`make_zz_layout` 的函数定义是：

```python
make_zz_layout(shape_or_buffer, axes=None, block_shape=(32, 32))
```

三个参数分别表示：

- `shape_or_buffer`：输入 tensor 的 shape，或者一个已有的 buffer
- `axes`：哪些维度参与 ZZ 分块；如果不写，默认取最后两个维度
- `block_shape`：每个 block 的形状，默认是 `(32, 32)`

例如：

```python
make_zz_layout((M, K), [0, 1], (32, 32))
```

表示：

- 对 shape 为 `(M, K)` 的 tensor 建立 ZZ layout
- 第 0 维和第 1 维参与 block 划分
- 每个 block 的大小是 `32 x 32`

- layout 写在 `MeshTensor` 上
- layout 描述的是 DRAM 上 tensor 的布局

### 经验法则

优先使用 `MeshTensor` 的默认 layout。只有在外部数据格式、非默认分块或调试需求明确时，才显式使用 `make_zz_layout` 等 layout constructor。

## FlashAttention 例子

下面先把 `examples/flash_attention/sunmmio_example_gqa_fwd_bhsd.py` 里的核心函数贴出来。后面 `T.Tiles` 一节直接基于这段代码说明。

```python
import tilelang.language as T
from tilelang.layout import make_zz_layout


def flashattn(batch, heads, seq_len, dim, groups=1, block_M=64, block_N=64, num_stages=0):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    placement = T.placement.full_shard(row_dim=0, col_dim=2)

    Q_layout = make_zz_layout(q_shape, [1, 3], (32, 32))
    K_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    V_layout = make_zz_layout(kv_shape, [1, 3], (32, 32))
    O_layout = make_zz_layout(q_shape, [1, 3], (32, 32))

    @T.prim_func
    def main(
        Q: T.MeshTensor(q_shape, placement, dtype, layout=Q_layout),
        K: T.MeshTensor(kv_shape, placement, dtype, layout=K_layout),
        V: T.MeshTensor(kv_shape, placement, dtype, layout=V_layout),
        Output: T.MeshTensor(q_shape, placement, dtype, layout=O_layout),
    ):
        with T.Kernel() as _cid:
            sharded_batch = Q.local_shape[0]
            sharded_heads = Q.local_shape[2]

            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_shared([block_M, block_N], accum_dtype)
            acc_s_cast_local = T.alloc_shared([block_M, block_N], dtype)
            acc_s_cast = T.alloc_shared([block_M, block_N], dtype)
            acc_o = T.alloc_shared([block_M, dim], accum_dtype)
            scores_max = T.alloc_shared([block_M], accum_dtype)
            scores_max_prev = T.alloc_shared([block_M], accum_dtype)
            scores_scale = T.alloc_shared([block_M], accum_dtype)
            scores_sum = T.alloc_shared([block_M], accum_dtype)
            logsum = T.alloc_shared([block_M], accum_dtype)

            for bz in T.serial(sharded_batch):
                for by in T.serial(sharded_heads):
                    for bx in T.serial(T.ceildiv(seq_len, block_M)):
                        T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
                        T.fill(acc_o, 0)
                        T.fill(logsum, 0)
                        T.fill(scores_max, -T.infinity(accum_dtype))

                        loop_range = T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))

                        for k in T.Pipelined(loop_range, num_stages=num_stages):
                            T.copy(K[bz, k * block_N : (k + 1) * block_N, by // groups, :], K_shared)
                            for i, j in T.Tiles([block_M, block_N]):
                                acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

                            T.copy(scores_max, scores_max_prev)
                            T.fill(scores_max, -T.infinity(accum_dtype))
                            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                            for i in T.Tiles([block_M]):
                                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                            for i in T.Tiles([block_M]):
                                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                            for i, j in T.Tiles([block_M, block_N]):
                                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                            T.reduce_sum(acc_s, scores_sum, dim=1)
                            for i in T.Tiles([block_M]):
                                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                            for i, j in T.Tiles([block_M, block_N]):
                                acc_s_cast_local[i, j] = acc_s[i, j]
                            T.copy(acc_s_cast_local, acc_s_cast)

                            for i, j in T.Tiles([block_M, dim]):
                                acc_o[i, j] *= scores_scale[i]

                            T.copy(V[bz, k * block_N : (k + 1) * block_N, by // groups, :], V_shared)
                            T.gemm(acc_s_cast, V_shared, acc_o)

                        for i, j in T.Tiles([block_M, dim]):
                            acc_o[i, j] /= logsum[i]
                        T.copy(acc_o, O_shared)
                        T.copy(O_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main
```

## `T.Tiles`

### 1. 为什么需要 `T.Tiles`

`sunmmio_example_gqa_fwd_bhsd.py` 代码里有很多逐元素计算：

- `scores_max` 更新
- `scores_scale` 计算
- `exp2` 变换
- `acc_o` 的逐元素缩放和归一化

这些逻辑在数学上是标量运算，但在 SunMMIO 上，目标是把它们组织成符合 4096-bit 位宽的 vector 计算。

`T.Tiles` 就是做这件事的前端入口。

### 2. 这份例子里的 `T.Tiles`

这份 flash attention 例子里，`T.Tiles` 的典型写法如下：

```python
for i, j in T.Tiles([block_M, block_N]):
    acc_s[i, j] = T.if_then_else(
        bx * block_M + i >= k * block_N + j,
        0,
        -T.infinity(acc_s.dtype),
    )

for i in T.Tiles([block_M]):
    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

for i in T.Tiles([block_M]):
    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

for i, j in T.Tiles([block_M, block_N]):
    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

for i, j in T.Tiles([block_M, block_N]):
    acc_s_cast_local[i, j] = acc_s[i, j]

for i, j in T.Tiles([block_M, dim]):
    acc_o[i, j] *= scores_scale[i]

for i, j in T.Tiles([block_M, dim]):
    acc_o[i, j] /= logsum[i]
```

这些代码的共同点很明确：

- 输入输出都在 shared buffer 上
- 运算是 tile 内部的大量并行逐元素逻辑
- 后端需要把它们识别成 tile 级 vector 计算

### 3. 对比标量写法和 `T.Tiles` 的写法

如果不用 `T.Tiles`，同样的逻辑会写成普通标量循环。

例如 causal mask：

```python
for i, j in T.Parallel(block_M, block_N):
    acc_s[i, j] = T.if_then_else(
        bx * block_M + i >= k * block_N + j,
        0,
        -T.infinity(acc_s.dtype),
    )
```

再比如 `scores_scale`：

```python
for i in T.Parallel(block_M):
    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
```

这种写法能表达计算，但它表达的是很多独立的标量操作。

同样的逻辑，用 `T.Tiles` 写成：

```python
for i, j in T.Tiles([block_M, block_N]):
    acc_s[i, j] = T.if_then_else(
        bx * block_M + i >= k * block_N + j,
        0,
        -T.infinity(acc_s.dtype),
    )
```

以及：

```python
for i in T.Tiles([block_M]):
    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
```

可以看到，对于一个 buffer 上的标量计算，往往可以表达为 tile 级别计算，这样就可以让后端按 tile 去 lower 成 Vector Core 计算。

### 4. `T.Tiles` 的初衷

`T.Tiles` 初衷是把并行的标量运算组织成 tile 级运算，让后端能够按统一的 tile 结构去处理，一是方便访存，二是可以一步 lower 成符合 4096-bit 位宽的 vector 计算。

在这份 flash attention 例子里：

- `T.Tiles([block_M, block_N])` 描述一个二维 tile
- `T.Tiles([block_M])` 描述一个一维 tile
- 后端看到这些 tile 后，可以继续把它们规整成稳定的 tile-level loop

### 5. 什么时候用 `T.Tiles`

如果目标是 buffer 内部的大量并行标量计算，用 `T.Tiles`。下面这些场景直接用 `T.Tiles`：

- shared buffer 上的逐元素算术
- tile 内的缩放、归一化、激活
- tile 内 mask 写入
- tile 内 dtype cast

### 注意

Tile 大小不用用户手动指定，编译器会根据 buffer layout 和它参与的计算，自动推断 tile 大小。

## 总结

总结下来，写 SunMMIO TileLang kernel 需要注意以下几点：

- SunMMIO 架构是多 Core mesh。一个 kernel 会在多个 Core 上同时执行。每个 Core 只处理自己的数据块。
- placement 决定全局 tensor 如何落到 Core mesh 上。`MeshTensor` 和 `T.placement` 定义的就是这件事。
- layout 决定 tensor 从逻辑坐标到物理存储的映射。普通 kernel 优先使用默认 layout，外部格式或非默认分块等高级场景再显式指定。
- tile loop 用来表达 tile 内的大量并行标量计算。`T.Tiles` 的作用是把这些计算组织成 tile 级结构，再交给后端 lower 成 Vector Core 计算。
- 核间通信要由前端显式写出。常用接口是 `broadcast`、`put`、`all_gather`、`all_reduce`。GEMM 例子里最核心的是 `all_gather`。

## 相关文档

下面这些文档可以配合本文一起参考：

- [SunMMIO TileLang 用户手册](sunmmio_tilelang_user_guide_zh_cn.md)
- [安装文档](../get_started/Installation.md)
- [TileLang 编程说明](../programming_guides/overview.md)
- [SunMMIO TileLang kernel 示例](https://github.com/SUNMMIO/Tilelang/tree/tilelang_mesh_main/examples)
