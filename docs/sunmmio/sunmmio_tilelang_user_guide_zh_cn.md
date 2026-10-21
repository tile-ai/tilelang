# SunMMIO TileLang 用户指南

[English](sunmmio_tilelang_user_guide.md) | [快速入门](sunmmio_tilelang_getting_started_zh_cn.md)

本文面向已经熟悉基础 TileLang、但需要在 SunMMIO 目标上编写、迁移或调试 kernel 的用户。重点说明用户在前端编程时需要理解的硬件结构、编程模型、前端函数和常见写法。

文档里的硬件名称使用 `SunMMIO`，TileLang target 使用规范的小写字符串 `sunmmio`；target 解析器仍兼容 `Sunmmio`。

## 1. 硬件结构特点

### 1.1 架构总览

SunMMIO NPU 的核心硬件特征可以概括为：多核 2D mesh、每 core 独立 DRAM、分层片上 SRAM、多类硬件引擎。

SunMMIO 采用 core-level 的执行组织。每个 core 拥有自己的计算、搬运和片上存储资源，core 之间通过二维 mesh 互联。用户编写 kernel 时通常需要显式考虑当前 core 负责哪一块数据，以及需要和哪些相邻或同组 core 交换数据。

硬件上数据流是显式的：数据需要进入对应 SRAM 区域，再由 ODMA、Tensor Core、Vector Core、HLink/VLink 等不同硬件单元消费或转移。因此，SunMMIO 程序通常围绕数据分布、数据搬运、局部计算和核间通讯来组织。

### 1.2 2D Mesh Core

SunMMIO 由多个 core 组成二维 mesh。当前默认设备配置为 `(4, 4)`，即 4 行、4 列，共 16 个 core。

mesh 有明确的 row / column 方向。横向方向连接同一行内的 core，纵向方向连接同一列内的 core。全 mesh 范围的数据移动通常需要横向与纵向链路组合完成。

core 可以用二维坐标 `(row, col)` 表示，也可以线性化为 id：

```text
core_id = row * ncol + col
```

这个坐标体系是数据分布和核间通讯的基础。

### 1.3 每 Core 的计算与搬运引擎

每个 core 内部有多类可并行工作的硬件单元：

- Tensor Core：面向 GEMM / MMA。
- Vector Core：面向 tile 内逐元素算术、比较、类型转换、fill/clear、局部归约等批量处理。
- ODMA：负责本 core DRAM 与片上 SRAM 之间的块级数据搬运。
- HLink / VLink：负责 core 间横向/纵向核间通讯。

这些引擎是独立硬件资源，可以并行工作。用户在组织 kernel 时，需要理解计算、搬运和核间通讯可以形成重叠执行空间。

### 1.4 每 Core DRAM 与 SRAM Scope

SunMMIO 的存储层次包含每个 core 配套的 DRAM 和 core 内 SRAM。每个 core 的 DRAM 主要承载该 core 负责的输入 shard、输出 shard 以及片上 SRAM 之外需要保存的数据。

相比片上 SRAM，DRAM 容量更大，但访问延迟更高，带宽也更容易成为瓶颈。片上 SRAM 离计算单元更近，容量有限，适合承载 tile / block 级计算所需的数据。

典型数据流是：每个 core 从自己的 DRAM 搬入某个片上 SRAM 区域，由 Tensor Core、Vector Core 或核间通讯链路消费；计算或核间通讯产生的结果再经由片上 SRAM 写回该 core 的 DRAM。

SunMMIO 的片上 SRAM 按用途分成不同 scope：

- `shared.asram`：主要服务 Tensor Core 的左操作数数据通路。
- `shared.wsram`：主要服务 Tensor Core 的右操作数数据通路。
- `shared.rsram`：承载 Tensor Core 的结果操作数、Tensor Core 输出结果和 Vector Core 数据处理，同时承担 DRAM、Tensor Core 结果和 Vector Core 结果之间的数据交换中枢。

不同 SRAM 区域对应不同硬件数据路径。某个硬件单元能否直接消费一块数据，取决于数据当前所在的 SRAM 区域。

### 1.5 Tensor Core / MMA

Tensor Core 是 core 内面向矩阵乘/矩阵累加的计算单元。它通过固定的片上 SRAM 路径获取矩阵数据。

在 Tensor Core 的 MMA 数据通路中：

- `ASRAM` 为 Tensor Core 提供左操作数。
- `WSRAM` 为 Tensor Core 提供右操作数。
- `RSRAM` 通常承载结果操作数以及 Tensor Core 输出结果，并作为后续 Vector Core 处理、核间通讯或写回 DRAM 的中转区域。结果操作数承担 GEMM / MMA 的累加和输出角色。

ASRAM、WSRAM、RSRAM 的区分既是 TileLang scope，也对应 Tensor Core 的硬件数据路径。

### 1.6 ODMA 与块级数据搬运

ODMA 是每个 core 内负责本 core DRAM 与片上 SRAM 之间块级数据搬运的硬件单元。它服务的是本 core 的 DRAM 访问路径，主要完成“片外 DRAM shard 与片上 SRAM buffer/region 之间”的数据移动。

在典型 kernel 中，ODMA 负责把输入 shard 从本 core DRAM 搬入片上 SRAM，也负责把片上计算得到的输出 region 写回本 core DRAM。常见方向包括 DRAM 到 RSRAM、DRAM 到 ASRAM/WSRAM，以及 RSRAM 到 DRAM。具体可用路径受 SRAM scope 和硬件数据通路约束，用户在编写 copy 时需要让目标 SRAM 区域匹配后续消费者。

ODMA 独立于 Tensor Core、Vector Core 和 HLink/VLink。一个 core 内可以形成“搬运、计算、核间通讯”并行工作的执行形态：ODMA 准备下一块数据时，Tensor Core 可以消费已经准备好的矩阵 tile，Vector Core 可以处理片上中间结果，HLink/VLink 可以在 core 间传递片上 buffer。用户层面需要明确数据当前位于 DRAM 还是某个 SRAM scope，以及该数据下一步会被哪个硬件单元消费。

### 1.7 Tile 执行粒度与 Vector Core

SunMMIO 的局部计算和访存常以 tile 为基本组织单位。tile 是一块具有明确形状、对齐和访问方向的数据区域，既影响片上 SRAM 中的数据组织，也影响 Vector Core 处理数据的方式。

Vector Core 是 core 内面向 tile 级数据处理的计算单元。它适合处理 tile 内的算术、比较、类型转换、fill/clear、mask 处理、局部归约以及 GEMM 后处理等操作。例如 bias 加法、scale、clamp、局部 reduce、softmax 中的部分片上处理，都属于 Vector Core 适合承载的计算形态。

Vector Core 通常围绕 RSRAM 中的数据工作。Tensor Core 产生的结果操作数进入 RSRAM 后，可以继续由 Vector Core 做后处理，再写回 DRAM 或参与 HLink/VLink 核间通讯。DRAM 搬入 RSRAM 的普通输入 tile，也可以直接由 Vector Core 处理。

Tile 的形状、对齐和访问方向会影响 Vector Core 的执行效率。常见高效写法会让 tile 形状贴合硬件处理宽度，并尽量让片上访问保持规则。对用户而言，选择合适的 tile shape、保持输入输出 region 对齐、避免复杂散乱访问，是写出稳定 SunMMIO kernel 的重要前提。

### 1.8 HLink / VLink 核间通讯

SunMMIO 的核间通讯通过专用链路完成：

- HLink：连接同一行内的 core，主要承载横向核间通讯。
- VLink：连接同一列内的 core，主要承载纵向核间通讯。

HLink / VLink 独立于 ODMA、Tensor Core 和 Vector Core。它们用于在 core 之间传递片上 buffer 数据，常见核间通讯形态包括 broadcast、point-to-point put、all-gather 和 all-reduce。

横向核间通讯通常沿 HLink 在同一行传播，例如把某个 core 上的分块数据广播给同一行的其他 core。纵向核间通讯通常沿 VLink 在同一列传播，例如把某个 core 上的分块数据广播给同一列的其他 core。需要覆盖整个 mesh 的核间通讯会组合横向和纵向链路，形成多阶段数据移动。

核间通讯除了数据传输本身，还涉及源 core、目标 core、核间通讯方向、接收 buffer 形状以及跨 core 可见性。用户在写程序时应明确当前数据位于哪个 core、要沿哪个方向传输，以及接收端 buffer 是否预留了足够的形状。

## 2. 编程模型

### 2.1 Kernel 执行模型概览

SunMMIO kernel 采用 SPMD（Single Program, Multiple Data）编程形式：所有 core 执行同一份 kernel 程序，输入输出数据由 `MeshTensor` 的 placement 分配到各个 core；kernel 内通过 `cid`、`row`、`col` 定位当前 core 对应的 shard，并决定片上 buffer 使用方式和核间通讯角色。用户通常不需要为每个 core 编写不同的程序分支，而是在同一份程序中用 core 坐标描述数据划分和协作方式。

SunMMIO kernel 也是 persistent kernel：一次 kernel 启动后，各个 core 常驻执行同一份程序，并在 kernel 内通过循环处理分配给自己的 tile 或 work item。已经启动的 core 会持续完成本次 kernel 覆盖的工作，直到程序执行到 kernel 结束位置。一次 kernel 启动中的所有 core 都执行结束后，整个 kernel 才一起完成退出。

在 SPMD 模型下，`MeshTensor` 描述逻辑完整 tensor 的全局形状和分布方式；进入 kernel 后，当前 core 访问的是 sharding 之后分配到该 core 的本地 shard，通常对应该 core 的 DRAM 侧数据分片。跨 core 数据依赖通过 HLink / VLink 相关接口显式表达，片上计算仍然写成当前 core 上的局部计算。

下面的简化骨架展示了 SunMMIO kernel 的典型结构，省略了边界处理和完整循环映射细节。

```python
import tilelang
import tilelang.language as T
from tilelang.layout import make_zz_layout

M, N, K = 1024, 1024, 1024
BM, BN, BK = 32, 32, 128
dtype = "float16"
accum_dtype = "float32"

A_placement = T.placement.row_shard(0)
B_placement = T.placement.col_shard(1)
C_placement = T.placement.full_shard(0, 1)
A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

A_ty = T.MeshTensor((M, K), A_placement, dtype, layout=A_layout)
B_ty = T.MeshTensor((K, N), B_placement, dtype, layout=B_layout)
C_ty = T.MeshTensor((M, N), C_placement, accum_dtype, layout=C_layout)

@tilelang.jit(target="sunmmio")
def gemm_kernel():
    @T.prim_func
    def main(A: A_ty, B: B_ty, C: C_ty):
        with T.Kernel() as _cid:
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape

            A_shared = T.alloc_shared((BM, BK), dtype)
            B_shared = T.alloc_shared((BK, BN), dtype)
            C_shared = T.alloc_shared((BM, BN), accum_dtype)

            for bm in T.serial(T.ceildiv(sharded_M, BM)):
                for bn in T.serial(T.ceildiv(sharded_N, BN)):
                    T.clear(C_shared)
                    for bk in T.Pipelined(T.ceildiv(sharded_K, BK), num_stages=3):
                        T.copy(A[bm * BM, bk * BK], A_shared)
                        T.copy(B[bk * BK, bn * BN], B_shared)
                        T.gemm(A_shared, B_shared, C_shared)

                    T.copy(C_shared, C[bm * BM, bn * BN])

    return main
```

SunMMIO 的编程模型可以概括为：

1. 用 `target="sunmmio"` 选择目标。
2. 用 `MeshTensor`、placement 和 layout 描述逻辑 tensor 如何分布到 2D mesh 以及如何在 DRAM 中组织。
3. 用 `T.Kernel()`（等价于 `T.Kernel(T.mesh_ncores())`）在符号化目标 mesh 上启动；只有算法需要显式坐标时才映射 `cid -> (row, col)`。
4. 每个 core 从 sharding 分配给自己的 DRAM shard 搬入片上 SRAM。
5. 需要跨 core 数据时，用 HLink / VLink 对应的核间通讯接口组织 broadcast、put、all-gather 或 all-reduce。
6. 用 `T.gemm`、`T.Tiles`、`T.reduce*` 等接口表达片上计算。
7. 将本 core 负责的结果写回 MeshTensor shard。

### 2.2 Target 设置

设置 target 的目的是告诉 TileLang 当前 kernel 要按照 SunMMIO 的硬件结构和目标规则处理。它会影响 device mesh 配置、memory scope 解释、layout 选择、数据搬运、核间通讯和 GEMM 等目标相关语义。

推荐使用：

```python
kernel = tilelang.compile(func, target="sunmmio")
```

或：

```python
@tilelang.jit(target="sunmmio")
def kernel_factory(*args, **kwargs):
    ...
```

当前 `auto` 不会自动探测 SunMMIO，因此建议始终显式指定 target。

### 2.3 Kernel 启动模型

SunMMIO kernel 在目标的符号化 mesh 上启动。`T.Kernel()` 默认使用 `T.mesh_ncores()` 作为 extent，循环变量 `cid` 是当前 core 的线性 id。显式整数 extent 可能与编译 target 绑定的 mesh 不一致，因此当前实现会拒绝这种写法。

```python
with T.Kernel() as cid:
    row = cid // T.mesh_ncols()
    col = cid % T.mesh_ncols()
    ...
```

显式等价写法是 `T.Kernel(T.mesh_ncores())`。SunMMIO 不使用 thread extent，因此不要设置 `threads`。

用户通常需要用 `cid`、`row`、`col` 决定当前 core 负责哪一块数据，以及核间通讯时当前 core 处于哪一行或哪一列。

从执行语义上看，`T.Kernel()` 启动的是一组同时参与同一 kernel 的 persistent core 实例。每个 core 在一次 kernel 启动内持续执行，通常通过循环处理本 core 被分配到的多个 tile 或 work item。即使某些 core 在某个阶段没有实际计算任务，也应按照同一份程序走到 kernel 结束位置；整个 kernel 的完成以所有参与 core 都执行结束为准。

### 2.4 MeshTensor 与 Placement

`T.MeshTensor` 是 SunMMIO 多 core 输入/输出的主要抽象。它写在函数参数位置，用来描述一个逻辑完整 tensor 如何分布到 2D mesh 上；进入 kernel 后，每个 core 看到的是 sharding 之后分配到该 core 的本地 shard。

```python
A: T.MeshTensor(
    (M, K),
    T.placement.full_shard(0, 1),
    dtype,
)
```

`MeshTensor` 需要用户明确给出逻辑全局 shape、placement 和 dtype；device mesh config 可以显式传入，也可以省略并使用目标的符号化 mesh。`layout` 参数通常也可以省略：普通 dtype 的 rank-1 tensor 默认使用 1024-byte 对齐的 row-major，rank >= 2 tensor 默认在最后两维使用 32x32 分块的 ZZ；MX dtype 的 rank >= 2 tensor 默认使用 MXZZ，rank 1 不受支持。这里的 `global` 是 TileLang 对 DRAM 侧 tensor 的 scope 命名，对应当前 core 的 DRAM 侧 shard。

Placement API 与 torch-sunmmio 使用相同的术语和调用形式。`T.placement` 提供五个构造函数，每个函数都返回不可变的 `PlacementSpec`：

```python
full = T.placement.full_shard(row_dim=0, col_dim=1)
by_row = T.placement.row_shard(dim=0)
by_col = T.placement.col_shard(dim=1)
replicated = T.placement.replicated()
across_mesh = T.placement.mesh_as_line(dim=0)
```

`full_shard(0, 1)` 表示 mesh row 轴按 `nrows` 切分 tensor 第 0 维，mesh col 轴按 `ncols` 切分第 1 维。row 和 col 具有物理含义，因此 `full_shard(0, 1)` 与 `full_shard(1, 0)` 不等价。对于矩阵类 kernel，通常需要让 placement 和算法的数据流一致：例如按 M 维分布输出行、按 N 维分布输出列，或者在 K 维上配合 `all_gather` 收集计算所需分块。

`mesh_as_line(0)` 将 2D mesh 按 row-major 顺序视为一条线，并把 tensor 第 0 维按 `nrows * ncols` 切分；core 的 shard index 为 `row * ncols + col`。非整除 shape 的本地 slot 使用向上取整，靠后的 core 可能具有较短的有效 extent；kernel 不应假设每个 core 的有效数据完全等长。

为了兼容已有程序，旧的 `T.MeshShardingPolicy` 和 `T.MeshReplicationType` 仍然可用，但新代码应优先使用 `T.placement`：

| 旧 API | 等价的新 API |
|---|---|
| `T.MeshShardingPolicy(y=a, x=b)` | `T.placement.full_shard(a, b)` |
| `T.MeshShardingPolicy(y=a, replicate=T.MeshReplicationType.ROW)` | `T.placement.row_shard(a)` |
| `T.MeshShardingPolicy(x=b, replicate=T.MeshReplicationType.COLUMN)` | `T.placement.col_shard(b)` |
| `T.MeshShardingPolicy(replicate=T.MeshReplicationType.ALL)` | `T.placement.replicated()` |
| `T.MeshShardingPolicy(cross_mesh_dim=d)` | `T.placement.mesh_as_line(d)` |

`full_shard(a, a)` 仍然保留 row 和 col 两个 mesh 轴的物理含义：先按 row 切分，再在每个 row shard 内按 col 切分，因此与旧 `MeshShardingPolicy(y=a, x=a)` 的行为一致。它不同于 `mesh_as_line(a)`；后者才会按 row-major 线性 core id 将 mesh 视为一条线。

旧的 `sharding_policy=` 关键字也继续作为 `placement=` 的别名支持；同一次调用不能同时指定两者。

### 2.5 Layout

Layout 描述 tensor 或 buffer 在内存中的组织方式。它和 sharding 是两个不同层面的概念：sharding 决定完整 tensor 分到哪个 core，layout 决定每个 shard 内部如何排布。

高级场景可用的 layout 构造包括：

```python
from tilelang.layout import (
    make_row_major,
    make_zz_layout,
    make_zn_layout,
    make_zzz_layout,
    make_nzz_layout,
)
```

如果 `MeshTensor` 不显式传 `layout`，当前实现会按 rank 和 dtype 分别为 global shape 与 shard shape 构造默认 layout：普通 dtype 的 rank 1 使用 1024-byte 对齐的 row-major，rank >= 2 使用最后两维为 32x32 分块的 ZZ；MX dtype 的 rank >= 2 使用 MXZZ，rank 1 不受支持。GEMM、块级搬运和核间 gather 等场景下，用户通常不需要手动声明 layout。

显式 layout 主要用于高级场景，例如需要与外部固定数据格式对接、复现已有 layout、或调试 layout 相关问题。此时可以手动构造 layout：

```python
A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
C_layout = make_zz_layout((M, N), [0, 1], (32, 32))
```

### 2.6 TileView

TileView 描述一个 buffer 如何被划分为 tile。它不描述内存物理布局，重点描述 tile loop 和 tile 内计算如何理解 buffer 的逻辑分块。

TileView 的核心信息包括：

- `buffer_shape`：原始 buffer shape。
- `tile_shape`：每个 tile 的 shape。
- `index_map`：哪些维度参与 tiling，支持负索引。

例如：

```python
from tilelang.tileview import make_tileview

T.annotate_tileview({
    A_shared: make_tileview(A_shared, [32, 32], [-2, -1]),
})
```

也可以使用简写：

```python
T.annotate_tileview({
    A_shared: ([32, 32], [-2, -1]),
})
```

对于 shape 为 `(64, 128)` 的 2D buffer，`tile_shape=(16, 32)`、`index_map=(-2, -1)` 表示按最后两个维度切 tile，逻辑 tiled shape 为 `(4, 4, 16, 32)`。

### 2.7 Copy 语义

用户通常使用 `T.copy` 表达数据搬运：

```python
T.copy(A[local_m, k * block_K], A_shared)
T.copy(C_shared, C[local_m, local_n])
```

下表中的 `DRAM` 指当前 core 的 DRAM 侧 shard。

| 路径                              | 用户语义                         |
| ------------------------------- | ---------------------------- |
| DRAM -> RSRAM                   | 从本 core DRAM shard 读取到 RSRAM |
| RSRAM -> DRAM                   | 从 RSRAM 写回本 core DRAM shard  |
| DRAM -> ASRAM/WSRAM             | 把输入搬到 Tensor Core 对应操作数区域；不支持的直接路径可由 legalization 经 RSRAM 中转 |
| RSRAM -> ASRAM/WSRAM            | 从 RSRAM 准备 Tensor Core 操作数   |
| RSRAM -> RSRAM                  | 片上 RSRAM 内部拷贝                |
| ASRAM/WSRAM -> DRAM             | 不支持                          |
| ASRAM <-> WSRAM                 | 不支持                          |
| ASRAM -> ASRAM / WSRAM -> WSRAM | 不支持                          |

`T.copy` 可以接受完整 buffer，也可以接受切片或 region。用户应尽量让源、目标 shape 对齐，并让目标 scope 符合后续计算单元的需求：左操作数进入 ASRAM，右操作数进入 WSRAM，结果操作数和多数中间结果放 RSRAM。

DMA 和 link 搬运要求源、目标 dtype 一致。改变 dtype 的 copy 只在 RSRAM 到 RSRAM 的 Tile 路径上受支持；先在 RSRAM 中完成暂存和转换，再搬到其他 SRAM scope。

### 2.8 T.Tiles

`T.Tiles` 用来表达 tile-level 循环，适合对片上 buffer 做 tile 内逐元素算术、fill/clear、局部 reduce 等操作。

推荐写法：

```python
for i, j in T.Tiles([block_M, block_N], parallel=True):
    C_shared[i, j] = C_shared[i, j] + Bias_shared[i, j]
```

兼容写法：

```python
for i, j in T.Tiles(C_shared, parallel=True):
    C_shared[i, j] = C_shared[i, j] + Bias_shared[i, j]
```

`parallel=True` 表示不同 tile 迭代之间没有 loop-carried dependency。若循环内存在累加、归约或跨迭代依赖，应使用更明确的 reduce 写法，或避免把该循环标成并行。复杂访问模式建议配合 `T.annotate_tileview` 使用。

使用约束：

- 不支持嵌套 `T.Tiles`。
- scope 内需要有可分析的 buffer access。
- access pattern 需要能绑定到可行的 1D 或 2D TileView。
- 隐式 reduction 不支持，reduction 应使用显式 reduce 或 `T.comm.all_reduce`。

### 2.9 核间通讯语义

核间通讯接口用于在 2D core mesh 上交换片上 buffer 数据。核间通讯方向支持：

```text
horizontal / h
vertical   / v
all        / a
```

core 可以用线性 id，也可以用 `(row, col)` 坐标表示。一般来说，沿行方向交换数据使用 `horizontal`，沿列方向交换数据使用 `vertical`，跨全 mesh 的 collective 使用 `all`。

常用接口：

```python
T.comm.broadcast(A_shared, B_shared, src_core=(0, 0), direction="horizontal")
T.comm.put(A_shared, B_shared, src_core=(1, 2), dst_core=(2, 3))
T.comm.all_gather(A_local, A_gathered, direction="horizontal", axis=-1)
T.comm.all_reduce(A_shared, Out_shared, "sum", direction="all")
```

选择接口时可以按语义判断：

- `broadcast`：一个源 core 向某个方向上的多个 core 发送相同数据。
- `put`：一个源 core 向一个目标 core 发送数据。
- `all_gather`：多个 core 各自贡献一段数据，并在接收 buffer 中拼接。
- `all_reduce`：对多个 core 上相同位置的元素做 reduce。它保留完整的本地 shape，不会归约本地 tensor 的某个维度。

`all_gather` 的 recv shape 需要与 direction 和 axis 匹配：

- `axis=None`：recv shape 为 `[K, *send_shape]`。
- `axis=0`：沿第 0 维拼接。
- `axis=-1`：沿最后一维拼接。

当前 `axis` 只支持 `0` 或 `-1`。其中 K 是参与 gather 的 core 数：horizontal 为 `ncols`，vertical 为 `nrows`，all 为 `nrows * ncols`。

### 2.10 GEMM

SunMMIO 下 `T.gemm` 表达 Tensor Core GEMM / MMA 计算。左操作数、右操作数和结果操作数在硬件上分别对应 ASRAM、WSRAM、RSRAM 路径；编译器可以根据 `T.gemm(A, B, C)` 中的参数角色推断合适的 SRAM scope。

下面的写法用于说明三类数据在硬件上的对应关系；实际 kernel 中可以先使用普通 shared buffer，让编译器完成 scope 推断：

```python
A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.asram")
B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.wsram")
C_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")

T.gemm(A_shared, B_shared, C_shared)
```

scope 对应关系：

- 左操作数放 `shared.asram`。
- 右操作数放 `shared.wsram`。
- 结果操作数放 `shared.rsram`。

用户真正需要关注的是调用 `T.gemm` 前左右操作数数据是否已经准备好，结果操作数是否已按算法需要初始化或保留累加值。

在 MeshTensor GEMM / SUMMA 类 kernel 中，常见模式是：

1. 每个 core 从自己的 MeshTensor shard 读取局部数据。
2. 使用 `T.comm.all_gather` 在行/列方向收集完整计算所需分块。
3. 调用 `T.gemm`。
4. 将 C 的本地结果写回 MeshTensor shard。

### 2.11 Pipeline

用户一般通过 `T.Pipelined` 表达循环流水线意图：

```python
for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=3):
    T.copy(...)
    T.comm.all_gather(...)
    T.gemm(...)
```

`T.Pipelined` 适合用于 K-loop 或连续 tile 处理，表示循环体中的搬运、核间通讯和计算存在流水化机会。`num_stages` 表示希望保留多少阶段的流水线空间。用户只需要保证循环体内的数据依赖语义正确，底层控制由编译器处理。

## 3. 前端函数说明

本节按类别组织用户会直接接触的前端接口。每个类别下逐个说明函数签名、参数含义和 SunMMIO 使用注意点。示例中的 `buffer` 可以是完整 buffer，也可以是切片或 region。

### 3.1 编译与设备信息

**`tilelang.compile`**

```python
tilelang.compile(func, target="sunmmio")
```

`tilelang.compile` 用于把 TileLang kernel 编译到指定目标。

- `func`：要编译的 `PrimFunc`。如果入口是带参数的 kernel factory，应使用 `tilelang.jit`。
- `target`：目标后端。SunMMIO 用户应显式写 `target="sunmmio"`，让编译流程按 SunMMIO 的 mesh、scope、layout、copy、核间通讯和 GEMM 语义处理。

**`tilelang.jit`**

```python
@tilelang.jit(target="sunmmio")
def kernel_factory(*args, **kwargs):
    ...
```

`tilelang.jit` 用于定义可 JIT 编译的 kernel factory。

- `target`：目标后端。SunMMIO kernel 建议显式写 `target="sunmmio"`。
- 被装饰的函数：通常返回一个 `@T.prim_func` 或内部 `main` 函数。

**`driver.get_sunmmio_device_mesh_config`**

```python
from tilelang.carver.arch import driver

nrows, ncols = driver.get_sunmmio_device_mesh_config()
```

`get_sunmmio_device_mesh_config()` 当前返回具体的默认 device mesh 形状 `(4, 4)`。

- 参数：无。
- 返回值：`(nrows, ncols)`。
- `nrows`：mesh 的行数，对应纵向 core 数。
- `ncols`：mesh 的列数，对应横向 core 数。
- 常见用途：host 侧检查和诊断。kernel 类型、启动 extent 和 buffer shape 通常应使用 `T.mesh_nrows()`、`T.mesh_ncols()`、`T.mesh_ncores()`，由编译 target 负责解析。

**`driver.get_sunmmio_device_properties`**

```python
props = driver.get_sunmmio_device_properties()
```

`get_sunmmio_device_properties()` 返回仓库当前静态定义的 A4E device 描述，尚不会查询 runtime device。

- `device_id`：可选 device 索引，当前未使用，默认值为 `0`。
- 返回值：`SunmmioDeviceProperties`，包含 `mesh_config`、`RsramPerCore`、`WSRAMperCore` 和 `ASRAMPerCore` 字段。
- 常见用途：host 侧检查；不要把这些值当作 runtime capability query。

### 3.2 MeshTensor、Placement 与 Layout

**`T.placement` / `PlacementSpec`**

```python
full = T.placement.full_shard(row_dim, col_dim)
by_row = T.placement.row_shard(dim)
by_col = T.placement.col_shard(dim)
replicated = T.placement.replicated()
across_mesh = T.placement.mesh_as_line(dim)
```

这些构造函数返回 `PlacementSpec`，可以直接传给 `T.MeshTensor`。`T.MeshTensor` 不接受裸列表形式的 placement。

- `full_shard(row_dim, col_dim)`：row 和 col 两个 mesh 轴分别切分指定的 tensor 维度。
- `row_shard(dim)`：仅 row 轴切分，col 轴复制。
- `col_shard(dim)`：row 轴复制，仅 col 轴切分。
- `replicated()`：两个 mesh 轴都复制。
- `mesh_as_line(dim)`：把 2D mesh 按 row-major 顺序线性化，并沿整个 mesh 切分指定维度。

`row_dim`、`col_dim` 和 `dim` 都是从 0 开始的 tensor 维度索引。传入 `T.MeshTensor` 后会根据 tensor rank 进行范围检查。

**`T.MeshTensor`**

```python
# 推荐的符号化 mesh 写法
T.MeshTensor(shape, placement, dtype="float32", layout=None)

# 具体 mesh 特化写法
T.MeshTensor(shape, placement, (nrows, ncols), dtype="float32", layout=None)
```

`MeshTensor` 用在 kernel 函数参数处，声明一个分布在 SunMMIO mesh 上的输入或输出 tensor。

- `shape`：逻辑完整 shape，例如 `(M, K)`。这是用户视角下的全局 tensor shape。
- `placement`：由 `T.placement` 构造的 `PlacementSpec`，描述全局 tensor 如何切分或复制到各个 core。
- `device_mesh_config`：形如 `(nrows, ncols)`。普通 kernel 应省略它并使用符号化目标 mesh；只有需要把 tensor 类型特化到某一 mesh 时才传入具体 tuple。
- `dtype`：元素类型，例如 `"float16"`、`"bfloat16"`、`"float32"`。
- `layout`：DRAM 中的全局数据布局。省略时，普通 dtype 的 rank 1 默认使用 1024-byte 对齐的 row-major，rank >= 2 默认使用 ZZ；MX dtype 的 rank >= 2 默认使用 MXZZ，rank 1 不受支持。用户通常不需要手动传入该参数。

进入 kernel 后，`MeshTensor` 参数对应当前 core 可见的本地 shard。使用 `A.global_shape` 查询逻辑完整 shape，使用 `A.local_shape` 查询统一分配的本地 slot shape，使用 `A.get_local_extent()` 查询当前 core 的有效 extent。

**`make_row_major`**

```python
from tilelang.layout import make_row_major

layout = make_row_major(shape)
```

`make_row_major` 构造 row-major layout。

- `shape`：tensor 的逻辑 shape。
- 返回值：可传给 `T.MeshTensor(..., layout=layout)` 或 `T.annotate_layout` 的 layout 对象。
- 常见用途：高级场景下需要显式表达 row-major，或调试 layout 相关问题。普通 kernel 通常依赖按 rank 和 dtype 选择的默认 layout。

**`make_zz_layout`**

```python
from tilelang.layout import make_zz_layout

layout = make_zz_layout(shape_or_buffer, axes=None, block_shape=(32, 32))
```

`make_zz_layout` 构造块级 ZZ layout。普通 GEMM 或块级访存通常由编译器推断或派生所需布局；该函数主要用于外部固定格式对接、复现已有 layout 或调试。

- `shape_or_buffer`：可以是 shape，也可以是 buffer、buffer load 或 buffer region。
- `axes`：参与分块布局的维度。省略时默认使用最后两个维度；因此 rank 至少为 2。
- `block_shape`：每个布局 block 的形状，常见值为 `(32, 32)`。
- 返回值：layout 对象。

显式构造示例：

```python
A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
C_layout = make_zz_layout((M, N), [0, 1], (32, 32))
```

**`make_zn_layout`**

```python
from tilelang.layout import make_zn_layout

layout = make_zn_layout(shape, axes, block_shape)
```

`make_zn_layout` 构造 ZN layout。

- `shape`：tensor 的逻辑 shape。
- `axes`：参与布局变换的维度。
- `block_shape`：分块形状。
- 返回值：layout 对象。
- 常见用途：需要特定矩阵操作数格式时显式指定。

**`make_zzz_layout`**

```python
from tilelang.layout import make_zzz_layout

layout = make_zzz_layout(shape, axes, block_shape, cluster_shape)
```

`make_zzz_layout` 构造带 cluster 组织的 ZZZ layout。

- `shape`：tensor 的逻辑 shape。
- `axes`：参与布局变换的维度。
- `block_shape`：每个 block 的形状。
- `cluster_shape`：cluster 级组织形状。
- 返回值：layout 对象。

**`make_nzz_layout`**

```python
from tilelang.layout import make_nzz_layout

layout = make_nzz_layout(shape, axes, block_shape, cluster_shape)
```

`make_nzz_layout` 构造带 cluster 组织的 NZZ layout。

- `shape`：tensor 的逻辑 shape。
- `axes`：参与布局变换的维度。
- `block_shape`：每个 block 的形状。
- `cluster_shape`：cluster 级组织形状。
- 返回值：layout 对象。

**`T.annotate_layout`**

```python
T.annotate_layout({buffer: layout})
```

`T.annotate_layout` 给 buffer 显式附加 layout，属于高级用法。

- 参数：一个 dict，key 是 buffer，value 是 layout 对象。
- `buffer`：需要标注布局的对象。
- `layout`：由 `make_row_major`、`make_zz_layout` 等函数构造。
- 常见用途：外部固定格式对接、复现已有 layout，或调试 layout 相关问题。普通 kernel 通常依赖编译器推断。

**`make_tileview`**

```python
from tilelang.tileview import make_tileview

tileview = make_tileview(buffer, tile_shape, index_map)
```

`make_tileview` 描述 buffer 如何被切成 tile，常和 `T.Tiles` 配合使用。

- `buffer`：要切分的 buffer、buffer load 或 buffer region。
- `tile_shape`：每个 tile 的形状，例如 `[16, 32]`。
- `index_map`：参与 tiling 的 buffer 维度，例如 `[-2, -1]` 表示最后两个维度。
- 返回值：TileView 对象。

SunMMIO 上常见 2D tile 的最后一维为 32，高度常取 8、16 或 32。复杂访问模式建议显式使用 TileView。

**`T.annotate_tileview`**

```python
T.annotate_tileview({buffer: make_tileview(buffer, tile_shape, index_map)})
T.annotate_tileview({buffer: (tile_shape, index_map)})
```

`T.annotate_tileview` 给 buffer 标注 TileView。

- 参数：一个 dict，key 是 buffer。
- value 可以是 `make_tileview(...)` 的返回值。
- value 也可以写成 `(tile_shape, index_map)` 的 tuple shorthand。
- `tile_shape`：tile 形状。
- `index_map`：参与 tiling 的维度。

当 `T.Tiles` 中的访问模式较复杂，或需要稳定表达 tile domain 时，建议使用该接口。

### 3.3 片上存储

**`T.alloc_shared`**

```python
T.alloc_shared(shape, dtype, scope="shared.dyn")
```

`T.alloc_shared` 申请 core 内片上 SRAM buffer。

- `shape`：buffer shape，例如 `(block_M, block_N)`。
- `dtype`：元素类型，例如 `"float16"`、`"bfloat16"`、`"float32"`。
- `scope`：片上 SRAM scope。默认 `"shared.dyn"`；SunMMIO 常用 `"shared.rsram"`、`"shared.asram"`、`"shared.wsram"`。
- 返回值：可在 kernel 中读写的 buffer。

常用 scope：

- `shared.asram`：Tensor Core 左操作数路径。
- `shared.wsram`：Tensor Core 右操作数路径。
- `shared.rsram`：结果操作数、中间结果、Vector Core 处理数据。

普通 GEMM 代码可以先使用默认 shared buffer；需要明确绑定硬件路径时再显式写 scope。

**`T.clear`**

```python
T.clear(buffer)
```

`T.clear` 把 buffer 或 region 清零。

- `buffer`：完整 buffer、slice 或 region。
- 返回值：表达清零操作的语句。
- 常见用途：GEMM 前初始化结果操作数，或初始化临时 buffer。

**`T.fill`**

```python
T.fill(buffer, value)
```

`T.fill` 用指定值填充 buffer 或 region。

- `buffer`：完整 buffer、slice 或 region。
- `value`：填充值，类型应能转换到 buffer dtype。
- 返回值：表达填充操作的语句。
- 常见用途：初始化常量、设置 mask 默认值、准备 reduce 初值。

### 3.4 数据搬运

**`T.copy`**

```text
T.copy(
    src,
    dst,
    *,
    coalesced_width=None,
    disable_tma=False,
    eviction_policy=None,
    annotations=None,
    loop_layout=None,
)
```

`T.copy` 是推荐的数据搬运入口。

- `src`：源 buffer、slice、region 或标量位置。
- `dst`：目标 buffer、slice、region 或标量位置。
- `coalesced_width`：合并访问宽度提示，默认 `None`。
- `disable_tma`：复制路径选择开关，SunMMIO 用户通常保持默认。
- `eviction_policy`：缓存/替换策略提示，可取 `"evict_normal"`、`"evict_first"`、`"evict_last"` 或 `None`。
- `annotations`：附加标注 dict。若和单独参数同时出现，dict 中的值优先。
- `loop_layout`：复制循环的布局提示，普通 SunMMIO 代码通常不需要填写。
- 返回值：表达 copy 操作的语句。

SunMMIO 常见路径包括 DRAM -> RSRAM、RSRAM -> DRAM、DRAM/RSRAM -> ASRAM、DRAM/RSRAM -> WSRAM、RSRAM -> RSRAM。不支持路径见 2.7。

**`T.transpose`**

```python
T.transpose(src, dst)
```

`T.transpose` 通过 A4E ODMA 在 RSRAM buffer 之间完成矩阵转置。

- `src`：shape 为 `[M, N]` 的完整 rank-2 RSRAM buffer。
- `dst`：shape 为 `[N, M]` 的完整 rank-2 RSRAM buffer，不能与 `src` 指向同一 buffer。
- 源和目标的 dtype 必须一致，当前支持 `bfloat16` 和 `float32`。
- shape 必须是静态值，且每一维都是 32 的倍数。两个 buffer 必须使用同一种两级 32x32 blockwise layout，即 ZZ 或 ZN。
- 返回值：表达转置操作的语句。

ODMA 转置是异步操作，编译器会在后续依赖访问之前插入同步。当前不支持 slice、局部 region 或 RSRAM 之外的 buffer。

### 3.5 Tile Loop

**`T.Tiles`**

```python
for i, j in T.Tiles(domain, parallel=False):
    ...
```

`T.Tiles` 构造 tile-level 循环。

- `domain`：循环逻辑域。推荐写显式 shape，例如 `[block_M, block_N]`；也可以传 buffer，表示使用 `buffer.shape`。
- `parallel`：是否声明不同 tile 迭代之间无跨迭代依赖，默认 `False`。
- 返回值：tile loop 迭代变量，变量个数由 `domain` rank 决定。

当循环体中存在累加、归约或跨迭代依赖时，不要把 `parallel` 设为 `True`。

### 3.6 核间通讯

核间通讯函数用于在 core 之间传递数据。源数据可以来自片上 buffer，也可以直接来自 DRAM 侧 `MeshTensor` 的 slice/region；后者常用于把本 core DRAM shard 中的某个分块直接作为 broadcast 或 all-gather 的输入。目标通常是接收端片上 buffer，因为后续计算一般会在片上 SRAM 中消费核间通讯结果。

**`T.comm.broadcast`**

```python
T.comm.broadcast(src, dst, src_core, direction="all", size=-1)
```

`broadcast` 表达一个源 core 向指定方向上的多个 core 广播数据。

- `src`：源 buffer 或 region，可以是片上 buffer，也可以是 DRAM 侧 `MeshTensor` slice/region。
- `dst`：接收 buffer 或 region，通常是片上 buffer。
- `src_core`：源 core，可以是线性 id，也可以是 `(row, col)`。
- `direction`：广播方向，支持 `"horizontal"` / `"h"`、`"vertical"` / `"v"`、`"all"` / `"a"`。
- `size`：广播元素数，`-1` 表示使用整个源 region。
- 返回值：表达 broadcast 的语句。

`horizontal` 沿同一 row 传播，`vertical` 沿同一 column 传播，`all` 覆盖整个 mesh。

**`T.comm.put`**

```python
T.comm.put(src, dst, src_core, dst_core, size=-1)
```

`put` 表达点对点发送。

- `src`：源 buffer 或 region，可以是片上 buffer，也可以是 DRAM 侧 `MeshTensor` slice/region。
- `dst`：目标 buffer 或 region，通常是目标 core 上的片上 buffer。
- `src_core`：源 core，可以是线性 id，也可以是 `(row, col)`。
- `dst_core`：目标 core，可以是线性 id，也可以是 `(row, col)`。
- `size`：发送元素数，`-1` 表示使用整个源 region。
- 返回值：表达 put 的语句。

`put` 适合一对一数据交付。若同一份数据要发给一行、一列或整个 mesh，通常使用 `broadcast`。

**`T.comm.all_gather`**

```python
T.comm.all_gather(
    send_buffer,
    recv_buffer,
    direction="all",
    size=-1,
    axis=None,
    src_offset_byte=0,
)
```

`all_gather` 表达多个 core 各自贡献局部片段，并在接收 buffer 中拼接。

- `send_buffer`：当前 core 贡献的源 buffer 或 region，可以是片上 buffer，也可以是 DRAM 侧 `MeshTensor` slice/region。
- `recv_buffer`：接收拼接结果的 buffer 或 region，通常是片上 buffer。
- `direction`：参与 gather 的方向，支持 `"horizontal"` / `"h"`、`"vertical"` / `"v"`、`"all"` / `"a"`。
- `size`：每个 core 发送的元素数，`-1` 表示使用整个 `send_buffer`。
- `axis`：拼接维度。`None` 表示新增第 0 维；`0` 表示沿第 0 维拼接；`-1` 表示沿最后一维拼接。
- `src_offset_byte`：bf16 GEMM legalization 使用的编译器内部源地址字节偏移；用户代码必须保持为 `0`。
- 返回值：表达 all-gather 的语句。

接收 shape 需要和 `direction`、`axis` 匹配。若参与 core 数为 `K`：

- `axis=None`：`recv_buffer.shape == [K, *send_buffer.shape]`。
- `axis=0`：第 0 维扩大为 `K * send_buffer.shape[0]`。
- `axis=-1`：最后一维扩大为 `K * send_buffer.shape[-1]`。

**`T.comm.all_reduce`**

```python
T.comm.all_reduce(buffer, out, reduce_type, direction, clear=True)
```

`all_reduce` 只表达跨 core reduce：对 `direction` 选中的各个 core 上相同位置的元素做归约，不会沿本地 buffer 的某个维度做归约。

- `buffer`：参与 reduce 的输入 buffer 或 region。
- `out`：保存 reduce 结果的输出 buffer 或 region。
- `reduce_type`：reduce 类型，支持 `"sum"`、`"abssum"`、`"max"`、`"min"`、`"absmax"`、`"bitand"`、`"bitor"`、`"bitxor"`。
- `direction`：参与 reduce 的方向，支持 `"horizontal"` / `"h"`、`"vertical"` / `"v"`、`"all"` / `"a"`。
- `clear`：仅限关键字传入，用于控制是否以 reducer 的中性值初始化 `out`；为 `False` 时，跨 core 归约结果会与 `out` 的原值继续归约。默认值为 `True`。
- 返回值：表达 all-reduce 的语句。

输入和输出 region 必须具有相同的 rank 和各维 extent。若还需要归约本地 buffer 内的某个维度，应使用 `T.reduce`。

### 3.7 GEMM

**`T.gemm`**

```python
T.gemm(
    A,
    B,
    C,
    transpose_A=False,
    transpose_B=False,
    policy=T.GemmWarpPolicy.Square,
    clear_accum=False,
    k_pack=1,
    wg_wait=0,
    mbar=None,
)
```

`T.gemm` 表达 Tensor Core GEMM / MMA。

- `A`：左操作数 buffer 或 region，硬件路径对应 ASRAM。
- `B`：右操作数 buffer 或 region，硬件路径对应 WSRAM。
- `C`：结果操作数 buffer 或 region，硬件路径对应 RSRAM，承担累加和输出角色。
- `transpose_A`：是否按转置方式解释左操作数，默认 `False`。
- `transpose_B`：是否按转置方式解释右操作数，默认 `False`。
- `policy`：GEMM warp 分配策略，SunMMIO 常用默认值。
- `clear_accum`：是否由 GEMM 操作清空结果操作数，默认 `False`。常见写法是在 GEMM 前显式 `T.clear(C)`。
- `k_pack`：packed K 参数，SunMMIO 普通用户保持默认 `1`。
- `wg_wait`：等待批次参数，SunMMIO 普通用户保持默认 `0`。
- `mbar`：barrier 参数，SunMMIO 普通用户保持默认 `None`。
- 返回值：表达 GEMM 的语句。

scope 可以根据 `A/B/C` 参数角色推断；显式 scope 写法主要用于说明或需要稳定控制数据路径的场景。

### 3.8 归约

**`T.reduce`**

```python
T.reduce(buffer, out, reduce_type, dim, clear)
```

`T.reduce` 表达片上 buffer 的局部 reduce。

- `buffer`：输入 buffer 或 region。
- `out`：输出 buffer 或 region。
- `reduce_type`：reduce 类型，例如 `"sum"`、`"max"`、`"min"`、`"abssum"`、`"absmax"`。
- `dim`：reduce 维度。
- `clear`：是否在 reduce 前清空输出。
- 返回值：表达 reduce 的语句。

跨 core reduce 使用 `T.comm.all_reduce`。

**`T.reduce_sum`** **/** **`T.reduce_max`** **/** **`T.reduce_min`**

```python
T.reduce_sum(buffer, out, dim=-1, clear=True)
T.reduce_max(buffer, out, dim=-1, clear=True)
T.reduce_min(buffer, out, dim=-1, clear=True)
```

这些函数是常用局部 reduce 的快捷入口。

- `buffer`：输入 buffer 或 region。
- `out`：输出 buffer 或 region。
- `dim`：reduce 维度，默认最后一维。
- `clear`：是否在 reduce 前清空输出，默认 `True`。
- 返回值：表达 reduce 的语句。

**`T.reduce_abssum`** **/** **`T.reduce_absmax`**

```python
T.reduce_abssum(buffer, out, dim=-1, clear=True)
T.reduce_absmax(buffer, out, dim=-1, clear=True)
```

这些函数用于绝对值相关的局部 reduce。

- `buffer`：输入 buffer 或 region。
- `out`：输出 buffer 或 region。
- `dim`：reduce 维度，默认最后一维。
- `clear`：是否在 reduce 前清空输出，默认 `True`。
- 返回值：表达 reduce 的语句。

**`T.reduce_bitand`** **/** **`T.reduce_bitor`** **/** **`T.reduce_bitxor`**

```python
T.reduce_bitand(buffer, out, dim=-1, clear=True)
T.reduce_bitor(buffer, out, dim=-1, clear=True)
T.reduce_bitxor(buffer, out, dim=-1, clear=True)
```

这些函数用于整数或 bit mask 场景中的局部 bitwise reduce。

- `buffer`：输入 buffer 或 region。
- `out`：输出 buffer 或 region。
- `dim`：reduce 维度，默认最后一维。
- `clear`：是否在 reduce 前清空输出，默认 `True`。
- 返回值：表达 reduce 的语句。

### 3.9 Kernel 与循环

**`T.Kernel`**

```python
with T.Kernel() as cid:
    ...

# 显式等价写法
with T.Kernel(T.mesh_ncores()) as cid:
    ...
```

`T.Kernel` 建立 kernel/core 执行入口。

- `blocks`：省略，或只传 `T.mesh_ncores()`。SunMMIO 不支持显式整数 extent 或多维 grid。
- `threads`：保持未设置。SunMMIO 在 TileLang launch 层是 threadless kernel。
- `is_cpu`：是否生成 CPU 形式的 kernel，SunMMIO 用户保持默认 `False`。
- `prelude`：额外注入代码，SunMMIO 普通用户保持默认 `None`。
- 返回值：线性 core id 绑定 `cid`。

**`T.serial`**

```text
T.serial(start, stop=None, step=None, *, annotations=None)

for i in T.serial(stop):
    ...

for i in T.serial(start, stop, step):
    ...
```

`T.serial` 表达串行循环。

- `start`：起始值。若只传一个参数，该参数会作为 `stop`，`start` 默认为 0。
- `stop`：结束值，循环范围不包含 `stop`。
- `step`：步长，默认 1。
- `annotations`：循环标注 dict，普通用户通常不需要填写。
- 返回值：循环变量。

**`T.Pipelined`**

```python
T.Pipelined(
    start,
    stop=None,
    num_stages=0,
    order=None,
    stage=None,
    sync=None,
    group=None,
)

for k in T.Pipelined(loop_extent, num_stages=3):
    ...
```

`T.Pipelined` 表达可流水化循环，常用于 K loop 或连续 tile 处理。

- `start`：若 `stop` 为空，`start` 表示循环次数；若 `stop` 不为空，`start` 表示起始值。
- `stop`：结束值，默认 `None`。
- `num_stages`：pipeline stage 数。`0` 表示不启用 pipeline；GEMM K loop 常见值为 2 或 3。
- `order`：stage 顺序控制，高级用法通常保持默认。
- `stage`：手动 stage 标注，高级用法通常保持默认。
- `sync`：同步关系描述，高级用法通常保持默认。
- `group`：pipeline group 描述，高级用法通常保持默认。
- 返回值：循环变量。

循环体中通常包含 copy、核间通讯和 compute。用户需要保证数据依赖清晰，例如在使用某个 tile 计算前，相关输入已经搬运或核间通讯完成。

## 4. 示例

本节给出 3 个完整 kernel 示例。为了突出主体结构，示例假设 problem size 能被 block size 和 mesh 切分整除；实际工程代码需要补充边界处理。

### 4.1 本地 Shard GEMM

这个 kernel 中，每个 core 只读取自己 DRAM 中的本地 shard，完成本地 GEMM 后写回自己的输出 shard。`A` 沿 mesh row 切分并在 column 间复制，`B` 在 row 间复制并沿 mesh column 切分，因此每个 core 都拥有计算本地输出 tile 所需的完整 K 维。

```python
import tilelang
import tilelang.language as T
from tilelang.layout import make_zz_layout


@tilelang.jit(target="sunmmio")
def local_shard_gemm(
    M=128,
    N=128,
    K=128,
    block_M=32,
    block_N=32,
    block_K=32,
    dtype="float16",
    accum_dtype="float32",
):
    A_placement = T.placement.row_shard(0)
    B_placement = T.placement.col_shard(1)
    C_placement = T.placement.full_shard(0, 1)
    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), A_placement, dtype, layout=A_layout),
        B: T.MeshTensor((K, N), B_placement, dtype, layout=B_layout),
        C: T.MeshTensor((M, N), C_placement, accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as _cid:
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape

            A_tile = T.alloc_shared((block_M, block_K), dtype)
            B_tile = T.alloc_shared((block_K, block_N), dtype)
            C_tile = T.alloc_shared((block_M, block_N), accum_dtype)

            for bm in T.serial(T.ceildiv(sharded_M, block_M)):
                for bn in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_tile)
                    for bk in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=3):
                        T.copy(
                            A[
                                bm * block_M : (bm + 1) * block_M,
                                bk * block_K : (bk + 1) * block_K,
                            ],
                            A_tile,
                        )
                        T.copy(
                            B[
                                bk * block_K : (bk + 1) * block_K,
                                bn * block_N : (bn + 1) * block_N,
                            ],
                            B_tile,
                        )
                        T.gemm(A_tile, B_tile, C_tile)

                    T.copy(C_tile, C[bm * block_M, bn * block_N])

    return main
```

### 4.2 SUMMA GEMM with HLink / VLink

这个 kernel 展示 SUMMA 风格数据流：同一 row 通过 HLink 收集左操作数分块，同一 column 通过 VLink 收集右操作数分块，然后在本 core 上执行 GEMM。

```python
import tilelang
import tilelang.language as T
from tilelang.layout import make_zz_layout


@tilelang.jit(target="sunmmio")
def summa_gemm(
    M=128,
    N=128,
    K=128,
    block_M=32,
    block_N=32,
    block_K=32,
    dtype="float16",
    accum_dtype="float32",
):
    placement = T.placement.full_shard(0, 1)
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

            A_panel = T.alloc_shared((block_M, block_K * T.mesh_ncols()), dtype)
            B_panel = T.alloc_shared((block_K * T.mesh_nrows(), block_N), dtype)
            C_tile = T.alloc_shared((block_M, block_N), accum_dtype)

            for bm in T.serial(T.ceildiv(sharded_M, block_M)):
                for bn in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_tile)
                    for bk in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=3):
                        T.comm.all_gather(
                            A[
                                bm * block_M : (bm + 1) * block_M,
                                bk * block_K : (bk + 1) * block_K,
                            ],
                            A_panel,
                            direction="horizontal",
                            axis=-1,
                        )
                        T.comm.all_gather(
                            B[
                                bk * block_K : (bk + 1) * block_K,
                                bn * block_N : (bn + 1) * block_N,
                            ],
                            B_panel,
                            direction="vertical",
                            axis=0,
                        )
                        T.gemm(A_panel, B_panel, C_tile)

                    T.copy(C_tile, C[bm * block_M, bn * block_N])

    return main
```

### 4.3 GEMM + Tile 内后处理

这个 kernel 在 GEMM 之后加载一个同 shape 的 bias tile，并使用 `T.Tiles` 对结果 tile 做逐元素加法。

```python
import tilelang
import tilelang.language as T
from tilelang.layout import make_zz_layout


@tilelang.jit(target="sunmmio")
def gemm_with_bias(
    M=128,
    N=128,
    K=128,
    block_M=32,
    block_N=32,
    block_K=32,
    dtype="float16",
    accum_dtype="float32",
):
    A_placement = T.placement.row_shard(0)
    B_placement = T.placement.col_shard(1)
    C_placement = T.placement.full_shard(0, 1)
    A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
    B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
    C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

    @T.prim_func
    def main(
        A: T.MeshTensor((M, K), A_placement, dtype, layout=A_layout),
        B: T.MeshTensor((K, N), B_placement, dtype, layout=B_layout),
        Bias: T.MeshTensor((M, N), C_placement, accum_dtype, layout=C_layout),
        C: T.MeshTensor((M, N), C_placement, accum_dtype, layout=C_layout),
    ):
        with T.Kernel() as _cid:
            sharded_M, sharded_K = A.local_shape
            _, sharded_N = B.local_shape

            A_tile = T.alloc_shared((block_M, block_K), dtype)
            B_tile = T.alloc_shared((block_K, block_N), dtype)
            Bias_tile = T.alloc_shared((block_M, block_N), accum_dtype)
            C_tile = T.alloc_shared((block_M, block_N), accum_dtype)

            for bm in T.serial(T.ceildiv(sharded_M, block_M)):
                for bn in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.clear(C_tile)
                    for bk in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=3):
                        T.copy(
                            A[
                                bm * block_M : (bm + 1) * block_M,
                                bk * block_K : (bk + 1) * block_K,
                            ],
                            A_tile,
                        )
                        T.copy(
                            B[
                                bk * block_K : (bk + 1) * block_K,
                                bn * block_N : (bn + 1) * block_N,
                            ],
                            B_tile,
                        )
                        T.gemm(A_tile, B_tile, C_tile)

                    T.copy(
                        Bias[
                            bm * block_M : (bm + 1) * block_M,
                            bn * block_N : (bn + 1) * block_N,
                        ],
                        Bias_tile,
                    )
                    for i, j in T.Tiles([block_M, block_N], parallel=True):
                        C_tile[i, j] = C_tile[i, j] + Bias_tile[i, j]

                    T.copy(C_tile, C[bm * block_M, bn * block_N])

    return main
```

## 5. 总结

- 显式使用规范的 `sunmmio` target 字符串；`auto` 不会探测该后端。
- kernel 代码优先使用符号化 mesh 表达式，并通过 `T.Kernel()` 启动。
- 新代码使用 `T.placement`，根据算法数据流分别选择各操作数的 placement，不要机械复用同一个 placement。
- 通讯 region、接收 shape、SRAM scope、layout 和 dtype 都应与下一步消费它们的操作保持一致。

## 相关文档

- [SunMMIO TileLang 快速入门](sunmmio_tilelang_getting_started_zh_cn.md)
- [安装文档](../get_started/Installation.md)
- [TileLang 编程说明](../programming_guides/overview.md)
- [SunMMIO 示例](https://github.com/SUNMMIO/Tilelang/tree/tilelang_mesh_main/examples)
