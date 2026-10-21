# SunMMIO TileLang User Guide

[中文](sunmmio_tilelang_user_guide_zh_cn.md) | [Quick start](sunmmio_tilelang_getting_started.md)

This guide is intended for users who already know basic TileLang and need to write, migrate, or debug kernels for the SunMMIO target. It focuses on the hardware structure, programming model, frontend functions, and common coding patterns that users need to understand when writing frontend code.

This document uses `SunMMIO` for the hardware name and the canonical lowercase string `sunmmio` for the TileLang target. The target parser also accepts `Sunmmio` for compatibility.

## 1. Hardware Structure

### 1.1 Architecture Overview

The key hardware characteristics of the SunMMIO NPU can be summarized as: a multi-core 2D mesh, per-core DRAM, hierarchical on-chip SRAM, and multiple types of hardware engines.

SunMMIO uses a core-level execution organization. Each core has its own compute, data movement, and on-chip storage resources. Cores are connected through a 2D mesh. When writing kernels, users usually need to explicitly reason about which data block is owned by the current core, and which neighboring or grouped cores need to exchange data.

The hardware dataflow is explicit: data must enter the corresponding SRAM region before it can be consumed or moved by ODMA, Tensor Core, Vector Core, HLink/VLink, or other hardware units. Therefore, SunMMIO programs are usually organized around data distribution, data movement, local computation, and inter-core communication.

### 1.2 2D Mesh Core

SunMMIO consists of multiple cores organized as a 2D mesh. The current default device configuration is `(4, 4)`, namely 4 rows and 4 columns, with 16 cores in total.

The mesh has explicit row and column directions. The horizontal direction connects cores in the same row, while the vertical direction connects cores in the same column. Data movement across the whole mesh usually combines horizontal and vertical links.

A core can be represented by a 2D coordinate `(row, col)`, or linearized into an id:

```text
core_id = row * ncol + col
```

This coordinate system is the basis for data distribution and inter-core communication.

### 1.3 Per-Core Compute and Data Movement Engines

Each core contains multiple hardware units that can work in parallel:

- Tensor Core: for GEMM / MMA.
- Vector Core: for tile-level arithmetic, comparison, type conversion, fill/clear, local reduction, and other batch-style processing.
- ODMA: responsible for block-level data movement between the current core's DRAM and on-chip SRAM.
- HLink / VLink: responsible for horizontal and vertical inter-core communication.

These engines are independent hardware resources and can work in parallel. When organizing a kernel, users need to understand that computation, data movement, and inter-core communication can form overlapping execution opportunities.

### 1.4 Per-Core DRAM and SRAM Scopes

SunMMIO's memory hierarchy includes DRAM paired with each core and SRAM inside each core. Each core's DRAM mainly stores the input shard, output shard, and data that must be kept outside on-chip SRAM for that core.

Compared with on-chip SRAM, DRAM has larger capacity, but higher access latency, and its bandwidth is more likely to become a bottleneck. On-chip SRAM is closer to compute units and has limited capacity, so it is suitable for data needed by tile/block-level computation.

A typical dataflow is: each core loads data from its own DRAM into an on-chip SRAM region; the data is consumed by Tensor Core, Vector Core, or inter-core communication links; results produced by computation or inter-core communication are then written back from on-chip SRAM to the core's DRAM.

SunMMIO on-chip SRAM is divided into different scopes by purpose:

- `shared.asram`: mainly serves the Tensor Core left-operand data path.
- `shared.wsram`: mainly serves the Tensor Core right-operand data path.
- `shared.rsram`: stores the Tensor Core result operand, Tensor Core output results, and data processed by Vector Core. It also acts as the data exchange hub among DRAM, Tensor Core results, and Vector Core results.

Different SRAM regions correspond to different hardware data paths. Whether a hardware unit can directly consume a piece of data depends on the SRAM region where the data currently resides.

### 1.5 Tensor Core / MMA

Tensor Core is the compute unit inside a core for matrix multiplication and matrix accumulation. It obtains matrix data through fixed on-chip SRAM paths.

In the Tensor Core MMA data path:

- `ASRAM` provides the left operand to Tensor Core.
- `WSRAM` provides the right operand to Tensor Core.
- `RSRAM` usually stores the result operand and Tensor Core output results, and serves as the staging area for later Vector Core processing, inter-core communication, or write-back to DRAM. The result operand carries the accumulation and output role in GEMM / MMA.

The distinction among ASRAM, WSRAM, and RSRAM is both a TileLang scope distinction and a hardware data path distinction for Tensor Core.

### 1.6 ODMA and Block-Level Data Movement

ODMA is the hardware unit inside each core responsible for block-level data movement between that core's DRAM and on-chip SRAM. It serves the current core's DRAM access path, and mainly moves data between the off-chip DRAM shard and on-chip SRAM buffers/regions.

In a typical kernel, ODMA loads input shards from the current core's DRAM into on-chip SRAM, and also writes on-chip computation output regions back to the current core's DRAM. Common directions include DRAM to RSRAM, DRAM to ASRAM/WSRAM, and RSRAM to DRAM. The exact available paths are constrained by SRAM scopes and hardware data paths. When writing copy operations, users should make the destination SRAM region match the next consumer.

ODMA is independent of Tensor Core, Vector Core, and HLink/VLink. Inside one core, execution can overlap data movement, computation, and inter-core communication: while ODMA prepares the next data block, Tensor Core can consume a matrix tile that is already ready, Vector Core can process on-chip intermediate results, and HLink/VLink can transfer on-chip buffers between cores. At the user level, it is important to know whether data is currently in DRAM or in a specific SRAM scope, and which hardware unit will consume it next.

### 1.7 Tile Execution Granularity and Vector Core

SunMMIO local computation and memory access are often organized around tiles. A tile is a data region with an explicit shape, alignment, and access direction. It affects both how data is organized in on-chip SRAM and how Vector Core processes the data.

Vector Core is the compute unit inside a core for tile-level data processing. It is suitable for arithmetic, comparison, type conversion, fill/clear, mask processing, local reduction, and GEMM post-processing within a tile. For example, bias add, scale, clamp, local reduce, and part of the on-chip processing in softmax are all compute patterns that fit Vector Core.

Vector Core usually works on data in RSRAM. After Tensor Core produces a result operand into RSRAM, Vector Core can continue post-processing it before writing it back to DRAM or participating in HLink/VLink inter-core communication. Ordinary input tiles loaded from DRAM into RSRAM can also be processed directly by Vector Core.

Tile shape, alignment, and access direction affect Vector Core execution efficiency. Common efficient code keeps tile shapes aligned with hardware processing widths and keeps on-chip accesses regular. For users, choosing a suitable tile shape, keeping input/output regions aligned, and avoiding complex scattered accesses are important prerequisites for writing stable SunMMIO kernels.

### 1.8 HLink / VLink Inter-Core Communication

SunMMIO inter-core communication is performed through dedicated links:

- HLink: connects cores in the same row and mainly carries horizontal inter-core communication.
- VLink: connects cores in the same column and mainly carries vertical inter-core communication.

HLink / VLink are independent of ODMA, Tensor Core, and Vector Core. They are used to transfer on-chip buffer data between cores. Common inter-core communication patterns include broadcast, point-to-point put, all-gather, and all-reduce.

Horizontal inter-core communication usually propagates along HLink within the same row, for example broadcasting a data block on one core to other cores in the same row. Vertical inter-core communication usually propagates along VLink within the same column, for example broadcasting a data block on one core to other cores in the same column. Inter-core communication that covers the whole mesh combines horizontal and vertical links and forms multi-stage data movement.

Inter-core communication involves not only data transfer itself, but also the source core, destination core, communication direction, receive buffer shape, and cross-core visibility. When writing programs, users should be clear about which core currently holds the data, which direction the data should move along, and whether the receive buffer has reserved enough shape.

## 2. Programming Model

### 2.1 Kernel Execution Model Overview

SunMMIO kernels use an SPMD (Single Program, Multiple Data) programming form: all cores execute the same kernel program, and input/output data is assigned to different cores by the `MeshTensor` placement. Inside the kernel, `cid`, `row`, and `col` are used to identify the shard corresponding to the current core, and to determine on-chip buffer usage and inter-core communication roles. Users usually do not need to write separate programs for different cores. Instead, the same program uses core coordinates to describe data partitioning and cooperation.

SunMMIO kernels are also persistent kernels: after one kernel launch, all cores stay resident and execute the same program, using loops inside the kernel to process the tiles or work items assigned to them. The launched cores continuously complete the work covered by the current kernel until the program reaches the kernel end. The whole kernel exits together only after all cores in that launch finish execution.

Under the SPMD model, `MeshTensor` describes the global shape and distribution of a complete logical tensor. After entering the kernel, the current core accesses the local shard assigned to it after sharding, usually corresponding to the DRAM-side data shard of that core. Cross-core data dependencies are expressed explicitly through HLink / VLink related APIs, while on-chip computation is still written as local computation on the current core.

The simplified structure below shows a typical SunMMIO kernel. Boundary handling and complete loop mapping details are omitted.

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

The SunMMIO programming model can be summarized as:

1. Use `target="sunmmio"` to select the target.
2. Use `MeshTensor`, placement, and layout to describe how logical tensors are distributed onto the 2D mesh and how they are organized in DRAM.
3. Use `T.Kernel()` (equivalent to `T.Kernel(T.mesh_ncores())`) to launch across the symbolic target mesh. Map `cid -> (row, col)` only when the algorithm needs explicit coordinates.
4. Each core loads data from the DRAM shard assigned to itself by sharding into on-chip SRAM.
5. When data is needed across cores, use the HLink / VLink inter-core communication APIs to organize broadcast, put, all-gather, or all-reduce.
6. Use APIs such as `T.gemm`, `T.Tiles`, and `T.reduce*` to express on-chip computation.
7. Write the result owned by the current core back to the MeshTensor shard.

### 2.2 Target Setting

The purpose of setting the target is to tell TileLang that the current kernel should be handled according to SunMMIO hardware structure and target-specific rules. It affects device mesh configuration, memory scope interpretation, layout selection, data movement, inter-core communication, GEMM, and other target-specific semantics.

Recommended usage:

```python
kernel = tilelang.compile(func, target="sunmmio")
```

Or:

```python
@tilelang.jit(target="sunmmio")
def kernel_factory(*args, **kwargs):
    ...
```

Currently, `auto` does not automatically detect SunMMIO, so users should always specify the target explicitly.

### 2.3 Kernel Launch Model

SunMMIO kernels launch across the target's symbolic mesh. `T.Kernel()` defaults its extent to `T.mesh_ncores()`, and the loop variable `cid` is the linear id of the current core. Passing an explicit integer extent is rejected because it can disagree with the mesh bound to the compilation target.

```python
with T.Kernel() as cid:
    row = cid // T.mesh_ncols()
    col = cid % T.mesh_ncols()
    ...
```

The explicit equivalent is `T.Kernel(T.mesh_ncores())`. SunMMIO does not use a thread extent, so leave `threads` unset.

Users usually use `cid`, `row`, and `col` to decide which data block the current core is responsible for, and which row or column the current core belongs to during inter-core communication.

From the execution semantics point of view, `T.Kernel()` launches a group of persistent core instances that participate in the same kernel. Each core keeps executing within one kernel launch, usually processing multiple tiles or work items assigned to that core through loops. Even if some cores have no actual computation task at a certain stage, they should still follow the same program to the kernel end. The whole kernel is considered complete only after all participating cores finish execution.

### 2.4 MeshTensor and Placement

`T.MeshTensor` is the main abstraction for multi-core SunMMIO inputs and outputs. It is written at function parameter positions and describes how a complete logical tensor is distributed onto the 2D mesh. After entering the kernel, each core sees the local shard assigned to that core after sharding.

```python
A: T.MeshTensor(
    (M, K),
    T.placement.full_shard(0, 1),
    dtype,
)
```

`MeshTensor` requires users to provide the global logical shape, placement, and dtype. The device mesh config can be supplied explicitly or omitted to use the target's symbolic mesh. The `layout` parameter can usually be omitted as well: a rank-1 tensor with a regular dtype defaults to a 1024-byte-aligned row-major layout, while a rank >= 2 tensor defaults to a 32x32-blocked ZZ layout on its last two dimensions. A rank >= 2 tensor with an MX dtype defaults to MXZZ; rank-1 MX tensors are unsupported. Here, `global` is TileLang's scope name for DRAM-side tensors, corresponding to the current core's DRAM-side shard.

The placement API uses the same terminology and call style as torch-sunmmio. `T.placement` provides five constructors, each returning an immutable `PlacementSpec`:

```python
full = T.placement.full_shard(row_dim=0, col_dim=1)
by_row = T.placement.row_shard(dim=0)
by_col = T.placement.col_shard(dim=1)
replicated = T.placement.replicated()
across_mesh = T.placement.mesh_as_line(dim=0)
```

`full_shard(0, 1)` means that the mesh row axis splits tensor dimension 0 by `nrows`, while the mesh column axis splits dimension 1 by `ncols`. Row and column have physical meaning, so `full_shard(0, 1)` and `full_shard(1, 0)` are not equivalent. For matrix kernels, placement usually needs to match the algorithm dataflow: for example, distributing output rows along M and output columns along N, or using `all_gather` along K to collect the blocks needed for computation.

`mesh_as_line(0)` treats the 2D mesh as a row-major line and splits tensor dimension 0 by `nrows * ncols`; a core's shard index is `row * ncols + col`. For a non-divisible shape, the local slot is rounded up and later cores can have shorter valid extents; kernels must not assume that every core has the same amount of valid data.

For source compatibility, the legacy `T.MeshShardingPolicy` and `T.MeshReplicationType` APIs remain available, but new code should prefer `T.placement`:

| Legacy API | Equivalent new API |
|---|---|
| `T.MeshShardingPolicy(y=a, x=b)` | `T.placement.full_shard(a, b)` |
| `T.MeshShardingPolicy(y=a, replicate=T.MeshReplicationType.ROW)` | `T.placement.row_shard(a)` |
| `T.MeshShardingPolicy(x=b, replicate=T.MeshReplicationType.COLUMN)` | `T.placement.col_shard(b)` |
| `T.MeshShardingPolicy(replicate=T.MeshReplicationType.ALL)` | `T.placement.replicated()` |
| `T.MeshShardingPolicy(cross_mesh_dim=d)` | `T.placement.mesh_as_line(d)` |

`full_shard(a, a)` still preserves the physical meaning of both mesh axes: it shards by row first, then shards each row-local extent by column, matching legacy `MeshShardingPolicy(y=a, x=a)`. It differs from `mesh_as_line(a)`, which treats the mesh as a row-major line and uses the linear core id.

The legacy `sharding_policy=` keyword is also supported as an alias for `placement=`; a call cannot specify both.

### 2.5 Layout

Layout describes how a tensor or buffer is organized in memory. It is a different concept from sharding: sharding decides which core receives which part of the complete tensor, while layout decides how each shard is arranged internally.

Layout constructors available for advanced scenarios include:

```python
from tilelang.layout import (
    make_row_major,
    make_zz_layout,
    make_zn_layout,
    make_zzz_layout,
    make_nzz_layout,
)
```

If `MeshTensor` does not explicitly pass `layout`, the current implementation constructs defaults for the global shape and shard shape according to rank and dtype. A rank-1 tensor with a regular dtype uses a 1024-byte-aligned row-major layout; a rank >= 2 tensor uses a 32x32-blocked ZZ layout on its last two dimensions. A rank >= 2 tensor with an MX dtype uses MXZZ, while rank-1 MX tensors are unsupported. In scenarios such as GEMM, block-level data movement, and inter-core gather, users usually do not need to declare a layout manually.

Explicit layout is mainly for advanced scenarios, such as interfacing with an externally fixed data format, reproducing an existing layout, or debugging layout-related issues. In those cases, users can manually construct layouts:

```python
A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
C_layout = make_zz_layout((M, N), [0, 1], (32, 32))
```

### 2.6 TileView

TileView describes how a buffer is divided into tiles. It does not describe the physical memory layout. Its focus is how tile loops and tile-level computation understand the logical tiling of a buffer.

The core information in a TileView includes:

- `buffer_shape`: the original buffer shape.
- `tile_shape`: the shape of each tile.
- `index_map`: the dimensions that participate in tiling, with negative indices supported.

For example:

```python
from tilelang.tileview import make_tileview

T.annotate_tileview({
    A_shared: make_tileview(A_shared, [32, 32], [-2, -1]),
})
```

You can also use a shorthand:

```python
T.annotate_tileview({
    A_shared: ([32, 32], [-2, -1]),
})
```

For a 2D buffer with shape `(64, 128)`, `tile_shape=(16, 32)` and `index_map=(-2, -1)` means tiling along the last two dimensions, with logical tiled shape `(4, 4, 16, 32)`.

### 2.7 Copy Semantics

Users usually use `T.copy` to express data movement:

```python
T.copy(A[local_m, k * block_K], A_shared)
T.copy(C_shared, C[local_m, local_n])
```

In the table below, `DRAM` refers to the current core's DRAM-side shard.

| Path                            | User Semantics                                      |
| ------------------------------- | --------------------------------------------------- |
| DRAM -> RSRAM                   | Read from the current core's DRAM shard into RSRAM  |
| RSRAM -> DRAM                   | Write from RSRAM back to the current core's DRAM shard |
| DRAM -> ASRAM/WSRAM             | Move input data to the Tensor Core operand region; unsupported direct paths can be staged through RSRAM by legalization |
| RSRAM -> ASRAM/WSRAM            | Prepare Tensor Core operands from RSRAM             |
| RSRAM -> RSRAM                  | On-chip copy inside RSRAM                           |
| ASRAM/WSRAM -> DRAM             | Unsupported                                         |
| ASRAM <-> WSRAM                 | Unsupported                                         |
| ASRAM -> ASRAM / WSRAM -> WSRAM | Unsupported                                         |

`T.copy` can accept a complete buffer, slice, or region. Users should keep source and destination shapes aligned as much as possible, and should make the destination scope match the needs of the following compute unit: the left operand goes to ASRAM, the right operand goes to WSRAM, and the result operand and most intermediate results go to RSRAM.

DMA and link transfers require matching source and destination dtypes. A dtype-changing copy is supported on the RSRAM-to-RSRAM Tile path; stage and cast there before moving data to another SRAM scope.

### 2.8 T.Tiles

`T.Tiles` expresses tile-level loops, and is suitable for tile-level arithmetic, fill/clear, local reduce, and similar operations on on-chip buffers.

Recommended form:

```python
for i, j in T.Tiles([block_M, block_N], parallel=True):
    C_shared[i, j] = C_shared[i, j] + Bias_shared[i, j]
```

Compatible form:

```python
for i, j in T.Tiles(C_shared, parallel=True):
    C_shared[i, j] = C_shared[i, j] + Bias_shared[i, j]
```

`parallel=True` means there is no loop-carried dependency between different tile iterations. If the loop body contains accumulation, reduction, or dependencies across iterations, users should use an explicit reduce form or avoid marking the loop as parallel. Complex access patterns are recommended to be used together with `T.annotate_tileview`.

Usage constraints:

- Nested `T.Tiles` is not supported.
- There must be analyzable buffer accesses inside the scope.
- The access pattern must be bindable to a feasible 1D or 2D TileView.
- Implicit reduction is not supported. Reductions should use explicit reduce APIs or `T.comm.all_reduce`.

### 2.9 Inter-Core Communication Semantics

Inter-core communication APIs are used to exchange on-chip buffer data over the 2D core mesh. Supported inter-core communication directions are:

```text
horizontal / h
vertical   / v
all        / a
```

A core can be represented by a linear id or by `(row, col)` coordinates. In general, row-wise exchange uses `horizontal`, column-wise exchange uses `vertical`, and collectives across the whole mesh use `all`.

Common APIs:

```python
T.comm.broadcast(A_shared, B_shared, src_core=(0, 0), direction="horizontal")
T.comm.put(A_shared, B_shared, src_core=(1, 2), dst_core=(2, 3))
T.comm.all_gather(A_local, A_gathered, direction="horizontal", axis=-1)
T.comm.all_reduce(A_shared, Out_shared, "sum", direction="all")
```

Choose an API by its semantics:

- `broadcast`: one source core sends the same data to multiple cores along a direction.
- `put`: one source core sends data to one destination core.
- `all_gather`: multiple cores each contribute a data segment, and the receive buffer concatenates the segments.
- `all_reduce`: corresponding elements from multiple cores are reduced. It preserves the complete local shape and does not reduce a local tensor dimension.

The receive shape of `all_gather` must match `direction` and `axis`:

- `axis=None`: the receive shape is `[K, *send_shape]`.
- `axis=0`: concatenate along dimension 0.
- `axis=-1`: concatenate along the last dimension.

Currently, `axis` only supports `0` or `-1`. Here, K is the number of cores participating in the gather: `ncols` for horizontal, `nrows` for vertical, and `nrows * ncols` for all.

### 2.10 GEMM

On SunMMIO, `T.gemm` expresses Tensor Core GEMM / MMA computation. The left operand, right operand, and result operand correspond to the ASRAM, WSRAM, and RSRAM hardware paths respectively. The compiler can infer suitable SRAM scopes from the parameter roles in `T.gemm(A, B, C)`.

The following code illustrates the hardware mapping of the three kinds of data. In actual kernels, users can first use ordinary shared buffers and let the compiler infer scopes:

```python
A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared.asram")
B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared.wsram")
C_shared = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")

T.gemm(A_shared, B_shared, C_shared)
```

Scope mapping:

- The left operand is placed in `shared.asram`.
- The right operand is placed in `shared.wsram`.
- The result operand is placed in `shared.rsram`.

What users really need to care about is whether the left and right operand data has been prepared before calling `T.gemm`, and whether the result operand has been initialized or preserved for accumulation as required by the algorithm.

In MeshTensor GEMM / SUMMA-style kernels, the common pattern is:

1. Each core reads local data from its own MeshTensor shard.
2. Use `T.comm.all_gather` in the row/column direction to collect the complete blocks needed for computation.
3. Call `T.gemm`.
4. Write the local result of C back to the MeshTensor shard.

### 2.11 Pipeline

Users generally use `T.Pipelined` to express loop pipelining intent:

```python
for k in T.Pipelined(T.ceildiv(sharded_K, block_K), num_stages=3):
    T.copy(...)
    T.comm.all_gather(...)
    T.gemm(...)
```

`T.Pipelined` is suitable for K loops or consecutive tile processing. It indicates that data movement, inter-core communication, and computation inside the loop body have pipelining opportunities. `num_stages` indicates how many pipeline stages the user wants to keep. Users only need to ensure that the data dependency semantics in the loop body are correct; low-level control is handled by the compiler.

## 3. Frontend Function Reference

This section groups the frontend APIs that users directly interact with. For each category, it describes function signatures, parameter meanings, and SunMMIO usage notes. In examples, `buffer` can be a complete buffer, a slice, or a region.

### 3.1 Compilation and Device Information

**`tilelang.compile`**

```python
tilelang.compile(func, target="sunmmio")
```

`tilelang.compile` compiles a TileLang kernel to the specified target.

- `func`: the `PrimFunc` to compile. Use `tilelang.jit` when starting from a parameterized kernel factory.
- `target`: the target backend. SunMMIO users should explicitly write `target="sunmmio"` so the compilation flow handles the kernel according to SunMMIO mesh, scope, layout, copy, inter-core communication, and GEMM semantics.

**`tilelang.jit`**

```python
@tilelang.jit(target="sunmmio")
def kernel_factory(*args, **kwargs):
    ...
```

`tilelang.jit` defines a JIT-compilable kernel factory.

- `target`: the target backend. SunMMIO kernels should explicitly use `target="sunmmio"`.
- The decorated function: usually returns a `@T.prim_func` or an inner `main` function.

**`driver.get_sunmmio_device_mesh_config`**

```python
from tilelang.carver.arch import driver

nrows, ncols = driver.get_sunmmio_device_mesh_config()
```

`get_sunmmio_device_mesh_config()` currently returns the concrete default device mesh shape `(4, 4)`.

- Parameters: none.
- Return value: `(nrows, ncols)`.
- `nrows`: number of rows in the mesh, corresponding to the number of cores in the vertical direction.
- `ncols`: number of columns in the mesh, corresponding to the number of cores in the horizontal direction.
- Common usage: host-side inspection and diagnostics. Kernel types, launch extents, and buffer shapes should normally use `T.mesh_nrows()`, `T.mesh_ncols()`, and `T.mesh_ncores()` so they resolve from the compilation target.

**`driver.get_sunmmio_device_properties`**

```python
props = driver.get_sunmmio_device_properties()
```

`get_sunmmio_device_properties()` returns the repository's current static A4E device description; it does not query a runtime device yet.

- `device_id`: optional device index, currently unused; the default is `0`.
- Return value: `SunmmioDeviceProperties`, with `mesh_config`, `RsramPerCore`, `WSRAMperCore`, and `ASRAMPerCore` fields.
- Common usage: host-side inspection. Do not treat these values as a runtime capability query.

### 3.2 MeshTensor, Placement, and Layout

**`T.placement` / `PlacementSpec`**

```python
full = T.placement.full_shard(row_dim, col_dim)
by_row = T.placement.row_shard(dim)
by_col = T.placement.col_shard(dim)
replicated = T.placement.replicated()
across_mesh = T.placement.mesh_as_line(dim)
```

A `PlacementSpec` returned by these constructors can be passed directly to `T.MeshTensor`. `T.MeshTensor` does not accept a raw placement list.

- `full_shard(row_dim, col_dim)`: shard the specified tensor dimensions on the row and column mesh axes.
- `row_shard(dim)`: shard on rows and replicate on columns.
- `col_shard(dim)`: replicate on rows and shard on columns.
- `replicated()`: replicate on both mesh axes.
- `mesh_as_line(dim)`: linearize the 2D mesh in row-major order and shard the specified dimension across the whole mesh.

`row_dim`, `col_dim`, and `dim` are zero-based tensor dimension indices. `T.MeshTensor` validates them against the tensor rank.

**`T.MeshTensor`**

```python
# Recommended symbolic-mesh form
T.MeshTensor(shape, placement, dtype="float32", layout=None)

# Concrete-mesh specialization
T.MeshTensor(shape, placement, (nrows, ncols), dtype="float32", layout=None)
```

`MeshTensor` is used at kernel function parameter positions to declare an input or output tensor distributed on the SunMMIO mesh.

- `shape`: complete logical shape, such as `(M, K)`. This is the global tensor shape from the user perspective.
- `placement`: a `PlacementSpec` constructed by `T.placement`, describing how the global tensor is sharded or replicated across cores.
- `device_mesh_config`: shaped like `(nrows, ncols)`. Omit it in normal kernels so the target's symbolic mesh is used. Pass a concrete tuple only when intentionally specializing the tensor type to one mesh.
- `dtype`: element type, such as `"float16"`, `"bfloat16"`, or `"float32"`.
- `layout`: global data layout in DRAM. When omitted, a rank-1 tensor with a regular dtype defaults to a 1024-byte-aligned row-major layout, while a rank >= 2 tensor defaults to ZZ. A rank >= 2 tensor with an MX dtype defaults to MXZZ; rank-1 MX tensors are unsupported. Users usually do not need to pass this parameter manually.

After entering the kernel, a `MeshTensor` parameter corresponds to the local shard visible to the current core. Use `A.global_shape` for the complete logical shape, `A.local_shape` for the uniformly allocated local slot shape, and `A.get_local_extent()` for the current core's valid extent.

**`make_row_major`**

```python
from tilelang.layout import make_row_major

layout = make_row_major(shape)
```

`make_row_major` constructs a row-major layout.

- `shape`: logical tensor shape.
- Return value: a layout object that can be passed to `T.MeshTensor(..., layout=layout)` or `T.annotate_layout`.
- Common usage: explicitly express row-major layout in advanced scenarios, or debug layout-related issues. Ordinary kernels usually rely on the default layout selected for their rank and dtype.

**`make_zz_layout`**

```python
from tilelang.layout import make_zz_layout

layout = make_zz_layout(shape_or_buffer, axes=None, block_shape=(32, 32))
```

`make_zz_layout` constructs a block-level ZZ layout. Ordinary GEMM or block-level memory access usually lets the compiler infer or derive the required layout. This function is mainly for interfacing with externally fixed formats, reproducing an existing layout, or debugging.

- `shape_or_buffer`: can be a shape, buffer, buffer load, or buffer region.
- `axes`: dimensions that participate in the blocked layout. When omitted, the last two dimensions are used by default, so the rank must be at least 2.
- `block_shape`: shape of each layout block. A common value is `(32, 32)`.
- Return value: a layout object.

Explicit construction example:

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

`make_zn_layout` constructs a ZN layout.

- `shape`: logical tensor shape.
- `axes`: dimensions participating in the layout transform.
- `block_shape`: block shape.
- Return value: a layout object.
- Common usage: explicitly specify a format needed by a specific matrix operand.

**`make_zzz_layout`**

```python
from tilelang.layout import make_zzz_layout

layout = make_zzz_layout(shape, axes, block_shape, cluster_shape)
```

`make_zzz_layout` constructs a ZZZ layout with cluster organization.

- `shape`: logical tensor shape.
- `axes`: dimensions participating in the layout transform.
- `block_shape`: shape of each block.
- `cluster_shape`: cluster-level organization shape.
- Return value: a layout object.

**`make_nzz_layout`**

```python
from tilelang.layout import make_nzz_layout

layout = make_nzz_layout(shape, axes, block_shape, cluster_shape)
```

`make_nzz_layout` constructs an NZZ layout with cluster organization.

- `shape`: logical tensor shape.
- `axes`: dimensions participating in the layout transform.
- `block_shape`: shape of each block.
- `cluster_shape`: cluster-level organization shape.
- Return value: a layout object.

**`T.annotate_layout`**

```python
T.annotate_layout({buffer: layout})
```

`T.annotate_layout` explicitly attaches a layout to a buffer. This is an advanced usage.

- Parameter: a dict whose key is a buffer and whose value is a layout object.
- `buffer`: the object to annotate with a layout.
- `layout`: constructed by functions such as `make_row_major` or `make_zz_layout`.
- Common usage: interfacing with externally fixed formats, reproducing existing layouts, or debugging layout-related issues. Ordinary kernels usually rely on compiler inference.

**`make_tileview`**

```python
from tilelang.tileview import make_tileview

tileview = make_tileview(buffer, tile_shape, index_map)
```

`make_tileview` describes how a buffer is divided into tiles, and is often used together with `T.Tiles`.

- `buffer`: the buffer, buffer load, or buffer region to tile.
- `tile_shape`: shape of each tile, for example `[16, 32]`.
- `index_map`: buffer dimensions participating in tiling, for example `[-2, -1]` means the last two dimensions.
- Return value: a TileView object.

For common 2D tiles on SunMMIO, the last dimension is often 32, and the height is often 8, 16, or 32. Complex access patterns should explicitly use TileView.

**`T.annotate_tileview`**

```python
T.annotate_tileview({buffer: make_tileview(buffer, tile_shape, index_map)})
T.annotate_tileview({buffer: (tile_shape, index_map)})
```

`T.annotate_tileview` annotates a buffer with a TileView.

- Parameter: a dict whose key is a buffer.
- The value can be the return value of `make_tileview(...)`.
- The value can also be written as the tuple shorthand `(tile_shape, index_map)`.
- `tile_shape`: tile shape.
- `index_map`: dimensions participating in tiling.

Use this API when the access pattern in `T.Tiles` is complex, or when a stable tile domain needs to be expressed.

### 3.3 On-Chip Storage

**`T.alloc_shared`**

```python
T.alloc_shared(shape, dtype, scope="shared.dyn")
```

`T.alloc_shared` allocates an on-chip SRAM buffer inside a core.

- `shape`: buffer shape, such as `(block_M, block_N)`.
- `dtype`: element type, such as `"float16"`, `"bfloat16"`, or `"float32"`.
- `scope`: on-chip SRAM scope. The default is `"shared.dyn"`. Common SunMMIO scopes include `"shared.rsram"`, `"shared.asram"`, and `"shared.wsram"`.
- Return value: a buffer that can be read and written inside the kernel.

Common scopes:

- `shared.asram`: Tensor Core left-operand path.
- `shared.wsram`: Tensor Core right-operand path.
- `shared.rsram`: result operand, intermediate results, and Vector Core processed data.

Ordinary GEMM code can start with default shared buffers. Explicit scopes are only needed when users want to bind hardware data paths more directly.

**`T.clear`**

```python
T.clear(buffer)
```

`T.clear` zeroes a buffer or region.

- `buffer`: a complete buffer, slice, or region.
- Return value: a statement expressing the clear operation.
- Common usage: initialize the result operand before GEMM, or initialize a temporary buffer.

**`T.fill`**

```python
T.fill(buffer, value)
```

`T.fill` fills a buffer or region with a specified value.

- `buffer`: a complete buffer, slice, or region.
- `value`: fill value. Its type should be convertible to the buffer dtype.
- Return value: a statement expressing the fill operation.
- Common usage: initialize constants, set default mask values, or prepare reduce initial values.

### 3.4 Data Movement

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

`T.copy` is the recommended entry point for data movement.

- `src`: source buffer, slice, region, or scalar position.
- `dst`: destination buffer, slice, region, or scalar position.
- `coalesced_width`: hint for coalesced access width. The default is `None`.
- `disable_tma`: switch for copy path selection. SunMMIO users usually keep the default.
- `eviction_policy`: cache/replacement policy hint. It can be `"evict_normal"`, `"evict_first"`, `"evict_last"`, or `None`.
- `annotations`: additional annotation dict. If it appears together with separate parameters, values in the dict take precedence.
- `loop_layout`: layout hint for the copy loop. Ordinary SunMMIO code usually does not need to fill this in.
- Return value: a statement expressing the copy operation.

Common SunMMIO paths include DRAM -> RSRAM, RSRAM -> DRAM, DRAM/RSRAM -> ASRAM, DRAM/RSRAM -> WSRAM, and RSRAM -> RSRAM. Unsupported paths are listed in 2.7.

**`T.transpose`**

```python
T.transpose(src, dst)
```

`T.transpose` transposes a matrix between RSRAM buffers through A4E ODMA.

- `src`: a complete rank-2 RSRAM buffer with shape `[M, N]`.
- `dst`: a complete rank-2 RSRAM buffer with shape `[N, M]`. It must not alias `src`.
- The source and destination must have the same dtype. The supported dtypes are `bfloat16` and `float32`.
- Shapes must be static, with every dimension a multiple of 32. Both buffers must use the same two-level 32x32 blockwise layout family, either ZZ or ZN.
- Return value: a statement expressing the transpose operation.

ODMA transpose is asynchronous. The compiler inserts synchronization before a dependent access. Slices, partial regions, and buffers outside RSRAM are not supported.

### 3.5 Tile Loop

**`T.Tiles`**

```python
for i, j in T.Tiles(domain, parallel=False):
    ...
```

`T.Tiles` constructs a tile-level loop.

- `domain`: logical loop domain. The recommended form is an explicit shape, such as `[block_M, block_N]`. A buffer can also be passed, meaning `buffer.shape` is used.
- `parallel`: whether to declare that there is no cross-iteration dependency between different tile iterations. The default is `False`.
- Return value: tile loop iteration variables. The number of variables is determined by the rank of `domain`.

When the loop body contains accumulation, reduction, or cross-iteration dependencies, do not set `parallel` to `True`.

### 3.6 Inter-Core Communication

Inter-core communication functions transfer data between cores. Source data can come from an on-chip buffer, or directly from a DRAM-side `MeshTensor` slice/region. The latter is often used to directly take a block from the current core's DRAM shard as the input to broadcast or all-gather. The destination is usually an on-chip buffer on the receiver side, because subsequent computation usually consumes inter-core communication results from on-chip SRAM.

**`T.comm.broadcast`**

```python
T.comm.broadcast(src, dst, src_core, direction="all", size=-1)
```

`broadcast` expresses one source core broadcasting data to multiple cores along a specified direction.

- `src`: source buffer or region. It can be an on-chip buffer or a DRAM-side `MeshTensor` slice/region.
- `dst`: receive buffer or region, usually an on-chip buffer.
- `src_core`: source core, either a linear id or `(row, col)`.
- `direction`: broadcast direction. Supported values are `"horizontal"` / `"h"`, `"vertical"` / `"v"`, and `"all"` / `"a"`.
- `size`: number of elements to broadcast. `-1` means using the whole source region.
- Return value: a statement expressing the broadcast.

`horizontal` propagates along the same row, `vertical` propagates along the same column, and `all` covers the whole mesh.

**`T.comm.put`**

```python
T.comm.put(src, dst, src_core, dst_core, size=-1)
```

`put` expresses point-to-point send.

- `src`: source buffer or region. It can be an on-chip buffer or a DRAM-side `MeshTensor` slice/region.
- `dst`: destination buffer or region, usually an on-chip buffer on the destination core.
- `src_core`: source core, either a linear id or `(row, col)`.
- `dst_core`: destination core, either a linear id or `(row, col)`.
- `size`: number of elements to send. `-1` means using the whole source region.
- Return value: a statement expressing the put.

`put` is suitable for one-to-one data delivery. If the same data needs to be sent to a row, a column, or the whole mesh, `broadcast` is usually used.

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

`all_gather` expresses that multiple cores each contribute a local segment, and the receive buffer concatenates the segments.

- `send_buffer`: the source buffer or region contributed by the current core. It can be an on-chip buffer or a DRAM-side `MeshTensor` slice/region.
- `recv_buffer`: buffer or region receiving the concatenated result, usually an on-chip buffer.
- `direction`: direction of participating cores. Supported values are `"horizontal"` / `"h"`, `"vertical"` / `"v"`, and `"all"` / `"a"`.
- `size`: number of elements sent by each core. `-1` means using the whole `send_buffer`.
- `axis`: concatenation dimension. `None` means adding a new dimension 0; `0` means concatenating along dimension 0; `-1` means concatenating along the last dimension.
- `src_offset_byte`: compiler-internal source address byte offset used by bf16 GEMM legalization. User code must leave it as `0`.
- Return value: a statement expressing all-gather.

The receive shape must match `direction` and `axis`. If the number of participating cores is `K`:

- `axis=None`: `recv_buffer.shape == [K, *send_buffer.shape]`.
- `axis=0`: dimension 0 expands to `K * send_buffer.shape[0]`.
- `axis=-1`: the last dimension expands to `K * send_buffer.shape[-1]`.

**`T.comm.all_reduce`**

```python
T.comm.all_reduce(buffer, out, reduce_type, direction, clear=True)
```

`all_reduce` expresses cross-core-only reduction. For every local element, it reduces the corresponding values from the cores selected by `direction`; it does not perform a reduction along a local buffer dimension.

- `buffer`: input buffer or region participating in reduce.
- `out`: output buffer or region storing the reduce result.
- `reduce_type`: reduce type. Supported values include `"sum"`, `"abssum"`, `"max"`, `"min"`, `"absmax"`, `"bitand"`, `"bitor"`, and `"bitxor"`.
- `direction`: direction participating in reduce. Supported values are `"horizontal"` / `"h"`, `"vertical"` / `"v"`, and `"all"` / `"a"`.
- `clear`: keyword-only argument controlling whether `out` is initialized with the reducer's neutral value. If `False`, the cross-core result is combined with the existing `out` value. The default is `True`.
- Return value: a statement expressing all-reduce.

The input and output regions must have the same rank and extents. Use `T.reduce` when a dimension inside the local buffer must also be reduced.

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

`T.gemm` expresses Tensor Core GEMM / MMA.

- `A`: left operand buffer or region. The hardware path corresponds to ASRAM.
- `B`: right operand buffer or region. The hardware path corresponds to WSRAM.
- `C`: result operand buffer or region. The hardware path corresponds to RSRAM, and it carries the accumulation and output role.
- `transpose_A`: whether to interpret the left operand as transposed. The default is `False`.
- `transpose_B`: whether to interpret the right operand as transposed. The default is `False`.
- `policy`: GEMM warp allocation policy. SunMMIO usually uses the default.
- `clear_accum`: whether the GEMM operation clears the result operand. The default is `False`. A common pattern is to explicitly call `T.clear(C)` before GEMM.
- `k_pack`: packed K parameter. Ordinary SunMMIO users keep the default `1`.
- `wg_wait`: wait batch parameter. Ordinary SunMMIO users keep the default `0`.
- `mbar`: barrier parameter. Ordinary SunMMIO users keep the default `None`.
- Return value: a statement expressing GEMM.

Scopes can be inferred from the parameter roles of `A/B/C`. Explicit scope usage is mainly for explanation or for scenarios that need stable control of data paths.

### 3.8 Reduction

**`T.reduce`**

```python
T.reduce(buffer, out, reduce_type, dim, clear)
```

`T.reduce` expresses local reduction on an on-chip buffer.

- `buffer`: input buffer or region.
- `out`: output buffer or region.
- `reduce_type`: reduce type, such as `"sum"`, `"max"`, `"min"`, `"abssum"`, or `"absmax"`.
- `dim`: reduction dimension.
- `clear`: whether to clear the output before reduce.
- Return value: a statement expressing reduce.

Use `T.comm.all_reduce` for cross-core reduce.

**`T.reduce_sum`** **/** **`T.reduce_max`** **/** **`T.reduce_min`**

```python
T.reduce_sum(buffer, out, dim=-1, clear=True)
T.reduce_max(buffer, out, dim=-1, clear=True)
T.reduce_min(buffer, out, dim=-1, clear=True)
```

These functions are shortcut entries for common local reductions.

- `buffer`: input buffer or region.
- `out`: output buffer or region.
- `dim`: reduction dimension. The default is the last dimension.
- `clear`: whether to clear the output before reduce. The default is `True`.
- Return value: a statement expressing reduce.

**`T.reduce_abssum`** **/** **`T.reduce_absmax`**

```python
T.reduce_abssum(buffer, out, dim=-1, clear=True)
T.reduce_absmax(buffer, out, dim=-1, clear=True)
```

These functions are for absolute-value-related local reductions.

- `buffer`: input buffer or region.
- `out`: output buffer or region.
- `dim`: reduction dimension. The default is the last dimension.
- `clear`: whether to clear the output before reduce. The default is `True`.
- Return value: a statement expressing reduce.

**`T.reduce_bitand`** **/** **`T.reduce_bitor`** **/** **`T.reduce_bitxor`**

```python
T.reduce_bitand(buffer, out, dim=-1, clear=True)
T.reduce_bitor(buffer, out, dim=-1, clear=True)
T.reduce_bitxor(buffer, out, dim=-1, clear=True)
```

These functions are for local bitwise reductions in integer or bit-mask scenarios.

- `buffer`: input buffer or region.
- `out`: output buffer or region.
- `dim`: reduction dimension. The default is the last dimension.
- `clear`: whether to clear the output before reduce. The default is `True`.
- Return value: a statement expressing reduce.

### 3.9 Kernel and Loops

**`T.Kernel`**

```python
with T.Kernel() as cid:
    ...

# Explicit equivalent
with T.Kernel(T.mesh_ncores()) as cid:
    ...
```

`T.Kernel` creates the kernel/core execution entry.

- `blocks`: omit it, or pass exactly `T.mesh_ncores()`. Explicit integers and multidimensional grids are not supported for SunMMIO.
- `threads`: leave this unset. SunMMIO kernels are threadless at the TileLang launch level.
- `is_cpu`: whether to generate a CPU-style kernel. SunMMIO users keep the default `False`.
- `prelude`: extra injected code. Ordinary SunMMIO users keep the default `None`.
- Return value: the linear core id binding `cid`.

**`T.serial`**

```text
T.serial(start, stop=None, step=None, *, annotations=None)

for i in T.serial(stop):
    ...

for i in T.serial(start, stop, step):
    ...
```

`T.serial` expresses a serial loop.

- `start`: start value. If only one argument is passed, that argument is used as `stop`, and `start` defaults to 0.
- `stop`: end value. The loop range excludes `stop`.
- `step`: step size. The default is 1.
- `annotations`: loop annotation dict. Ordinary users usually do not need to fill this in.
- Return value: loop variable.

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

`T.Pipelined` expresses a loop that can be pipelined, and is often used for K loops or consecutive tile processing.

- `start`: if `stop` is empty, `start` means loop extent; if `stop` is not empty, `start` means the start value.
- `stop`: end value. The default is `None`.
- `num_stages`: number of pipeline stages. `0` means pipeline is disabled. Common values for GEMM K loops are 2 or 3.
- `order`: stage order control. Advanced usage usually keeps the default.
- `stage`: manual stage annotation. Advanced usage usually keeps the default.
- `sync`: synchronization relationship description. Advanced usage usually keeps the default.
- `group`: pipeline group description. Advanced usage usually keeps the default.
- Return value: loop variable.

The loop body usually contains copy, inter-core communication, and compute. Users need to keep data dependencies clear, for example making sure the related inputs have already been moved or communicated before using a tile for computation.

## 4. Examples

This section gives 3 complete kernel examples. To keep the main structure clear, the examples assume that the problem size is divisible by block size and mesh partitioning. Real engineering code needs additional boundary handling.

### 4.1 Local Shard GEMM

In this kernel, each core only reads the local shard in its own DRAM, performs local GEMM, and writes back to its own output shard. `A` is sharded by mesh row and replicated across columns, while `B` is replicated across rows and sharded by mesh column. Every core therefore owns the full K dimension required for its local output tile.

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

This kernel shows a SUMMA-style dataflow: the same row gathers left-operand blocks through HLink, the same column gathers right-operand blocks through VLink, and then GEMM is executed on the current core.

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

### 4.3 GEMM + In-Tile Post-Processing

This kernel loads a bias tile with the same shape after GEMM, and uses `T.Tiles` to perform element-by-element addition on the result tile.

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

## 5. Summary

- Use the canonical `sunmmio` target string explicitly; `auto` does not detect this backend.
- Prefer symbolic mesh expressions in kernel code and launch with `T.Kernel()`.
- Use `T.placement` for new code, and choose each operand's placement from the algorithm's dataflow rather than reusing one placement mechanically.
- Keep communication regions, receive shapes, SRAM scopes, layouts, and dtypes consistent with the operation that consumes them next.

## Related Documentation

- [SunMMIO TileLang quick start](sunmmio_tilelang_getting_started.md)
- [Installation guide](../get_started/Installation.md)
- [TileLang programming guide](../programming_guides/overview.md)
- [SunMMIO examples](https://github.com/SUNMMIO/Tilelang/tree/tilelang_mesh_main/examples)
