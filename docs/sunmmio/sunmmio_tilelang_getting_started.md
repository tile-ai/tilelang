# Getting Started with SunMMIO TileLang Kernels

[中文](sunmmio_tilelang_getting_started_zh_cn.md) | [Full user guide](sunmmio_tilelang_user_guide.md)

## SunMMIO A4E Architecture Overview

### Architecture

![SunMMIO NPU architecture](imgs/sunmmio_a4e_architecture.png)

*Overall architecture of the SunMMIO NPU, including the 2D mesh, multiple types of compute units, and on-chip memory hierarchy.*

The SunMMIO NPU has the following architectural features:

- The chip contains 16 Cores organized as a 2D mesh.
- Each Core contains a Tensor Core, Vector Unit, and ODMA.
- Cores can broadcast and exchange data through HLink and VLink.
- On-chip SRAM is divided into `ASRAM / WSRAM / RSRAM`.

### Mental Model for Writing a SunMMIO TileLang Kernel

![Mental model for SunMMIO TileLang](imgs/sunmmio_kernel_mental_model.png)

*Mental model for writing a SunMMIO TileLang kernel, covering sharding, execution, and communication.*

Keep the following mental model in mind when writing a TileLang kernel for SunMMIO:

1. The same kernel runs on different Cores and processes different data on each Core.
2. Host data is sharded across the DRAM of different Cores. Each Core owns only part of the data.
3. A Core may need data owned by other Cores. This requires inter-Core communication such as `all_gather`, `broadcast`, or `put`.

The hardware characteristics of the Tensor Core and Vector Core introduce two additional requirements:

4. Express parallel scalar operations as tile-level operations.
5. Understand the default DRAM tensor layouts and specify a layout only when integrating a fixed format or debugging.

## An Example

The following distributed matrix multiplication example demonstrates the most important concepts for writing a SunMMIO kernel:

- `MeshTensor` / `Placement`
- `T.Kernel`
- Inter-Core communication
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

The factory returns a `PrimFunc`. Compile it with the SunMMIO target explicitly because `target="auto"` does not detect this backend:

```python
import tilelang

func = matmul_persistent(1024, 1024, 1024, 32, 32, 128, 3)
kernel = tilelang.compile(func, target="sunmmio")
```

The following sections build on this example.

## Sharding Tensors Across the Core Mesh

### 1. Sharding Mechanism

![Sharding examples](imgs/sunmmio_mesh_sharding.png)

*Tensor sharding methods on a Core Mesh, showing how logical dimensions map to mesh directions.*

Use `T.MeshTensor` and `T.placement` to define how a global tensor is distributed across the mesh.

The most common forms are:

- `T.placement.full_shard(row_dim, col_dim)`: Shard selected tensor dimensions across mesh rows and columns.
- `T.placement.row_shard(dim)`: Shard across rows and replicate across columns.
- `T.placement.col_shard(dim)`: Replicate across rows and shard across columns.
- `T.placement.replicated()`: Replicate across all Cores.
- `T.placement.mesh_as_line(dim)`: Treat the row-major mesh as one dimension and shard the selected tensor dimension.

In this GEMM example:

```python
placement = T.placement.full_shard(row_dim=0, col_dim=1)

A: T.MeshTensor((M, K), placement, dtype, layout=A_layout)
B: T.MeshTensor((K, N), placement, dtype, layout=B_layout)
C: T.MeshTensor((M, N), placement, accum_dtype, layout=C_layout)
```

`T.placement.full_shard(0, 1)` means:

- Shard dimension 0 along the row direction of the mesh.
- Shard dimension 1 along the column direction of the mesh.

### 2. Using Sharded Tensors in a Kernel

For a `MeshTensor`, `global_shape` is the complete logical shape and `local_shape` is the local slot shape allocated to each Core.

A typical pattern is:

```python
# A: T.MeshTensor((M, K), T.placement.full_shard(0, 1), dtype, ...)
sharded_M, sharded_K = A.local_shape
_, sharded_N = B.local_shape
```

Subsequent loop bounds use these local shapes.

### 3. Sharding in This Example

In this GEMM example, each Core holds a shard of `A` and `B`, not the complete tensors.

- The `A` shards must later be gathered horizontally.
- The `B` shards must later be gathered vertically.
- Each Core is responsible for writing one output shard of `C`.

This is why the kernel uses `all_gather` later.

### 4. A Concrete Sharding Example

Consider a tensor with shape `(64, 32)`:

```python
C: T.MeshTensor(
    (64, 32),
    T.placement.full_shard(row_dim=0, col_dim=1),
    (4, 4),
    dtype,
)
```

Assume the mesh is `4 x 4`.

- `row_dim=0` shards dimension 0 across the rows of the mesh.
- `col_dim=1` shards dimension 1 across the columns of the mesh.

Therefore:

- Dimension 0, with size `64`, is split across 4 Core rows. Each row receives `16` elements.
- Dimension 1, with size `32`, is split across 4 Core columns. Each column receives `8` elements.

The local shard shape visible to each Core is:

```python
(16, 8)
```

If a Core coordinate is written as `(row, col)`, then:

- `(0, 0)` receives `C[0:16, 0:8]`.
- `(0, 1)` receives `C[0:16, 8:16]`.
- `(0, 2)` receives `C[0:16, 16:24]`.
- `(0, 3)` receives `C[0:16, 24:32]`.
- `(1, 0)` receives `C[16:32, 0:8]`.
- `(1, 1)` receives `C[16:32, 8:16]`.
- `...`
- `(3, 3)` receives `C[48:64, 24:32]`.

![full_shard(0, 1) example](imgs/sunmmio_full_shard_example.png)

*Concrete example of `T.placement.full_shard(0, 1)`, showing how a `(64, 32)` tensor is distributed across a `4 x 4` mesh.*

This is how `MeshTensor` works: define the global shape, define its placement on the mesh, and each Core automatically sees its local shard.

## Kernel Launch

### 1. What `T.Kernel()` Does

In this example:

```python
with T.Kernel() as _cid:
    ...
```

`T.Kernel()` uses the target's symbolic mesh. The kernel is launched across the entire mesh, and every Core executes the same kernel code.

`_cid` is the linear ID of the current Core. This example does not use `_cid` directly because `MeshTensor` already handles data distribution.

### 2. Why There Is Only One `T.Kernel`

This example does not add another loop to assign different tiles to different Cores. The distribution is already encoded in `MeshTensor`.

The execution order is:

1. `T.Kernel()` launches the same kernel on all Cores.
2. Each Core automatically receives its shard through `MeshTensor`.
3. Each Core runs block-level loops over its local shard.

### 3. Block Loops in This Example

```python
for bx in T.serial(T.ceildiv(sharded_M, block_M)):
    for by in T.serial(T.ceildiv(sharded_N, block_N)):
        ...
```

`bx` and `by` are block coordinates within the current Core. Their bounds are determined by the local shard dimensions `sharded_M` and `sharded_N`.

## Inter-Core Communication

### 1. Why This Example Needs Inter-Core Communication

In GEMM, each Core receives only local shards of `A` and `B`. To execute one `T.gemm`, the current Core must also collect the other shards from the same mesh row or column.

This example uses `all_gather`.

### 2. `all_gather` for A

```python
T.comm.all_gather(
    A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K],
    A_shared_dist,
    direction="horizontal",
    axis=-1,
)
```

This operation:

- Extracts the `A` block owned by the current Core.
- Gathers data horizontally across the mesh.
- Concatenates the gathered data along the last dimension of `A_shared_dist`.

The shape of `A_shared_dist` is:

```python
(block_M, block_K * T.mesh_ncols())
```

### 3. `all_gather` for B

```python
T.comm.all_gather(
    B[k * block_K : (k + 1) * block_K, by * block_N : (by + 1) * block_N],
    B_shared_dist,
    direction="vertical",
    axis=0,
)
```

This operation:

- Extracts the `B` block owned by the current Core.
- Gathers data vertically across the mesh.
- Concatenates the gathered data along dimension 0 of `B_shared_dist`.

The shape of `B_shared_dist` is:

```python
(block_K * T.mesh_nrows(), block_N)
```

### 4. Running `T.gemm` After Communication

After communication:

- `A_shared_dist` contains the complete row of `A` data required by the GEMM.
- `B_shared_dist` contains the complete column of `B` data required by the GEMM.

The kernel can now execute:

```python
T.gemm(A_shared_dist, B_shared_dist, C_shared)
```

The relationship between communication and computation in this example is straightforward:

1. Slice the local block from the current Core's `MeshTensor`.
2. Run `all_gather`.
3. Run GEMM using the gathered shared buffers.

### 5. Inter-Core Communication Operations Available in the Frontend

The frontend currently provides the following communication operations:

- `T.comm.broadcast(src, dst, src_core, direction=...)`
- `T.comm.put(src, dst, src_core, dst_core)`
- `T.comm.all_gather(send_buffer, recv_buffer, direction=..., axis=...)`
- `T.comm.all_reduce(buffer, out, reduce_type, direction=..., clear=...)`

Their purposes are:

- `broadcast`: Broadcast data from one source Core to a row, column, or the entire mesh.
- `put`: Send data from one source Core to one destination Core.
- `all_gather`: Collect local data from multiple Cores into a larger receive buffer.
- `all_reduce`: Reduce corresponding elements across multiple Cores and write the result to an output buffer with the same local shape. It does not reduce a dimension inside the local buffer.

## Layout

### 1. What Is a Layout?

A layout maps a logical arrangement to a physical arrangement. The logical arrangement is the tensor indexing visible to the user, such as `A[i, j]`. The physical arrangement is the actual storage order of the tensor in DRAM or an on-chip buffer.

The same logical shape can use different physical arrangements. Common layouts include:

- Row major: The last dimension is contiguous.
- Column major: The first dimension is contiguous.
- ZZ: Data is divided into blocks and then stored block by block. This layout is commonly used for GEMM on SunMMIO.

The frontend currently provides the following SunMMIO layout constructors:

- `make_row_major`
- `make_aligned_row_major`
- `make_zz_layout`
- `make_zn_layout`
- `make_zzz_layout`
- `make_nzz_layout`

Their meanings are:

- `make_row_major` creates a standard row-major layout.
- `make_aligned_row_major` creates a row-major layout with alignment constraints.
- `make_zz_layout` creates a blockwise row-major layout.
- `make_zn_layout` creates a blockwise column-major layout.
- `make_zzz_layout` creates a clustered row-major layout.
- `make_nzz_layout` creates a clustered column-major layout.

### 2. When to Specify a Layout Explicitly

On SunMMIO, layout affects:

- Whether Tensor Core and Vector Core accesses are legal.
- Whether DMA memory accesses are aligned.
- Whether data produced by `all_gather` or `broadcast` is arranged correctly for subsequent operators.

Users mainly need to reason about the layout of tensors in **DRAM**. The compiler infers the layouts of tensors in shared memory. When a `MeshTensor` omits `layout`, a regular rank-1 tensor defaults to 1024-byte-aligned row-major, while a regular tensor with rank 2 or greater defaults to a 32x32 ZZ layout on its last two dimensions.

Most kernels can use these defaults. Specify a layout when integrating a fixed external format, choosing non-default blocked axes, or debugging layout behavior. This guide constructs ZZ layouts explicitly to demonstrate the API.

### 3. Common Syntax for Explicit Layouts

This example specifies layouts as follows:

```python
from tilelang.layout import make_zz_layout

A_layout = make_zz_layout((M, K), [0, 1], (32, 32))
B_layout = make_zz_layout((K, N), [0, 1], (32, 32))
C_layout = make_zz_layout((M, N), [0, 1], (32, 32))

A: T.MeshTensor((M, K), placement, dtype, layout=A_layout)
```

The function signature of `make_zz_layout` is:

```python
make_zz_layout(shape_or_buffer, axes=None, block_shape=(32, 32))
```

The parameters are:

- `shape_or_buffer`: The input tensor shape or an existing buffer.
- `axes`: The dimensions to partition into ZZ blocks. If omitted, the last two dimensions are used.
- `block_shape`: The shape of each block. The default is `(32, 32)`.

For example:

```python
make_zz_layout((M, K), [0, 1], (32, 32))
```

This means:

- Create a ZZ layout for a tensor with shape `(M, K)`.
- Partition dimensions 0 and 1 into blocks.
- Use a block size of `32 x 32`.

- The layout is attached to `MeshTensor`.
- The layout describes how the tensor is stored in DRAM.

### Rule of Thumb

Prefer the default `MeshTensor` layout. Use `make_zz_layout` or another explicit layout constructor only for a fixed external format, non-default blocking, or layout debugging.

## FlashAttention Example

The core function from `examples/flash_attention/sunmmio_example_gqa_fwd_bhsd.py` is shown below. The following section explains `T.Tiles` using this code.

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

### 1. Why `T.Tiles` Is Needed

`sunmmio_example_gqa_fwd_bhsd.py` contains many elementwise operations:

- Updating `scores_max`.
- Computing `scores_scale`.
- Applying `exp2`.
- Scaling and normalizing `acc_o` element by element.

These operations are scalar operations mathematically. On SunMMIO, they need to be organized into vector computations that match the 4096-bit vector width.

`T.Tiles` is the frontend construct for expressing this organization.

### 2. `T.Tiles` in This Example

The FlashAttention example uses `T.Tiles` in the following patterns:

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

These code blocks have three properties in common:

- Their inputs and outputs are shared buffers.
- They perform many parallel elementwise operations within a tile.
- The backend must recognize them as tile-level vector computations.

### 3. Comparing Scalar-Parallel Syntax with `T.Tiles`

Without `T.Tiles`, the same logic can be expressed with ordinary scalar-parallel loops.

For example, the causal mask can be written as:

```python
for i, j in T.Parallel(block_M, block_N):
    acc_s[i, j] = T.if_then_else(
        bx * block_M + i >= k * block_N + j,
        0,
        -T.infinity(acc_s.dtype),
    )
```

Similarly, `scores_scale` can be written as:

```python
for i in T.Parallel(block_M):
    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
```

This syntax expresses the computation as many independent scalar operations.

The same logic using `T.Tiles` is:

```python
for i, j in T.Tiles([block_M, block_N]):
    acc_s[i, j] = T.if_then_else(
        bx * block_M + i >= k * block_N + j,
        0,
        -T.infinity(acc_s.dtype),
    )
```

and:

```python
for i in T.Tiles([block_M]):
    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
```

Scalar computations over a buffer can often be expressed as tile-level operations. This allows the backend to lower each tile into Vector Core computation.

### 4. Design Goal of `T.Tiles`

`T.Tiles` organizes parallel scalar operations into tile-level operations. The backend can then process them using a uniform tile structure, simplify memory access, and lower them directly into vector computation that matches the 4096-bit vector width.

In this FlashAttention example:

- `T.Tiles([block_M, block_N])` describes a two-dimensional tile.
- `T.Tiles([block_M])` describes a one-dimensional tile.
- The backend can normalize these tiles into stable tile-level loops.

### 5. When to Use `T.Tiles`

Use `T.Tiles` for large numbers of parallel scalar operations within a buffer. Typical use cases include:

- Elementwise arithmetic on shared buffers.
- Scaling, normalization, and activation within a tile.
- Writing masks within a tile.
- Data type casts within a tile.

### Note

Users do not need to specify the tile size manually. The compiler infers it from the buffer layout and the operations that use the buffer.

## Summary

Keep the following points in mind when writing a SunMMIO TileLang kernel:

- SunMMIO uses a multi-Core mesh architecture. One kernel runs on multiple Cores simultaneously, and each Core processes its own data shard.
- Placement determines how a global tensor is distributed across the Core mesh. `MeshTensor` and `T.placement` define this distribution.
- Layout defines the mapping from logical tensor coordinates to physical storage. Prefer the default layout for ordinary kernels and specify one explicitly for advanced external-format or non-default-blocking scenarios.
- Tile loops express large numbers of parallel scalar operations within a tile. `T.Tiles` organizes these operations into a tile-level structure that the backend lowers into Vector Core computation.
- Inter-Core communication must be written explicitly in the frontend. Common operations are `broadcast`, `put`, `all_gather`, and `all_reduce`. `all_gather` is the key communication operation in the GEMM example.

## Related Documentation

Refer to the following resources together with this guide:

- [SunMMIO TileLang user guide](sunmmio_tilelang_user_guide.md)
- [Installation guide](../get_started/Installation.md)
- [TileLang programming guide](../programming_guides/overview.md)
- [SunMMIO TileLang kernel examples](https://github.com/SUNMMIO/Tilelang/tree/tilelang_mesh_main/examples)
