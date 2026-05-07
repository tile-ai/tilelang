# TileScale v0 — Distributed API Reference

This document covers all APIs extended on top of TileLang for distributed multi-GPU programming.

> **Environment**: Set `export TILELANG_USE_DISTRIBUTED=1` before running any distributed kernel.

---

## 1. Kernel-Side Language Intrinsics (`tilelang.language` / `T.*`)

These are TIR intrinsics used inside `@T.prim_func` kernel definitions.

### 1.1 Rank Queries

| API | Return | Description |
|-----|--------|-------------|
| `T.get_rank()` | `uint64` | Current rank (PE index) |
| `T.get_num_ranks()` | `uint64` | Total number of ranks |

### 1.2 Remote Copy (NVSHMEM-style, requires nvshmem — not available in v0)

| API | Scope | Description |
|-----|-------|-------------|
| `T.put_warp(src, dst, size, dst_pe, unroll_factor, aggressive_vec)` | Warp | Push to remote buffer |
| `T.get_warp(src, dst, size, src_pe, unroll_factor, aggressive_vec)` | Warp | Fetch from remote buffer |
| `T.put_block(src, dst, size, dst_pe)` | Block | Push to remote buffer |
| `T.get_block(src, dst, size, src_pe)` | Block | Fetch from remote buffer |

### 1.3 Signal / Wait Primitives

Spin-wait on a memory location until the condition is satisfied. Used for inter-rank synchronization without NCCL.

```python
T.wait_eq(value, expected, peer=-1)   # wait until *value == expected
T.wait_ne(value, expected, peer=-1)   # wait until *value != expected
T.wait_ge(value, expected, peer=-1)   # wait until *value >= expected
T.wait_le(value, expected, peer=-1)   # wait until *value <= expected
T.wait_gt(value, expected, peer=-1)   # wait until *value >  expected
T.wait_lt(value, expected, peer=-1)   # wait until *value <  expected
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value` | `PrimExpr` | — | Memory location to poll (pointer expression) |
| `expected` | `PrimExpr` | — | Comparison value |
| `peer` | `PrimExpr` | `-1` | Source PE (unused in most cases) |

**Generated code**: `tl::wait_eq(ptr, val)` etc. in `tl_templates/cuda/sync.h`.

### 1.4 Multimem Operations (NVSwitch SHARP)

Requires multicast-capable hardware (NVSwitch). Operates on multicast virtual addresses allocated via `allocator._allocate_mcast_tensor()`.

#### Reduction Operators

```python
T.MultimemReduceOp.ADD   # Sum
T.MultimemReduceOp.MIN   # Minimum
T.MultimemReduceOp.MAX   # Maximum
T.MultimemReduceOp.NONE  # Plain store (no reduction)
```

#### Operations

| API | Description |
|-----|-------------|
| `T.multimem_ld_reduce(src, dst, reduce_op=ADD)` | Load-reduce from multicast addr → local buffer |
| `T.multimem_st(src, dst)` | Broadcast store from local → multicast addr |
| `T.multimem_red(src, dst, reduce_op=ADD)` | Reduce-accumulate into multicast addr (no read-back) |
| `T.multimem_tma_store(src, dst, reduce_op=None)` | Async TMA store to multicast addr (Hopper+) |
| `T.multimem_signal(addr, value, dtype_tag="uint32")` | Write signal value to multicast address |

All multimem ops use the same `src`/`dst` signature as `T.copy()` — accepts `Buffer`, `BufferLoad`, or `BufferRegion`.

---

## 2. Python Runtime Utilities (`tilelang.distributed`)

### 2.1 Process Initialization

#### `init_dist(local_rank, num_local_ranks) -> (rank, world_size, group)`

Initialize NCCL process group and set CUDA device. Reads `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK` from environment.

```python
from tilelang.distributed import init_dist
rank, num_ranks, group = init_dist(local_rank=0, num_local_ranks=8)
```

### 2.2 Utility Functions

| API | Signature | Description |
|-----|-----------|-------------|
| `CUDA_CHECK(result)` | `(CUresult) -> None` | Assert CUDA driver call succeeded |
| `perf_fn(fn, warmup=50, rep=50)` | `-> float (ms)` | Benchmark with L2 flush, returns avg ms |
| `has_fullmesh_nvlink()` | `-> bool` | Check if all GPU pairs have NVLink |
| `set_signal(tensor, value, stream)` | Host-side `cuStreamWriteValue32` | |
| `wait_eq(tensor, value, stream)` | Host-side `cuStreamWaitValue32` | |
| `dist_print(*args, allowed_ranks=[0])` | Rank-filtered print | |
| `supports_p2p_native_atomic()` | `-> bool` | Check P2P atomic support between GPU 0 and 1 |

### 2.3 Tensor Creation

#### `create_tensor(shape, dtype) -> torch.Tensor`

Allocate via `cudaMalloc` (IPC-compatible).

#### `create_dist_tensor(local_rank, num_local_ranks, data, rank, group, use_vmm=None) -> torch.Tensor`

Exchange IPC/VMM handles across ranks, return a `uint64` tensor of peer base pointers.

#### `create_mapped_tensor(shape, dtype) -> (host_tensor, device_tensor)`

Pinned host + CUDA device tensor pair.

---

## 3. Shared Memory FFI (`tilelang.distributed.shared_memory`)

Low-level C++ FFI functions for IPC and VMM handle management.

### IPC Handles

| Function | Description |
|----------|-------------|
| `_create_ipc_handle(ptr)` | Export CUDA IPC handle for a device pointer |
| `_sync_ipc_handles(handle, rank, group)` | Exchange IPC handles across all ranks |

### VMM Fabric Handles

| Function | Description |
|----------|-------------|
| `_vmm_malloc(size)` | Allocate via VMM fabric |
| `_vmm_free(ptr, size)` | Free VMM allocation |
| `_create_vmm_handle(ptr, size)` | Export VMM handle |
| `_open_vmm_handle(handle)` | Import VMM handle from another rank |
| `_close_vmm_handle(handle)` | Release VMM handle |
| `_sync_vmm_handles(handle, rank, group)` | Exchange VMM handles across all ranks |
| `_supports_vmm_fabric()` | Check if VMM fabric is available |
| `_supports_multicast()` | Check if NVSwitch multicast is available |

### Tensor-from-Pointer

#### `tensor_from_ptr(ptr_val, shape, dtype_str="float32", device=0) -> torch.Tensor`

Create a zero-copy CUDA tensor viewing externally-allocated device memory. Supports all standard dtypes including `bfloat16`, `uint32`, `uint64`.

---

## 4. Distributed Allocator (`tilelang.utils.allocator`)

### `BaseAllocator`

Bump-pointer allocator over a pre-allocated CUDA buffer with cross-rank peer-pointer table and optional multicast buffer.

```python
allocator = tilelang.get_allocator(
    size=2**30,               # 1 GiB
    device="cuda",
    is_distributed=True,
    local_rank=local_rank,
    num_local_ranks=num_local_ranks,
    group=group,
    use_vmm=None,             # None = defer to TILESCALE_USE_VMM env var
    mcast_size=None,          # set for multicast buffer (requires VMM + NVSwitch)
)
```

| Property | Type | Description |
|----------|------|-------------|
| `allocator.device` | `int` | CUDA device index |
| `allocator.table` | `torch.Tensor` | CPU uint64 tensor: `[local_rank, num_ranks, ptr_0, ..., ptr_N-1]` |
| `allocator.table_size` | `int` | Number of table entries |
| `allocator.initialized()` | `bool` | Whether allocation succeeded |

#### Key Methods

| Method | Description |
|--------|-------------|
| `close()` | Collectively free all resources |
| Context manager (`with allocator:`) | Auto-close on exit |

---

## 5. Distributed Tensor Allocation (`tilelang.tensor`)

```python
# Regular tensor (fallback to torch.empty)
t = tilelang.tensor((M, K), dtype=torch.float16, device="cuda")

# Distributed tensor with IPC peer views
ag_buffer = tilelang.tensor((M, K), dtype, allocator=allocator, return_peers=True)
# ag_buffer is a list[torch.Tensor] — one view per rank into the shared buffer

# Distributed tensor without peer views
B = tilelang.tensor((K, N), dtype, allocator=allocator)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | sequence | — | Tensor shape |
| `dtype` | `torch.dtype` | — | Element dtype |
| `device` | device-like | `None` | Must match allocator device if both provided |
| `allocator` | `BaseAllocator` | `None` | Use distributed allocation when provided |
| `return_peers` | `bool` | `None` | Return peer tensors (one per rank) |

---

## 6. JIT Kernel Initialization (`JITKernel.initialize`)

```python
gemm_func = tilelang.jit(...)(kernel_def)
gemm_func.initialize(allocator=allocator)  # upload peer-pointer table to device
gemm_func(A, B, C)                         # run the kernel
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allocator` | `BaseAllocator` | — | Must be initialized |
| `stream` | `int` | `None` | Raw CUDA stream handle; 0 = default stream |

Copies `allocator.table` to device constant memory `meta_data[1024]` via `cudaMemcpyToSymbolAsync`. Must be called before any kernel that reads peer base pointers.

---

## 7. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TILELANG_USE_DISTRIBUTED` | `0` | Enable distributed codegen (include sync/distributed templates) |
| `TILESCALE_USE_VMM` | `0` | Use VMM fabric instead of IPC for shared memory |
| `NCCL_IB_DISABLE` | — | Set to `1` if InfiniBand is not available |

---

## 8. Example Compatibility Matrix

| Example | IPC | VMM | Multicast | Status |
|---------|-----|-----|-----------|--------|
| `example_allgather_gemm_overlapped` | yes | no | no | **Working** |
| `example_allgather_gemm_specialized` | yes | yes | yes | Requires NVSwitch |
| `example_gemm_allreduce` | — | yes | yes | Requires NVSwitch |
| `example_gemm_rs_overlapped` | yes | no | no | Needs `T.atom_add(scope=, sem=)` |
| `example_multimem_allreduce` | — | yes | yes | Requires NVSwitch |
| `example_sp_ag_attention_intra_node` | yes | no | no | Needs `T.barrier_blocks` |
