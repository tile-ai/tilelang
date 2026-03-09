# Cluster TMA: Multicast and SM-to-SM Copy

This page describes two advanced data-movement features that are available on
NVIDIA Hopper (SM90) and later: **TMA multicast** and **SM-to-SM cluster
copy**. Both features are exposed through extensions to the existing `T.copy`
operator and require a kernel launched with thread block cluster, i.e., with `cluster_dims != (1, 1, 1)`.

Requirements:
- CUDA Compute Capability ≥ 9.0 (Hopper / Blackwell / RTX 5090)

---

## Background: Thread Block Clusters

A *thread block cluster* is a group of CTAs that share a common virtual address
space for their shared-memory regions and can communicate without going through
global memory. Within a cluster, each CTA has a *block rank* (0-indexed
position inside the cluster), and all CTAs can observe each other's shared
memory via the `shared::cluster` address space.

```python
with T.Kernel(grid_x, grid_y, threads=128, cluster_dims=(4, 1, 1)) as (bx, by):
    rank  = T.block_rank_in_cluster()   # 0..3 within this cluster
    cid   = T.get_cluster_id()           # which cluster am I in
    nctas = T.get_cluster_block_nums()
    T.cluster_sync()                     # barrier across all CTAs in cluster
```

---

## Feature 1 — TMA Multicast (`cluster_mask`)

### What it does

Normally each CTA issues its own TMA load, fetching a tile from global memory
into its private shared memory. With multicast, **a single TMA transaction
broadcasts one global tile to every participating CTA simultaneously**, saving
repeated DRAM traffic when multiple CTAs in a cluster need the same data (e.g.,
the same K-panel in a split-K GEMM).

```text
Global memory ──TMA multicast──▶ shared memory (rank 0)
                              └─▶ shared memory (rank 1)   (same tile, no extra DRAM read)
                  TMA load    ──▶ shared memory (rank 2)   (independent tile)
                  TMA load    ──▶ shared memory (rank 3)   (independent tile)
```

### API

```python
T.copy(src_global, dst_shared, cluster_mask=<int>)
```

`cluster_mask` is a bitmask where each set bit identifies a CTA rank that
participates in the multicast. The CTA whose rank equals the lowest set bit
in the mask issues `cp.async.bulk.tensor … multicast::cluster`; every other
CTA in the mask receives the data passively (no instruction issued). CTAs
outside the mask perform a regular TMA load for their own tile.

### Example

```python
import tilelang
import tilelang.language as T

def make_tma_multicast_kernel(M, N, block_M, block_N, cluster_mask):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
    ):
        # 4 CTAs per cluster; ranks 0 and 1 share the same tile via multicast.
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
            cluster_dims=(4, 1, 1)
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), "float16")

            # cluster_mask=0b0011: ranks 0 and 1 participate.
            # Rank 0 issues tma_load_multicast; rank 1 receives passively.
            # Ranks 2 and 3 each issue a regular tma_load.
            T.copy(A[by * block_M, bx * block_N], A_shared,
                   cluster_mask=cluster_mask)

            T.copy(A_shared, B[by * block_M, bx * block_N])

    return kernel
```

Running the kernel above with `cluster_mask = 0b0011`:

| Rank | Action | `B` slice receives |
|------|--------|--------------------|
| 0 | issues multicast load | A tile at rank-0 address |
| 1 | passively receives | **same** A tile as rank 0 |
| 2 | regular TMA load | A tile at rank-2 address |
| 3 | regular TMA load | A tile at rank-3 address |

### Notes

- The compiler lowers `cluster_mask != 0` to
  `cp.async.bulk.tensor.Nd.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster`
    for the issuing CTA; CTAs in the mask but not elected as issuer receive
    passively, and only CTAs outside the mask issue a standard
    `cp.async.bulk.tensor`.
- Software-pipelining (`T.Pipelined`) is fully supported; the warp-specialized
  rewriter recognises `tma_load_multicast` as a producer operation.
- `cluster_mask` is a compile-time constant; dynamic masks are not supported.

---

## Feature 2 — SM-to-SM Cluster Copy (`dst_block`)

### What it does

SM-to-SM copy lets one CTA **push data directly from its own shared memory
into another CTA's shared memory** within the same cluster, without a round
trip through global memory. This is useful for patterns such as:

- Partial result exchange (e.g., split-K partial sums across SM boundaries)
- Producer–consumer pipelines where the producer fills a neighbor's buffer
- All-to-all collective communication within a cluster

Two sub-variants are provided depending on whether an mbarrier is supplied:

| Variant | Parameter | Hardware instruction | Threads used |
|---------|-----------|---------------------|--------------|
| **Fast path** | `dst_block` + `remote_barrier` | `cp.async.bulk.shared::cluster` | 1 (async DMA) |
| **Slow path** | `dst_block` only | `map_shared_rank` + scalar stores | all (SIMT loop) |

### Fast path — bulk async copy with mbarrier

```python
T.copy(src_shared, dst_shared, dst_block=<rank>, remote_barrier=<mbarrier>)
```

A single elected thread issues one `cp.async.bulk.shared::cluster` instruction.
The hardware DMA engine transfers the entire tile asynchronously and signals
the destination CTA's mbarrier on completion. The destination CTA waits with
`T.mbarrier_wait_parity`.

Steps:
1. Both CTAs allocate the **same** shared memory layout so their mbarriers live
   at the same offset.
2. Every CTA initialises its own barrier for 1 arrival.
3. The source CTA (`pid == 0` below) calls `T.copy(... dst_block=1, remote_barrier=...)`.
4. The destination CTA (`pid == 1`) waits on its local barrier copy.

```python
import tilelang
import tilelang.language as T

@tilelang.jit(verbose=True, execution_backend="cython")
def make_cluster_copy_kernel(N: int):
    @T.prim_func
    def kernel(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(2, threads=128, cluster_dims=(2, 1, 1)) as pid:
            s_src     = T.alloc_shared((N,), "float32")
            s_dst     = T.alloc_shared((N,), "float32")
            s_barrier = T.alloc_shared((1,), "uint64")

            T.fill(s_src, 0.0)
            T.fill(s_dst, 0.0)

            # Each CTA initialises its own barrier: 1 expected arrival.
            if T.get_thread_binding() == 0:
                T.mbarrier_init(s_barrier[0], 1)

            T.cluster_sync()

            if pid == 0:
                # Load A into local shared memory.
                for i in T.Parallel(N):
                    s_src[i] = A[i]

                # Async-push s_src → s_dst in CTA 1, signal CTA 1's barrier.
                T.copy(s_src, s_dst, dst_block=1,
                       remote_barrier=s_barrier[0])

            if pid == 1:
                # Wait until CTA 0 finishes writing.
                T.mbarrier_wait_parity(s_barrier[0], 0)

                for i in T.Parallel(N):
                    B[i] = s_dst[i]

    return kernel
```

Generated producer code (single-thread guard, one PTX instruction):

```cuda
if (((int)threadIdx.x) == 0) {
    tl::tma_store_cluster(&s_dst[0], &s_src[0], 1,
                          (uint32_t)(N * 4), s_barrier[0]);
}
```

### Slow path — element-by-element SIMT fallback

Omit `remote_barrier` to use the slow path:

```python
T.copy(s_src, s_dst, dst_block=1)
```

This lowers to a SIMT parallel loop where every thread writes one (or a few)
elements into the remote CTA's shared memory via
`cooperative_groups::map_shared_rank`. Because `map_shared_rank` returns a
scalar pointer, vectorised writes are not possible. Use this path only when an
mbarrier is unavailable or when the tile is too small to justify barrier
overhead.

### Synchronisation contract

| | Fast path | Slow path |
|-|-----------|-----------|
| Source CTA | no wait needed; copy is async | effectively sync after the loop |
| Destination CTA | `T.mbarrier_wait_parity(barrier, parity)` | external `T.cluster_sync()` or equivalent |

### Notes

- Both paths require `src` and `dst` to be in `shared` or `shared.dyn` scope.
- The mbarrier must be allocated with `T.alloc_shared((count,), "uint64")` and
  initialised with `T.mbarrier_init` before use.
- `T.cluster_sync()` after allocation but before the copy is required to ensure
  all CTAs have reached the barrier-init barrier before any data is pushed.
- `dst_block` may be a compile-time integer or a runtime `tir.PrimExpr`.

---

## Cluster Helper Builtins

| Builtin | Return | Description |
|---------|--------|-------------|
| `T.get_cluster_id()` | `int32` | Index of this cluster in the grid |
| `T.block_rank_in_cluster()` | `int32` | Block rank (0-indexed) within the cluster |
| `T.get_cluster_block_nums()` | `int32` | Total number of CTAs in the cluster |
| `T.cluster_sync()` | — | Barrier synchronisation across all cluster CTAs |
| `T.mbarrier_init(bar, count)` | — | Initialise an mbarrier for `count` arrivals |
| `T.mbarrier_arrive(bar)` | — | Signal one arrival on an mbarrier |
| `T.mbarrier_wait_parity(bar, parity)` | — | Wait until `bar` flips to `parity` |

---

## Putting It Together: Split-K Sketch

A common pattern combining both features: multicast the shared K-panel to
all cluster CTAs (saving DRAM bandwidth), then reduce partial sums with
SM-to-SM copy (saving global-memory round trips).

```python
@T.prim_func
def split_k_gemm(A, B, C):
    with T.Kernel(grid_x, grid_y, threads=256, cluster_dims=(4, 1, 1)) as (bx, by):
        rank    = T.block_rank_in_cluster()
        A_s     = T.alloc_shared((BM, BK), "float16")
        B_s     = T.alloc_shared((BK, BN), "float16")
        C_f     = T.alloc_fragment((BM, BN), "float32")
        C_s     = T.alloc_shared((BM, BN), "float32")
        barrier = T.alloc_shared((1,), "uint64")
        T.clear(C_f)

        # Phase 1: each CTA loads its K-slice; A is multicast to rank 0 and 1.
        for ko in T.Pipelined(T.ceildiv(K, BK * 4), num_stages=3):
            k_off = (rank + ko * 4) * BK
            T.copy(A[by * BM, k_off], A_s, cluster_mask=0b0011)
            T.copy(B[k_off, bx * BN], B_s)
            T.gemm(A_s, B_s, C_f)

        # Phase 2: push each rank's partial sums to rank 0 for accumulation.
        #
        # Use a per-rank staging slot so every non-zero rank writes to a
        # distinct destination region — avoiding both a destination race and
        # an arrival-count mismatch.  Each CTA stores its own partial into
        # C_parts[rank]; non-zero ranks then push that slot to the matching
        # slot in rank 0's shared memory.
        #
        # Arrival count must equal the number of producers: cluster_size - 1.
        C_parts = T.alloc_shared((4, BM, BN), "float32")  # one slot per rank
        T.copy(C_f, C_parts[rank])

        # Only rank 0 needs its barrier initialised (it is the sole consumer).
        # Arrival count = 3: ranks 1, 2, and 3 each signal exactly once.
        if T.get_thread_binding() == 0 and rank == 0:
            T.mbarrier_init(barrier[0], 3)
        T.cluster_sync()

        if rank != 0:
            # Push this rank's slot to the *same* slot index in rank 0's
            # C_parts — different offsets, so no destination race.
            T.copy(C_parts[rank], C_parts[rank],
                   dst_block=0, remote_barrier=barrier[0])

        if rank == 0:
            T.mbarrier_wait_parity(barrier[0], 0)  # wakes after all 3 arrivals
            # C_parts[0..3] in rank 0's smem now hold all four partial sums.
            # accumulate and store ...
            T.copy(C_parts[0], C[by * BM, bx * BN])
```

---

## See Also

- `testing/python/cuda/test_tma_multicast_demo.py` — multicast validation
- `testing/python/cuda/test_tma_dsmem.py` — SM-to-SM copy validation
- Programming Guides → Instructions — complete `T.copy` parameter reference
- Programming Guides → Control Flow — `T.Pipelined` and warp-specialized pipelines
