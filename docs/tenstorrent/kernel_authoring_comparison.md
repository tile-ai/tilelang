# GPU vs. Tenstorrent — Kernel Authoring Patterns

This note complements the architecture comparison by focusing on **how you author kernels** for GPUs (CUDA‑style) vs. **Tenstorrent (TT‑Metalium)**, including how TileLang maps a grid‑style kernel body to TT’s persistent execution.

---

## Side‑by‑Side (authoring perspective)

| Topic | GPU (CUDA‑style / Triton) | Tenstorrent (TT‑Metalium) / TileLang→TT |
|---|---|---|
| **Launch unit** | `<<<grid, block>>>`: many **thread blocks**, each scheduled dynamically to an SM. | Host builds a **Program**, selects **CoreRange/CoreRangeSet**, and launches **persistent kernels** (reader / compute / writer) on each participating core. |
| **Kernel body indexers** | Use `blockIdx.{x,y,z}`, `threadIdx.{x,y,z}`; or high‑level schedules (Triton’s `tl.arange`, `num_warps/stages`). | **Author in TileLang with `bx,by`** (grid blocks). Backend generates a **per‑core outer loop** that recovers `(bx,by)` from a static tile list `(start_id, count)`. |
| **Thread‑level parallelism** | Warps/threads cooperate; shared memory tiling; `__syncthreads()`. | No SIMT threads. Compute engines operate on **tiles**; parallelism comes from **cores** and **pipelined CBs**. |
| **Persistent execution** | Optional (persistent‑threads pattern) but not required. | **Default**: per‑core **persistent loop** over assigned tiles. |
| **Data staging** | Global→**shared memory** via `cp.async`/TMA; compute from shared; write back to global. | **DRAM tiles → L1 Circular Buffers (CBs)** via **reader**; compute consumes from CBs; writer drains CBs → DRAM. |
| **SW pipelining** | `cp.async` stages; double/triple buffering within a block. | **CB depth** and **separate kernels** (reader/compute/writer) implement double/multi‑buffering. |
| **Block / tile size** | Chosen by developer; may align with tensor core fragment sizes. | **Native 32×32 tiles** (dtype‑dependent bytes). Pad edges or handle tails explicitly. |
| **Work distribution** | Implicit via hardware scheduler; limited control. | **Explicit**: choose **contiguous / strided / rectangular** carves; pass per‑core `(start_id, count)`; or generate rectangle loops. |
| **Multicast / reuse** | Generally via global/L2; SM‑to‑SM multicast is limited/specialized. | **Explicit multicast** over on‑chip NoC to **rectangles** of cores for A/B or Q/K reuse. |
| **Synchronization** | Barrier in a block (`__syncthreads()`); memory scopes for cp.async groups. | Synchronization implicit in **CB protocols** and kernel staging; reader/compute/writer coordination. |
| **Tuning knobs** | `num_warps`, `num_stages`, block size, shared‑mem footprint, occupancy. | **Core ranges**, **CB counts/depths**, **tile carve policy**, **chunk sizes** (e.g., `Kt`, `Sk`), multicast topology. |
| **Annotations (TileLang)** | Layout hints; vectorization, etc. | `annotate_tt_schedule(...)` for static schedule; `annotate_tt_sharding(...)` for tilization/sharding/placement. |
| **Fallbacks / libraries** | cuBLAS/cuDNN or Triton kernels. | TTNN ops or Metalium templates; untilize/tilize helpers. |
| **Debugging** | Nsight Systems/Compute; kernel printf. | Host logs, **plan dumps** (e.g., `tt.plan.json` in our backend), device traces. |

---

## Code skeletons

### GPU‑style (TileLang body written against `bx,by`)

```python
import tilelang.language as T
BLOCK = 32

@T.prim_func
def gemm(A: T.Buffer((M, K), "bf16"),
         B: T.Buffer((K, N), "bf16"),
         C: T.Buffer((M, N), "bf16")):
    Mt, Nt, Kt = T.ceildiv(M, BLOCK), T.ceildiv(N, BLOCK), T.ceildiv(K, BLOCK)
    with T.Kernel(grid_x=Nt, grid_y=Mt, threads=(32, 4)) as (bx, by):
        i0, j0 = by * BLOCK, bx * BLOCK
        Cacc = T.alloc_fragment((BLOCK, BLOCK), "bf16"); T.fill(Cacc, 0)
        for kk in range(Kt):
            Ablk = T.alloc_shared((BLOCK, BLOCK), "bf16")
            Bblk = T.alloc_shared((BLOCK, BLOCK), "bf16")
            T.copy(T.region(A[i0, kk*BLOCK], "r", BLOCK, BLOCK), Ablk)
            T.copy(T.region(B[kk*BLOCK, j0], "r", BLOCK, BLOCK), Bblk)
            T.gemm(Ablk, Bblk, Cacc)
        T.copy(Cacc, T.region(C[i0, j0], "w", BLOCK, BLOCK))
```

### Tenstorrent compute stub (generated outer scheduler loop; pseudo‑C++)

```cpp
// per-core runtime args: start_id, count, grid_x (Nt), Kt, etc.
for (uint32_t i = 0; i < count; ++i) {
    uint32_t tid = start_id + i;
    uint32_t by  = tid / grid_x;
    uint32_t bx  = tid % grid_x;

    // Reader has already queued A(by,kk) and B(kk,bx) into L1 CBs.
    for (uint32_t kk = 0; kk < Kt; ++kk) {
        // tile GEMM primitive; indices derived from (by, bx, kk)
        // ckernel::matmul_tiles(cb_a, cb_b, /*indices*/, dst_cb);
    }
    // Writer drains C tile from CB to DRAM.
}
```

---

## Adoption steps (TileLang → TT)

1. **Keep your kernel body GPU‑style** (index by `bx,by`).  
2. (Optional) Add `annotate_tt_schedule(...)` to pick **contiguous/strided/rectangular** carve or chunk sizes.  
3. (Optional) Add `annotate_tt_sharding(...)` to specify DRAM tilization & sharding.  
4. Let the backend generate the **per‑core outer loop** and **reader/compute/writer** pipeline; run.

