<img src=./images/logo-row.svg />

<div align="center">

# Tile Language
[![PyPI version](https://badge.fury.io/py/tilelang.svg)](https://badge.fury.io/py/tilelang)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tile-ai/tilelang) [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/TUrHyJnKPG)

</div>

Tile Language (**tile-lang**) is a concise domain-specific language designed to streamline the development of high-performance GPU/CPU kernels (e.g., GEMM, Dequant GEMM, FlashAttention, LinearAttention) as well as accelerators such as [Tenstorrent AI architecture](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md) and Huawei Ascend NPU. By employing a Pythonic syntax with an underlying compiler infrastructure on top of [TVM](https://tvm.apache.org/), tile-lang allows developers to focus on productivity without sacrificing the low-level optimizations necessary for state-of-the-art performance.

<img src=./images/MatmulExample.png />

# TileLang → Tenstorrent (TT-Metalium) Backend

**Status:** Draft proposal for community discussion  
**Goal:** Add a first‑class **Tenstorrent TT‑Metalium** backend to TileLang, alongside the existing NVIDIA (CUDA), AMD (HIP), and Ascend targets.

This README doubles as a **technical plan** and a **call for contributions**. The intended path is to begin in a **public fork** and upstream in stages once CI and core features stabilize.

---

## Table of Contents

- [Motivation](#motivation)
- [Background: Persistent Kernels & Tiles on Tenstorrent](#background-persistent-kernels--tiles-on-tenstorrent)
- [Key Idea: Grid‑to‑Persistent Mapping](#key-idea-grid-to-persistent-mapping)
- [User‑Facing Annotations](#user-facing-annotations)
  - [Static Schedule Annotations](#static-schedule-annotations)
  - [Sharding & Layout Annotations](#sharding--layout-annotations)
  - [Defaults & Backward Compatibility](#defaults--backward-compatibility)
- [End‑to‑End Examples](#end-to-end-examples)
  - [GEMM (no annotations → defaults)](#gemm-no-annotations--defaults)
  - [Attention (with schedule & layout hints)](#attention-with-schedule--layout-hints)
- [Compiler & Codegen Plan (TVM/TileLang)](#compiler--codegen-plan-tvmtilelang)
  - [Phase 0 — MVP (GEMM, Elementwise)](#phase-0--mvp-gemm-elementwise)
  - [Phase 1 — SDPA, Dequant‑GEMM, Reuse/Multicast](#phase-1--sdpa-dequant-gemm-reusemulticast)
  - [Phase 2 — Ergonomics, Safety, Diagnostics](#phase-2--ergonomics-safety-diagnostics)
- [Runtime Integration & Build](#runtime-integration--build)
- [Developer Workflow & Repository Layout](#developer-workflow--repository-layout)
- [Risks & Mitigations](#risks--mitigations)
- [Call for Contributions](#call-for-contributions)
- [Appendix](#appendix)
  - [Why the Defaults Are Safe](#why-the-defaults-are-safe)
  - [Attribute & API Sketch](#attribute--api-sketch)
  - [Open Questions](#open-questions)
  - [License](#license)

---

## Motivation

- **Tenstorrent’s execution model is persistent**: each selected core runs a long‑lived kernel and iterates over a statically assigned set of **tiles** (typically 32×32 elements), while dedicated reader/compute/writer stages move tiles between **DRAM ↔ L1** and perform compute.
- **TileLang already supports GPU‑style grid kernels** (`bx, by`) and layout hints. We propose a backend that **automatically converts grid kernels into persistent TT kernels** by generating an **outer per‑core scheduler loop** inside the compute kernel.
- Users keep writing **grid‑style** kernels. When targeting TT, the backend injects a static, per‑core loop that visits the blocks (tiles) assigned to that core. Optional **annotations** let users choose the static schedule and **TT sharding/layout**. **Sane defaults** ensure most GPU‑style kernels “just work”.

---

## Background: Persistent Kernels & Tiles on Tenstorrent

- **Static partitioning:** The host partitions the global tile space into per‑core chunks (e.g., `(start_id, count)`), then launches one persistent kernel per participating core.  
- **Tiles:** Compute operates on **tile‑formatted** tensors (e.g., 32×32). Tiles may **reside in DRAM**; reader kernels stream tiles into L1 circular buffers; compute kernels consume them; writer kernels commit results back to DRAM.
- **Program model:** A host **Program** creates kernels on a **CoreRange / CoreRangeSet**, wires circular buffers, sets runtime args, and enqueues work.

---

## Key Idea: Grid‑to‑Persistent Mapping

**Write once (GPU‑style) in TileLang:**

```python
with T.Kernel(grid_x=Nt, grid_y=Mt, threads=(...)) as (bx, by):
    compute_one_block(bx, by)   # body indexes by bx/by; no TT specifics
```

**Generated for TT (inside the compute kernel):**

```cpp
// Runtime args per core: start_id, count, grid_x (Nt), grid_y (Mt), etc.
for (uint32_t i = 0; i < count; ++i) {       // persistent outer loop
    uint32_t tid = start_id + i;             // row-major block id
    uint32_t by  = tid / grid_x;             // recover (bx, by)
    uint32_t bx  = tid % grid_x;
    compute_one_block(bx, by);               // same inner body as GPU-style kernel
}
```

This preserves the developer’s **grid mental model** while embracing TT’s **persistent, statically scheduled** execution.

---

## User‑Facing Annotations

### Static Schedule Annotations

Control how the global 2‑D block grid (`grid_x × grid_y`) is **partitioned across cores** and iterated **inside** the per‑core outer loop.

```python
T.annotate_tt_schedule(
    policy="contiguous",          # "contiguous" | "strided" | "rect"
    order="row_major",            # "row_major" | "block_linear(k)"
    rect=(by0, bx0, H, W),        # for policy="rect"
    stride=(first, step),         # for policy="strided"
    chunk_k_tiles=None,           # optional: K-panel chunking for GEMM
    qk_chunk_tiles=None,          # optional: K/V chunking for Attention
)
```

- **contiguous** (default): even, contiguous intervals `(start_id, count)` per core.
- **strided**: `tid = first + n*step` sequence per core; useful for load balancing irregular blocks.
- **rect**: assign **rectangles** of blocks to cores/groups; pairs well with reuse/multicast.
- **order**: default `row_major`, with optional `block_linear(k)` for cache/NoC locality.
- **chunk knobs**: feed into reader/compute loops (e.g., `Kt` for GEMM, `Sk` chunks for SDPA).

### Sharding & Layout Annotations

Describe how tensors are **tilized**, **sharded across cores**, and **placed** (DRAM/L1). Extends TileLang’s layout hints with **TT‑specific sharding**.

```python
T.annotate_tt_sharding({
    A: T.TTShard(axis=0,           tiles=("rows", 32), placement="DRAM",
                 order="row_major", faces="16x16"),
    B: T.TTShard(axis=1,           tiles=("cols", 32), placement="DRAM",
                 order="row_major"),
    C: T.TTShard(axis=(0, 1),      tiles=("rows","cols", 32), placement="DRAM"),
})
```

- **axis**: which dimension(s) are sharded into tiles across cores.
- **tiles**: 32×32 by default; dtype determines bytes per tile.
- **placement**: `"DRAM"` for persistent tensors; temporaries use **L1** circular buffers automatically.
- **order** / **faces**: row/col tile orders; optional faces/packing hints if needed.

### Defaults & Backward Compatibility

If **no annotations** are provided:

- **Schedule default:** `policy="contiguous"`, `order="row_major"`.  
- **Layout default:** **row‑major 32×32 DRAM tilization**; L1 CBs are synthesized around `T.copy` sites.  
- Result: **existing GPU‑style kernels run unchanged** on TT (subject to tile padding rules).

---

## End‑to‑End Examples

### GEMM (no annotations → defaults)

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

**TT mapping generated by backend:**

- Per core runtime args `(start_id, count, grid_x=Nt, grid_y=Mt, Kt, …)`.
- Compute kernel outer loop iterates `i in [0..count)` and recovers `(bx,by)` from `start_id+i`.
- Reader/Writer kernels move DRAM tiles to/from L1 CBs; compute kernel calls TT tile primitives in the K‑panel loop.

### Attention (with schedule & layout hints)

```python
# Schedule & layout annotations (optional – can be omitted)
T.annotate_tt_schedule(policy="contiguous", order="row_major", qk_chunk_tiles=16)
T.annotate_tt_sharding({
    Q: T.TTShard(axis=0, tiles=("rows",32), placement="DRAM"),
    K: T.TTShard(axis=0, tiles=("rows",32), placement="DRAM"),
    V: T.TTShard(axis=0, tiles=("rows",32), placement="DRAM"),
    O: T.TTShard(axis=0, tiles=("rows",32), placement="DRAM"),
})

@T.prim_func
def sdpa(Q, K, V, O, scale: T.float32, causal: T.int32):
    Sq_t = T.ceildiv(Sq, 32)   # Q tiles
    Sk_t = T.ceildiv(Sk, 32)   # K/V tiles
    BH   = B * H               # fused batch×heads

    # grid = (Sq_t, BH); bx = q-tile, by = (b,h)
    with T.Kernel(grid_x=Sq_t, grid_y=BH, threads=(...)) as (bx, by):
        # streaming softmax state for (by, bx)
        for k0 in range(0, Sk_t, 16):   # comes from qk_chunk_tiles
            # read Q(bx), K/V(k0 : k0+chunk)
            # scores = Q @ K^T (tile GEMMs) → update (m,l)
            # O(bx) += P @ V
        # write O(bx)
```

**TT mapping generated by backend:**

- Outer per‑core loop over `tid in [start_id, start_id+count)`, with `by = tid / grid_x`, `bx = tid % grid_x`.
- Reader streams K/V in chunks (`qk_chunk_tiles`), compute updates streaming softmax, writer stores outputs.

---

## Compiler & Codegen Plan (TVM/TileLang)

> We integrate via TVM’s **BYOC** (external codegen), keeping the TT backend cleanly modular.

### Phase 0 — MVP (GEMM, Elementwise)

1. **`GridToPersistentTT` (new pass)**  
   - **In:** TIR/TileLang PrimFunc using `T.Kernel(grid_x, grid_y)` and `bx/by`.  
   - **Out:** Function wrapped in a **per‑core outer loop** driven by the selected schedule.  
   - **Spec:**  
     - Compute `total = grid_x * grid_y`; materialize policy = contiguous/strided/rect.  
     - Replace `bx/by` with expressions of `tid` recovered inside the loop.  
     - Attach PrimFunc attrs:  
       - `tt.grid = (grid_y, grid_x)`  
       - `tt.schedule = {policy, order, rect?, stride?, chunk_k_tiles?, qk_chunk_tiles?}`  
       - `tt.runtime_args = ["start_id","count", …]`  
     - **Error cases:** missing `grid_x/grid_y`; unsupported nest shapes; negative extents.

2. **`TTShardToCoreMap` (new pass)**  
   - **In:** TT sharding/layout annotations.  
   - **Out:** Concrete **CoreRangeSet** and per‑tensor sharding metadata.  
   - **Spec:**  
     - Translate high‑level `TTShard` into `(axis, tilization, order, placement)` + core ranges.  
     - Attach `tt.core_ranges`, `tt.shards` to buffers/PrimFunc.  
     - **Error cases:** non‑tile‑multiple shapes (defer to `TilePadTT`), inconsistent placements.

3. **`TilePadTT` (new pass)**  
   - **In:** Tensors with extents not multiple of 32 on tiled axes.  
   - **Out:** Insert pad/unpad around compute or request zero‑fill tails in readers/writers.  
   - **Spec:** dtype‑aware tile bytes; optionally fuse pad into reader; mark effective shape.

4. **`MemorySpaceLowerTT` (new pass)**  
   - **In:** TIR with `T.copy` & shared/fragment allocations.  
   - **Out:** Explicit **DRAM↔L1** moves, **circular buffer** descriptors, syncs.  
   - **Spec:**  
     - Map `T.alloc_shared` → L1 CB segments; compute depths from schedule/chunk knobs.  
     - Lower copies to reader/writer enqueue ops; add attrs `tt.cb.{depth,format,bytes}`.

5. **`TensorizeTT` (new pass)**  
   - **In:** Canonical tile GEMM/epilogue patterns.  
   - **Out:** Calls to TT tile micro‑kernels (e.g., `matmul_tiles`).  
   - **Spec:** pattern match, replace with intrinsic calls; keep fallbacks if not matched.

6. **`EmitTTKernels` (codegen)**  
   - **Out:**  
     - **Compute kernel** C++ source with the generated **outer scheduler loop** + intrinsic calls.  
     - **Reader/Writer kernels** C++ sources with DRAM address math from `(bx,by)` or rectangles.  
     - **Host stub** that builds the Program, creates kernels on **CoreRange/CoreRangeSet**, allocates CBs, sets **runtime args** (`start_id`, `count`, `grid`, `Kt`/chunk), and enqueues.

7. **Runtime glue**  
   - Produce a `tvm.runtime.Module` that compiles the host stub and kernels, resolves TT‑Metalium SDK, and exposes a callable `run(...)`.  
   - CMake guards: `-DTL_TT_BACKEND=ON`, `TT_METAL_HOME` discovery; non‑TT builds remain unaffected.

### Phase 1 — SDPA, Dequant‑GEMM, Reuse/Multicast

8. **`SDPAFusionTT` (new pass)**  
   - Fuse `Q·Kᵀ → softmax → P·V` into a streaming loop over **(B×H, Q tiles)** with **K‑chunking**.  
   - Emit per‑core persistent outer loop; map `qk_chunk_tiles` into reader/compute loops.

9. **`TTMulticastReuse` (opt pass)**  
   - Where layout implies neighbor reuse (A/B in GEMM, Q or K in SDPA), introduce sender/receiver ranges and multicast paths; synthesize variant readers/writers per range.

10. **`RasterizationTT` (opt pass)**  
    - Switch `tid → (by,bx)` mapping to `block_linear(k)` or other locality‑aware orders.

### Phase 2 — Ergonomics, Safety, Diagnostics

11. **Legalize & Guards**  
    - Insert masks/tails where partial tiles are unavoidable; fall back to scalar or smaller vectors.

12. **Diagnostics**  
    - Validate shard/schedule feasibility; emit actionable errors.  
    - Dump `tt.plan.json` containing per‑core `(start_id, count)` or rectangle maps for inspection.

---

## Runtime Integration & Build

- Integrate as a **BYOC external codegen** module (e.g., `tilelang_tt`) with clean boundaries.  
- Build only when `TL_TT_BACKEND=ON` and TT SDK is discoverable.  
- Provide a **“dry‑run”** mode that emits the host/kernel sources and `tt.plan.json` without executing (useful for CI without hardware).

---

## Developer Workflow & Repository Layout

**Phase 1 (public fork):** start at `tile-ai/tilelang-tt` (or similar)

```
tilelang-tt/
├─ python/tilelang_tt/annotations.py        # annotate_tt_schedule / annotate_tt_sharding
├─ src/tt/passes/*.cc                       # GridToPersistentTT, TTShardToCoreMap, ...
├─ src/tt/codegen/*.cc                      # EmitTTKernels + host stubs
├─ include/tilelang_tt/*.h
├─ cmake/TTMetal.cmake                      # SDK discovery
├─ tests/tt/*.py                            # compile-only & dry-run tests
└─ docs/                                    # design notes, tt.plan.json examples
```

- Keep **vendor SDK deps** behind CMake options; never block other backends.
- Land **Phase 0** (GEMM) with compile‑time tests and at least one **hardware smoke test**.
- Publish **design docs** and **plans** per pass; keep PRs small and reviewable.

**Phase 2 (upstream):** open a TileLang **RFC PR** to integrate as an official backend once:
- CI is green (build‑only + optional HIL),  
- the API surface (annotations & attrs) is stable,  
- core operators (GEMM, elementwise) and at least one **attention** path are in.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Shapes not multiple of tile size | `TilePadTT` + reader/writer tails; clear diagnostics. |
| Backend drift / SDK changes | Version‑gated CMake; isolate TT APIs in one module. |
| CI without TT hardware | “Dry‑run” build that prints generated sources + `tt.plan.json`. |
| Over‑eager tensorization | Keep fallbacks; allow `--disable-tt-tensorize` for debugging. |

---

## Call for Contributions

We’re looking for collaborators in these areas:

- **Pass implementation:** `GridToPersistentTT`, `MemorySpaceLowerTT`, `TensorizeTT`.  
- **Kernel stencils:** robust **reader / compute / writer** templates for GEMM & SDPA.  
- **Sharding heuristics:** sensible defaults for **CoreRangeSet** selection per device.  
- **Testing:** correctness (NumPy/PyTorch refs), perf baselines, CI scaffold (dry‑run + optional HIL).  
- **Docs & examples:** dequant‑GEMM, Flash/MLA‑style attention with `qk_chunk_tiles`.

Please open issues/PRs in the fork and tag **`area:tt-backend`**. Include hardware/driver details where relevant.

---

## Appendix

### Why the Defaults Are Safe

- **Schedule:** `contiguous + row_major` matches the standard static split used in multi‑core matmul tutorials—each core gets a contiguous range of tile IDs.  
- **Layout:** **Row‑major 32×32 tilization in DRAM** aligns with TT’s common tile format; L1 circular buffers are synthesized automatically around copy sites.

### Attribute & API Sketch

**Python (user annotations)**

```python
# Scheduling
T.annotate_tt_schedule(policy="contiguous",
                       order="row_major",
                       rect=None,
                       stride=None,
                       chunk_k_tiles=None,
                       qk_chunk_tiles=None)

# Sharding / layout
T.annotate_tt_sharding({
    TensorA: T.TTShard(axis=0, tiles=("rows", 32), placement="DRAM"),
    TensorB: T.TTShard(axis=1, tiles=("cols", 32), placement="DRAM"),
})
```

**PrimFunc / Buffer attributes (internal)**

```text
tt.grid           = (grid_y, grid_x)
tt.schedule       = { policy, order, rect?, stride?, chunk_k_tiles?, qk_chunk_tiles? }
tt.core_ranges    = CoreRangeSet(...)
tt.shards         = { buffer_name: { axis, tiles, placement, order, faces? } }
tt.runtime_args   = ["start_id","count", ...]
tt.cb             = { name: { depth, format, l1_bytes } }
```

**`tt.plan.json` (debug dump)**

```json
{
  "grid": [Mt, Nt],
  "policy": "contiguous",
  "mapping": [
    {"core": [y,x], "start_id": 0,  "count": 128},
    {"core": [y,x], "start_id": 128,"count": 128}
  ]
}
```

### Open Questions

- Do we expose **CoreRangeSet selection** in Python, or compute it from sharding and device defaults?  
- Preferred **default CB depths** per op and dtype? (derive from chunk sizes?)  
- How soon to enable **multicast / reuse** by default for attention/GEMM rectangles?  
- Which **TT devices** and SDK versions to qualify first (e.g., Wormhole/Blackhole)?

### License

This backend will be contributed under the same license as TileLang. Vendor SDK headers/libraries remain under their respective licenses.

---

**Next steps:**  
- Create the public fork, land **Phase 0** (GEMM) with compile‑time CI + optional hardware smoke tests.  
- Iterate on annotations/spec, then open an upstream **RFC PR** to integrate as an official backend.

