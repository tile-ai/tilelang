# Design: tcgen05.st (Register → TMEM Store) + gemm_ts End-to-End

## 1. Motivation

### Why TS exists

`tcgen05.mma` has 4 operand sourcing variants:

| Variant | A source      | B source     | Typical use case            |
|---------|---------------|-------------|-----------------------------|
| SS      | Shared Memory | Shared Memory | Standard GEMM             |
| RS      | Register      | Shared Memory | (less common)              |
| TS      | **Tensor Memory** | Shared Memory | **Chained GEMM (FMHA, MLA)** |
| SR      | Shared Memory | Register      | (less common)              |

TS eliminates unnecessary TMEM → SMEM round-trips in multi-stage computation. The
canonical use case is **Flash Attention** on Blackwell:

```
Stage 1 (SS):  S = Q × K^T           →  float32 accumulator in TMEM
Stage 2:       tcgen05.ld  S → regs   →  softmax(S)  →  convert float32 → bf16
Stage 3:       tcgen05.st  P → TMEM   →  bf16 probabilities stored in TMEM
Stage 4 (TS):  O = P × V             →  bf16 from TMEM × bf16 from SMEM → float32
```

Without TS, Stage 4 would need `P` copied from TMEM → registers → SMEM → SS MMA,
wasting ~50% memory bandwidth and SMEM capacity.

### What's missing in TileLang

| Component                            | Status                         |
|--------------------------------------|--------------------------------|
| `tcgen05.mma` TS codegen             | ✅ Implemented (this branch)   |
| `tcgen05.ld` (TMEM → Register)       | ✅ Working                     |
| `CopyInst::kTMemStore` enum          | ✅ Exists                      |
| `CheckTMemStore()` detection         | ✅ Exists                      |
| `fence_view_async_tmem_store()` PTX  | ✅ Exists in `tcgen_05.h`      |
| **`tcgen05.st` PTX wrappers**        | ❌ No `tcgen_05_st.h`          |
| **`LowerTmemCopy()` store path**     | ❌ Blocked by `ICHECK(!is_st)` |
| **`InferLayout` for kTMemStore**     | ❌ Blocked by `ICHECK`         |
| **Layout compatibility (st ↔ MMA)**  | ❌ Not addressed                |

## 2. Architecture Overview

### tcgen05.ld pipeline (existing, as reference)

```
User DSL:       T.copy(C_tmem, C_local)           # TMEM → register
                         │
InferLayout:    kTMemLoad detected
                src (TMEM) layout known → infer dst (register) layout
                Uses Tcgen05Meta (getTcgen05Meta_32dp32b/64b/128b/256b)
                         │
Lower:          LowerTmemCopy() → emit call_extern:
                "tl::tcgen05_ld_32dp32bNx<N, pack16>(tmem_addr, col_offset, dst_ptr)"
                         │
LowerSharedTmem: Translate BufferLoad(tmem_buf, [row, col])
                → BufferLoad(tmem_addr_buf, [0]) + (phy_row << 16 | phy_col)
                         │
Codegen:        C++ template in copy_sm100.h calls tcgen_05_ld.h
                → inline PTX: tcgen05.ld.sync.aligned.{shape}.xN.b32
```

### tcgen05.st pipeline (to implement)

```
User DSL:       T.copy(A_local, A_tmem)            # register → TMEM
                         │
InferLayout:    kTMemStore detected
                dst (TMEM) layout known → infer src (register) layout
                Uses same Tcgen05Meta (layout is symmetric)
                         │
Lower:          LowerTmemCopy() → emit call_extern:
                "tl::tcgen05_st_32dp32bNx<N, unpack16>(tmem_addr, col_offset, src_ptr)"
                         │
LowerSharedTmem: Same translation as ld (tmem_buf → physical address)
                         │
Codegen:        C++ template in copy_sm100.h calls tcgen_05_st.h
                → inline PTX: tcgen05.st.sync.aligned.{shape}.xN.b32
                followed by:  tcgen05.wait::st.sync.aligned
```

## 3. Detailed Implementation Plan

### Step 1: Create `tcgen_05_st.h` — PTX Store Wrappers

**File:** `src/tl_templates/cuda/tcgen_05_st.h`

Mirror of `tcgen_05_ld.h`, with reversed operand direction.

**PTX instruction format:**
```
// Load:  tcgen05.ld.sync.aligned.{shape}.xN[.pack::16b].b32   {dst}, [tmem_addr]
// Store: tcgen05.st.sync.aligned.{shape}.xN[.unpack::16b].b32 [tmem_addr], {src}
```

Classes to implement (mirror of ld):
```
tmem_st_32dp32bNx<Unpack16>   — 32 data path, 32-bit pattern
tmem_st_16dp64bNx<Unpack16>   — 16 data path, 64-bit pattern
tmem_st_16dp128bNx<Unpack16>  — 16 data path, 128-bit pattern
tmem_st_16dp256bNx<Unpack16>  — 16 data path, 256-bit pattern
tmem_st_32dp64bNx<Unpack16>   — 2×16dp64b (composite)
tmem_st_32dp128bNx<Unpack16>  — 2×16dp128b (composite)
tmem_st_32dp256bNx<Unpack16>  — 2×16dp256b (composite)
```

Each supports `xN` where N ∈ {1, 2, 4, 8, 16, 32, 64, 128}.

Example PTX for 32x32b.x1:
```cpp
template <> class tmem_st_32dp32bNx<false> {
public:
  template <int N>
  static TL_DEVICE void copy(uint32_t const &dst_addr, uint32_t *src_ptr) {
    if constexpr (N == 1) {
      asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32"
                   "[%0],"
                   "{%1};\n"
                   : : "r"(dst_addr), "r"(src_ptr[0]));
    }
    // ... N = 2, 4, 8, ... 128
  }
};
```

### Step 2: Add convenience functions in `copy_sm100.h`

**File:** `src/tl_templates/cuda/copy_sm100.h`

Mirror of `tcgen05_ld_*` functions:

```cpp
template <int N, bool unpack16, typename src_t>
__device__ __forceinline__ void
tcgen05_st_32dp32bNx(uint32_t const &tmem_start_col,
                     uint32_t const &tmem_col_offset, src_t *src_ptr) {
  tcgen05_st_core<tl::tmem_st_32dp32bNx<unpack16>, 7, N>(
      tmem_start_col + tmem_col_offset, src_ptr);
  tl::fence_view_async_tmem_store();  // <-- use existing store fence
}
```

Repeat for 32dp64b, 32dp128b, 32dp256b.

### Step 3: Add store metadata in `tcgen05_layout.cc`

**File:** `src/layout/tcgen05_layout.cc`

The data path layout for store is symmetric to load. Add corresponding store
meta factories:

```cpp
Tcgen05Meta getTcgen05MetaSt_32dp32b() {
  // Same fragment layout as load, different intrinsics_name
  constexpr int INST_WIDTH = 1;
  IterVar inst_row = make_itervar("row", 128);
  IterVar inst_col = make_itervar("col", INST_WIDTH);
  return Tcgen05Meta{"tl::tcgen05_st_32dp32bNx",
                     Fragment({inst_row, inst_col}, {inst_col}, {inst_row},
                              make_itervar("rep", Range(0, 1))),
                     INST_WIDTH};
}
// ... same for 64b, 128b, 256b
```

Alternatively, we can parameterize the existing `getTcgen05Meta_*` with a
direction argument to avoid code duplication:

```cpp
Tcgen05Meta getTcgen05Meta_32dp32b(bool is_store = false);
```

### Step 4: Implement `LowerTmemCopy()` store path

**File:** `src/op/copy.cc`, function `LowerTmemCopy()`

Remove the `ICHECK(!is_st)` guard and implement the store lowering. The logic
mirrors the load path but with reversed src/dst roles:

```cpp
// Current (lines 982-984):
ICHECK(!is_st) << "Copy from register to tensor memory is not supported yet";

// Replace with:
if (is_st) {
  // For store: src is register (fragment), dst is TMEM
  // Extract TMEM layout from dst, register layout from src
  Layout dst_layout = T.layout_map[dst];          // TMEM layout
  Fragment src_layout = Downcast<Fragment>(T.layout_map[src]);  // register layout

  // TMEM physical coordinate analysis (same as ld but on dst side)
  Array<PrimExpr> logical_indices = MakeIndices(loop_vars, 1);  // dst indices
  Array<PrimExpr> phy_indices = dst_layout->Forward(logical_indices);
  // ... (same bounds analysis) ...

  // Try tcgen05.st instruction variants (reuse same Tcgen05Meta approach)
  auto try_tcgen05_st = [&](Tcgen05Meta meta) {
    // Same logic as ld, but:
    // - args[0] = intrinsics name with "tcgen05_st_" prefix
    // - args[1] = TMEM dst BufferLoad (→ physical address)
    // - args[2] = col_offset
    // - args[3] = src register access_ptr (read direction)
    // - Use dst_needs_unpack instead of src_needs_pack
  };
}
```

Key differences from load:
- `src_needs_pack` → `dst_needs_unpack` (16-bit data uses `.unpack::16b`)
- `MakeIndices(loop_vars, 0)` for src → `MakeIndices(loop_vars, 1)` for dst (TMEM side)
- Buffer access directions reversed

### Step 5: Implement `InferLayout` for kTMemStore

**File:** `src/op/copy.cc`, function `CopyNode::InferLayout()`

Current code (lines 312-318) blocks store:
```cpp
if (copy_inst == CopyInst::kTMemLoad || copy_inst == CopyInst::kTMemStore) {
    ICHECK(copy_inst == CopyInst::kTMemLoad) << "Only support... currently";
```

Replace with symmetric handling:
```cpp
if (copy_inst == CopyInst::kTMemStore) {
    // dst (TMEM) layout known → infer src (register) layout
    if (!T.layout_map.count(src) && T.layout_map.count(dst)) {
        Layout dst_layout = T.layout_map[dst];
        // Derive register fragment layout from TMEM layout + tcgen05.st data path
        // (symmetric to ld: same expandTcgen05Layout, reversed src/dst)
    }
}
```

### Step 6: Handle `lower_shared_tmem.cc` for store direction

**File:** `src/transform/lower_shared_tmem.cc`

Current `VisitStmt_(BufferStoreNode*)` has:
```cpp
ICHECK(buffer.scope() != "shared.tmem")
    << "We should never directly store data into tmem!";
```

This assertion is correct for direct stores (which go through `call_extern`),
but we need to make sure the `call_extern` for `tcgen05_st_*` correctly
receives the translated TMEM address.

The `BufferLoad` translation in `VisitExpr_` already handles TMEM address
translation for the tmem argument. Since `tcgen05_st_*` receives the TMEM
address via `BufferLoad` (same as `tcgen05_ld_*`), **no changes needed** in
this file.

## 4. Layout Compatibility: tcgen05.st ↔ MMA TS

### The Problem

When using TS MMA, data flow is:
```
Register → tcgen05.st → TMEM → tcgen05.mma TS
```

The tcgen05.st data path layout determines HOW register data maps to TMEM columns.
The MMA TS data path layout determines HOW TMEM columns map to matrix elements.
**These two layouts must be consistent.**

### CUTLASS's Approach

In CUTLASS FMHA, they use `to_tiled_mma_sm100_ts()` to create a TS MMA that's
compatible with the `SM100_TMEM_STORE_32dp32b` store layout. The key insight is
that both tcgen05.st and MMA TS use the same TMEM addressing space — the hardware
data path layouts are designed to be compatible.

### Our Approach

For the first implementation, we use the **simplest data path (32dp32b)**:
- Each thread writes one 32-bit value per TMEM column
- For bf16 with `.unpack::16b`, one 32-bit value → two bf16 elements in two columns
- The MMA TS instruction reads TMEM columns in its own data path layout

The TMEM layout we assign to the input buffer must satisfy both:
1. **tcgen05.st**: the register layout must produce values in the order that
   tcgen05.st expects
2. **MMA TS**: the TMEM columns must contain matrix elements in the order
   that the MMA instruction expects

**Approach for this PR:**
- Use the existing `make_mma_load_layout()` (in the TS macro generator) to
  define the TMEM layout based on the MMA's A input data path
- Derive the register layout from this TMEM layout using the tcgen05.st
  data path mapping
- Verify layout compatibility during InferLayout

If the layouts are incompatible with simple data paths, we may need to
add a register-level permutation step. This is a known complexity in CUTLASS
(handled by `TiledCopy` + layout algebra).

## 5. End-to-End Example

### Simple TS GEMM (for correctness testing)

```python
@T.prim_func
def main(A: T.Tensor((M, K), T.bfloat16),
         B: T.Tensor((N, K), T.bfloat16),
         C: T.Tensor((M, N), T.bfloat16)):
    with T.Kernel(...) as (...):
        A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
        A_local  = T.alloc_fragment((block_M, block_K), T.bfloat16)
        A_tmem   = T.alloc_tmem([block_M, block_K], T.bfloat16)

        B_shared = T.alloc_shared((block_N, block_K), T.bfloat16)
        C_tmem   = T.alloc_tmem([block_M, block_N], T.float32)
        mbar     = T.alloc_barrier(1)

        # Load A into TMEM
        T.copy(A[...], A_shared)        # global → shared
        T.copy(A_shared, A_local)       # shared → register
        T.copy(A_local, A_tmem)         # register → TMEM (NEW: tcgen05.st)

        # Load B into shared
        T.copy(B[...], B_shared)

        # TS MMA: A from TMEM, B from shared
        T.gemm(A_tmem, B_shared, C_tmem, mbar=mbar, ...)
        T.mbarrier_wait_parity(mbar, 0)

        # Copy result out
        T.copy(C_tmem, C_local)
        ...
```

### FMHA-style chained GEMM (stretch goal)

```python
# Stage 1 (SS): S = Q × K^T → float32 in TMEM
# Stage 2: S → registers → softmax → convert → bf16 P → TMEM (tcgen05.st)
# Stage 3 (TS): O = P × V → float32 in TMEM
```

## 6. File Change Summary

| File                                      | Change type | Description                                |
|-------------------------------------------|-------------|--------------------------------------------|
| `src/tl_templates/cuda/tcgen_05_st.h`     | **New**     | PTX wrappers for tcgen05.st                |
| `src/tl_templates/cuda/copy_sm100.h`      | Modify      | Add `tcgen05_st_*` convenience functions   |
| `src/layout/tcgen05_layout.h`             | Modify      | Declare store meta factories               |
| `src/layout/tcgen05_layout.cc`            | Modify      | Implement store meta (or parameterize ld)  |
| `src/op/copy.cc` (`InferLayout`)          | Modify      | Handle `kTMemStore` layout inference       |
| `src/op/copy.cc` (`LowerTmemCopy`)        | Modify      | Implement `is_st` lowering path            |
| `examples/gemm_sm100/gemm_tcgen5mma_ts.py`| Modify      | Working example with correctness check     |

**Files NOT changed:**
- `src/transform/lower_shared_tmem.cc` — TMEM address translation already handles both directions
- `src/op/copy.h` — `CopyInst::kTMemStore` and `CheckTMemStore()` already exist
- `src/tl_templates/cuda/tcgen_05.h` — `fence_view_async_tmem_store()` already exists

## 7. Testing Strategy

1. **Unit test**: `T.copy(fragment, tmem)` compiles and generates `tcgen05.st` PTX
2. **Integration test**: Simple TS GEMM (A loaded via tcgen05.st, B from SMEM)
   - Compare kernel output with CPU reference: `C = A @ B.T`
   - Use M=N=K=128, bf16 input, float32 accumulator
3. **Regression**: Existing SS GEMM tests still pass

## 8. Risks and Open Questions

1. **Layout compatibility**: The tcgen05.st data path layout may not be directly
   compatible with the MMA TS input data path layout. If so, we need a
   register-level permutation between store and MMA, which adds complexity.
   → Mitigation: Start with 32dp32b (simplest layout), verify on hardware.

2. **TMEM allocation for non-accumulator buffers**: Current `alloc_tmem` assumes
   float32 accumulators. For bf16 input via tcgen05.st, TMEM column calculation
   differs (2 bf16 per 32-bit column with `.unpack::16b`).
   → Need to check `lower_shared_tmem.cc` column allocation math.

3. **Register layout inference direction**: For load, TMEM layout → register layout.
   For store, register layout → TMEM layout (or vice versa). Need to verify that
   the InferLayout direction is correct.

## 9. Implementation Order

1. `tcgen_05_st.h` (low risk, pure PTX wrappers)
2. `copy_sm100.h` convenience functions (low risk)
3. `tcgen05_layout.cc` store metadata (low risk)
4. `copy.cc` `LowerTmemCopy` store path (medium risk)
5. `copy.cc` `InferLayout` store path (medium risk, layout matching)
6. Example + correctness test (high integration risk)

Each step can be validated independently before proceeding to the next.
