"""Minimal repro: int64-strided vectorized load that codegen cannot emit.

Distilled from a `score_frag[i, j] = state_cache[slot, cache, ..., j]` load where
`state_cache` is a StridedTensor with a *dynamic int64* leading stride.

The essence of the bug
----------------------
`state_cache`'s leading stride `stride0` is a dynamic int64 value, so every
flattened element offset into it is an int64 expression. Each thread owns a
contiguous VEC-element chunk of the innermost axis, so the unit-stride loop is
vectorized to a lanes>1 (int64-indexed) load. `T.assume(stride0 % (VEC*DIM) == 0)`
makes the alignment provable on the dynamic shape, so the vectorizer commits to
VEC instead of falling back to lanes=1.

At codegen, `CodeGenC::VisitExpr_(BufferLoadNode)` re-proves alignment via
`arith::Analyzer().modular_set(ramp->base)` with a *fresh empty* analyzer. The
int64 modular fact proved by `T.assume` upstream is not available there, so the
proof fails and codegen drops into the scalar element-wise fallback. That
fallback materializes the whole int64 ramp index as an `int64xN` SSA temp
(`SSAGetID(PrintExpr(index), index.dtype())`), and `PrintType(int64xN)` then
hits `LOG(FATAL) "Cannot convert type int64x2 ... to Ascend type"`.

That codegen fallback FATAL is the bug this repro pins.

Run
---
    source haienv tilelang
    sudo -E env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        ASCEND_NPU_ARCH=dav-3510 TILELANG_DISABLE_CACHE=1 \
        python examples/ascend/example_int64_stride_vectorize_load.py
"""

import tilelang
import tilelang.ascend.language as T

DIM = 128
VEC = 2
# Fewer threads than DIM so each thread owns a contiguous VEC-element chunk of
# the innermost axis -> a unit-stride ramp the vectorizer can widen to floatN.
THREADS = DIM // VEC  # 64


def repro_kernel() -> object:
    num_slots = T.dynamic("num_slots")
    cache_size = T.dynamic("cache_size")
    # Dynamic int64 leading stride -> every offset into state_cache is int64.
    stride0 = T.dynamic("stride0", dtype="int64")
    total_c = T.dynamic("total_c")

    @T.prim_func
    def _min(
        state_cache: T.StridedTensor(
            shape=[num_slots, cache_size, DIM],
            strides=[stride0, DIM, 1],
            dtype="float32",
        ),
        slot_idx: T.Tensor([1], "int32"),
        out: T.Tensor([total_c, DIM], "float32"),
    ) -> None:
        T.assume(stride0 % (VEC * DIM) == 0)
        with T.Kernel(1) as _, T.SimtVF(threads=THREADS):
            frag = T.alloc_fragment([DIM], "float32")
            s = T.int32(slot_idx[0])
            T.assume(0 <= s < num_slots)
            for j in T.Parallel(DIM):
                frag[j] = state_cache[s, 0, j]
            for j in T.Parallel(DIM):
                out[0, j] = frag[j]

    return _min


if __name__ == "__main__":
    import torch

    device = torch.device("npu")

    num_slots, cache_size, total_c = 2, 32, 4
    state_cache = torch.randn(num_slots, cache_size, DIM, dtype=torch.float32, device=device)
    slot_idx = torch.zeros(1, dtype=torch.int32, device=device)
    out = torch.zeros(total_c, DIM, dtype=torch.float32, device=device)

    kernel = tilelang.compile(repro_kernel())
    kernel(state_cache, slot_idx, out)
    torch.npu.synchronize()
    print("PASS")
