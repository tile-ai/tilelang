"""Regression test for the sm_120 (Blackwell consumer) TMA shared-memory
alignment bug.

Pre-fix `SharedMemoryAlignmentPlanner` (`merge_shared_memory_allocations.cc`)
gated the 1024-byte alignment of TMA-touched smem buffers on
`TargetIsHopper(target)` (arch in `[90, 100)`). For sm_100 / sm_120 (also
TMA-capable: see `cp.async.bulk.tensor.{1..5}d.shared::cta.global.*` PTX
support gated by `TargetHasBulkCopy(target)` in `src/target/utils.cc`)
the planner fell back to the global default (16 bytes) and TMA destinations
landed at 16-byte-aligned offsets. `cp.async.bulk.tensor.*` requires the
destination smem pointer to be 128-byte aligned, so on sm_100/sm_120 this
caused `CUDA_ERROR_MISALIGNED_ADDRESS` whenever the dynamic-smem arena
started with a small (e.g. 4-byte float) buffer that pushed subsequent TMA
buffers off the 128-byte boundary.

Post-fix the predicate is `TargetHasBulkCopy(target)`, which is true for
arch >= 90 (sm_90, sm_100, sm_103, sm_120 and any future TMA-capable
target) — the correct set, since TMA support and TMA's alignment
requirements are coextensive.

The test has two parts:

1. **Generated TIR check** (host-independent). Lowers the buggy-arena
   kernel for explicit `cuda -arch=sm_{90,100,120}` targets and asserts
   every `tl.tma_load` destination `tir.tvm_access_ptr` byte offset is
   provably 128-byte aligned. Runs on any host.

2. **Runtime check** on the host GPU. Compiles + executes the same kernel
   for the host arch and asserts it doesn't raise
   `CUDA_ERROR_MISALIGNED_ADDRESS`. Pre-fix on sm_120 this crashes; on
   sm_90 (Hopper) the kernel happens to land buffers correctly even
   pre-fix. Post-fix it passes on every TMA-capable arch.
"""

import torch

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang import tvm
from tilelang.engine.lower import lower as tilelang_lower


def _make_buggy_arena_kernel(M, N, K, BM, BN, BK):
    """2-stage pipelined bf16 GEMM with a tiny T.float32 shared scalar
    whose lifetime *overlaps* the TMA-loaded tiles. The scalar is
    written before the loop, accumulated inside it, and read after.
    That overlapping lifetime forces the smem-allocation planner to
    keep the scalar alive in the merged dynamic-smem arena alongside
    the TMA destinations — a shorter-lived scalar (e.g. one that is
    only used before the loop) gets reordered to a higher offset and
    masks the bug.

    Modelled on the structure of the fla
    `chunk_bwd_dqkwg_tilelang` kernel (small dg-last accumulator that
    is updated *inside* the V-loop alongside the TMA-loaded
    `s_v` / `s_do` / `s_h` / `s_dh` tiles, then used in the K-loop
    after).
    """

    @T.prim_func
    def gemm(
        a: T.Tensor((M, K), "bfloat16"),
        b: T.Tensor((K, N), "bfloat16"),
        c: T.Tensor((M, N), "bfloat16"),
        c_scalar: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=128) as (bx, by):
            # Small scalar with a lifetime that spans the whole pipelined
            # loop — it is initialised before the loop, updated inside
            # the loop, and used after the loop.
            s_acc_scalar = T.alloc_shared((1,), "float32")
            for _i in T.Parallel(1):
                s_acc_scalar[0] = 0.0
            T.sync_threads()

            s_a = T.alloc_shared((BM, BK), "bfloat16")
            s_b = T.alloc_shared((BK, BN), "bfloat16")
            c_local = T.alloc_fragment((BM, BN), "float32")
            T.clear(c_local)

            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                T.copy(a[bx * BM : (bx + 1) * BM, k * BK : (k + 1) * BK], s_a)
                T.copy(b[k * BK : (k + 1) * BK, by * BN : (by + 1) * BN], s_b)
                T.gemm(s_a, s_b, c_local)
                # Touch the scalar *inside* the loop so its lifetime
                # overlaps the TMA-loaded `s_a` / `s_b` tiles. Without
                # this overlap the planner reorders the scalar past the
                # TMA buffers and the misalignment goes undetected.
                # Read a single element of an already-TMA-loaded buffer
                # so we don't conflict with `c_local`'s gemm layout.
                for _i in T.Parallel(1):
                    s_acc_scalar[0] = s_acc_scalar[0] + T.cast(s_a[0, 0], "float32")

            T.copy(c_local, c[bx * BM : (bx + 1) * BM, by * BN : (by + 1) * BN])
            # Use s_acc_scalar after the loop so it can't be dead-code
            # eliminated.
            for _i in T.Parallel(1):
                if bx == 0 and by == 0:
                    c_scalar[0] = s_acc_scalar[0]

    return gemm


# --------------------------------------------------------------------------
# Generated TIR check (host-independent)
# --------------------------------------------------------------------------


def _lower_for(arch: str):
    M, N, K = 128, 128, 256
    BM, BN, BK = 64, 64, 32
    target = tvm.target.Target(f"cuda -arch={arch}")
    tilelang.disable_cache()
    try:
        with target:
            artifact = tilelang_lower(
                _make_buggy_arena_kernel(M, N, K, BM, BN, BK),
                target=target,
            )
    finally:
        tilelang.enable_cache()
    return artifact.device_mod


def _is_op_call(node, op_name: str) -> bool:
    return isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == op_name


def _collect_tma_loads(mod):
    loads = []

    def _visit(node):
        if _is_op_call(node, "tl.tma_load"):
            loads.append(node)

    for _, func in mod.functions.items():
        if isinstance(func, tvm.tir.PrimFunc):
            tvm.tir.stmt_functor.post_order_visit(func.body, _visit)
    return loads


def _byte_offset(access_ptr):
    dtype = access_ptr.args[0].dtype
    return access_ptr.args[2] * dtype.bytes * dtype.lanes


def _assert_tma_destinations_128_aligned(arch: str, mod):
    loads = _collect_tma_loads(mod)
    assert loads, f"[{arch}] expected at least one tl.tma_load in generated TIR — the kernel layout should produce TMA loads"

    analyzer = tvm.arith.Analyzer()
    for load in loads:
        assert len(load.args) >= 3, f"[{arch}] malformed tl.tma_load call: {load}"
        access_ptr = load.args[2]
        assert _is_op_call(access_ptr, "tir.tvm_access_ptr"), (
            f"[{arch}] expected tl.tma_load destination argument to be tir.tvm_access_ptr, got: {access_ptr}"
        )
        byte_offset = analyzer.simplify(_byte_offset(access_ptr))
        assert analyzer.can_prove(byte_offset % 128 == 0), (
            f"[{arch}] TMA load destination byte offset is not provably 128-byte aligned: {byte_offset}. access_ptr={access_ptr}"
        )


@tilelang.testing.requires_cuda
def test_tma_smem_alignment_codegen_sm90_hopper():
    """sm_90 (Hopper) — covered by the original `TargetIsHopper` predicate,
    pinned here as a regression test against re-narrowing."""
    mod = _lower_for("sm_90")
    _assert_tma_destinations_128_aligned("sm_90", mod)


@tilelang.testing.requires_cuda
def test_tma_smem_alignment_codegen_sm100_blackwell_dc():
    """sm_100 (Blackwell DC) — TMA-capable. Pre-fix this fell back to
    16-byte alignment because `TargetIsHopper` is true only for arch
    `[90, 100)`."""
    mod = _lower_for("sm_100")
    _assert_tma_destinations_128_aligned("sm_100", mod)


@tilelang.testing.requires_cuda
def test_tma_smem_alignment_codegen_sm120_blackwell_consumer():
    """sm_120 (Blackwell consumer / RTX 5090) — TMA-capable. Pre-fix this
    fell back to 16-byte alignment and triggered
    `CUDA_ERROR_MISALIGNED_ADDRESS` at runtime on real Blackwell consumer
    silicon."""
    mod = _lower_for("sm_120")
    _assert_tma_destinations_128_aligned("sm_120", mod)


# --------------------------------------------------------------------------
# Runtime check on host GPU
# --------------------------------------------------------------------------


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tma_smem_alignment_runtime_hostarch():
    """Compile + execute the buggy-arena GEMM on the host's TMA-capable
    GPU. Pre-fix on sm_100 / sm_120 this raises
    `CUDA_ERROR_MISALIGNED_ADDRESS` on the first TMA load. On sm_90
    (Hopper) the same kernel happens to land buffers at acceptable
    offsets and runs even pre-fix. This is intentionally hardware-backed;
    the compile-only regression coverage is the generated-TIR check above."""
    M, N, K = 128, 128, 256
    BM, BN, BK = 64, 64, 32
    tilelang.disable_cache()
    try:
        kernel = tilelang.compile(
            _make_buggy_arena_kernel(M, N, K, BM, BN, BK),
            target="cuda",
        )
    finally:
        tilelang.enable_cache()
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    c_scalar = torch.empty(1, device="cuda", dtype=torch.float32)
    kernel(a, b, c, c_scalar)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    torch.testing.assert_close(c, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    tilelang.testing.main()
