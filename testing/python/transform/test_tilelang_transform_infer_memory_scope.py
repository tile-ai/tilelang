"""Tests for tl.transform.InferMemoryScope (T.auto_alloc, issue #277).

All tests are GPU-free: pass-level tests apply InferMemoryScope manually and
inspect the TIR; pipeline-level tests use tilelang.lower() which only emits
source code; the numeric check runs on the CPU ("c") target.
"""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm.target import Target

PASS_NAME = "tl.transform.InferMemoryScope"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _alloc_scopes(func) -> dict:
    """Map each sblock-allocated buffer name to its scope string."""
    scopes = {}

    def _visit(node):
        if isinstance(node, tvm.tirx.SBlock):
            for buf in node.alloc_buffers:
                scopes[buf.name] = buf.scope()

    tvm.tirx.stmt_functor.post_order_visit(func.body, _visit)
    return scopes


def _infer(func) -> dict:
    """Apply InferMemoryScope to the parsed func and return buffer scopes."""
    mod = tvm.IRModule.from_expr(func)
    mod = tilelang.transform.InferMemoryScope()(mod)
    for _, f in mod.functions.items():
        return _alloc_scopes(f)
    raise AssertionError("empty module")


# ---------------------------------------------------------------------------
# kernels under test
# ---------------------------------------------------------------------------

M, N, K = 1024, 1024, 1024
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64


def _gemm_auto():
    """The issue #277 example, written with T.auto_alloc."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((N, K), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(N // BLOCK_N, M // BLOCK_M, threads=128) as (bx, by):
            A_buf = T.auto_alloc((BLOCK_M, BLOCK_K), "float16")
            B_buf = T.auto_alloc((BLOCK_N, BLOCK_K), "float16")
            C_buf = T.auto_alloc((BLOCK_M, BLOCK_N), "float16")
            T.clear(C_buf)
            for k in T.Pipelined(K // BLOCK_K, num_stages=3):
                T.copy(A[by * BLOCK_M, k * BLOCK_K], A_buf)
                T.copy(B[bx * BLOCK_N, k * BLOCK_K], B_buf)
                T.gemm(A_buf, B_buf, C_buf, transpose_B=True)
            T.copy(C_buf, C[by * BLOCK_M, bx * BLOCK_N])

    return main


def _gemm_explicit():
    """The same kernel with hand-written scopes."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((N, K), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(N // BLOCK_N, M // BLOCK_M, threads=128) as (bx, by):
            A_buf = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
            B_buf = T.alloc_shared((BLOCK_N, BLOCK_K), "float16")
            C_buf = T.alloc_fragment((BLOCK_M, BLOCK_N), "float16")
            T.clear(C_buf)
            for k in T.Pipelined(K // BLOCK_K, num_stages=3):
                T.copy(A[by * BLOCK_M, k * BLOCK_K], A_buf)
                T.copy(B[bx * BLOCK_N, k * BLOCK_K], B_buf)
                T.gemm(A_buf, B_buf, C_buf, transpose_B=True)
            T.copy(C_buf, C[by * BLOCK_M, bx * BLOCK_N])

    return main


# ---------------------------------------------------------------------------
# pass-level rule tests
# ---------------------------------------------------------------------------


def test_gemm_scopes():
    """R1/R2 (+R3): gemm C -> local.fragment, A/B -> shared.dyn."""
    scopes = _infer(_gemm_auto())
    assert scopes["A_buf"] == "shared.dyn"
    assert scopes["B_buf"] == "shared.dyn"
    assert scopes["C_buf"] == "local.fragment"


def test_pipelined_copy_dst_is_shared():
    """R3: copy(global -> X) dst inside a num_stages loop -> shared.dyn."""

    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"), C: T.Tensor((M, K), "float16")):
        with T.Kernel(M // BLOCK_M, threads=128) as bx:
            stage = T.auto_alloc((BLOCK_M, BLOCK_K), "float16")
            for k in T.Pipelined(K // BLOCK_K, num_stages=2):
                T.copy(A[bx * BLOCK_M, k * BLOCK_K], stage)
            T.copy(stage, C[bx * BLOCK_M, 0])

    assert _infer(main)["stage"] == "shared.dyn"


def test_cumsum_operand_is_shared():
    """R4: cumsum src/dst -> shared.dyn (frontend silently routes non-fragment
    buffers to the shared-memory tl.cumsum lowering)."""

    @T.prim_func
    def main(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")):
        with T.Kernel(1, threads=128):
            src = T.auto_alloc((128, 128), "float32")
            dst = T.auto_alloc((128, 128), "float32")
            T.copy(A, src)
            T.cumsum(src, dst)
            T.copy(dst, B)

    scopes = _infer(main)
    assert scopes["src"] == "shared.dyn"
    assert scopes["dst"] == "shared.dyn"


def test_elementwise_parallel_is_fragment():
    """R6: plain accesses in T.Parallel nests with a consistent bijective
    mapping -> local.fragment."""

    @T.prim_func
    def main(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")):
        with T.Kernel(1, threads=128):
            tmp = T.auto_alloc((128, 128), "float32")
            for i, j in T.Parallel(128, 128):
                tmp[i, j] = A[i, j] + 1.0
            for i, j in T.Parallel(128, 128):
                B[i, j] = tmp[i, j]

    assert _infer(main)["tmp"] == "local.fragment"


def test_transpose_access_is_shared():
    """R8: inconsistent index mappings across parallel nests -> shared.dyn."""

    @T.prim_func
    def main(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")):
        with T.Kernel(1, threads=128):
            tmp = T.auto_alloc((128, 128), "float32")
            for i, j in T.Parallel(128, 128):
                tmp[i, j] = A[i, j]
            for i, j in T.Parallel(128, 128):
                B[j, i] = tmp[j, i]  # transposed read: mapping [j, i] != [i, j]

    assert _infer(main)["tmp"] == "shared.dyn"


def test_broadcast_write_is_shared():
    """R8: an access whose mapping is not bijective (loop var unused) must not
    become a fragment."""

    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128, 128), "float32")):
        with T.Kernel(1, threads=128):
            tmp = T.auto_alloc((128,), "float32")
            for i in T.Parallel(128):
                tmp[i] = A[i]
            for i, j in T.Parallel(128, 128):
                B[i, j] = tmp[i]  # j does not index tmp: replicated read

    assert _infer(main)["tmp"] == "shared.dyn"


def test_sequential_only_is_local():
    """R7: plain accesses outside parallel loops -> local."""

    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            tmp = T.auto_alloc((128,), "float32")
            for i in T.serial(128):
                tmp[i] = A[i] * 2.0
            for i in T.serial(128):
                B[i] = tmp[i]

    assert _infer(main)["tmp"] == "local"


def test_thread_indexed_access_is_not_local():
    """Soundness (review P1): per-thread-indexed writes in sequential code must
    not become local — each thread's private copy would have only its own slot
    initialized and cross-thread reads would silently see garbage."""

    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(1, threads=16):
            tmp = T.auto_alloc((16,), "float32")
            tx = T.get_thread_binding()
            tmp[tx] = A[tx]
            for i in T.serial(16):
                B[i] = tmp[i]

    assert _infer(main)["tmp"] == "shared.dyn"


def test_thread_conditional_write_is_not_local():
    """Same soundness class as the thread-indexed case: a write guarded by a
    per-thread condition only lands in some threads' private copies."""

    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        with T.Kernel(1, threads=16):
            tmp = T.auto_alloc((16,), "float32")
            tx = T.get_thread_binding()
            if tx == 0:
                for i in T.serial(16):
                    tmp[i] = A[i]
            for i in T.serial(16):
                B[i] = tmp[i]

    assert _infer(main)["tmp"] == "shared.dyn"


def test_dead_buffer_is_local():
    """R10: a buffer with no accesses -> local."""

    @T.prim_func
    def main(A: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            _tmp = T.auto_alloc((128,), "float32")
            for i in T.Parallel(128):
                A[i] = A[i] + 1.0

    assert _infer(main)["_tmp"] == "local"


def test_bool_shared_decision_uses_static_shared():
    """R11: bool buffers decided as shared use "shared", not "shared.dyn"
    (MergeSharedMemoryAllocations cannot merge bool)."""

    @T.prim_func
    def main(A: T.Tensor((128, 128), "bool"), B: T.Tensor((128, 128), "bool")):
        with T.Kernel(1, threads=128):
            tmp = T.auto_alloc((128, 128), "bool")
            for i, j in T.Parallel(128, 128):
                tmp[i, j] = A[i, j]
            for i, j in T.Parallel(128, 128):
                B[j, i] = tmp[j, i]  # transposed read -> shared fallback

    assert _infer(main)["tmp"] == "shared"


def test_conflicting_gemm_roles_error():
    """R9: one buffer as both gemm accumulator and A operand -> error listing
    both conflicting uses."""

    @T.prim_func
    def main(
        A2: T.Tensor((128, 32), "float16"),
        B1: T.Tensor((128, 64), "float16"),
        B2: T.Tensor((64, 32), "float16"),
        C: T.Tensor((128, 128), "float16"),
    ):
        with T.Kernel(1, threads=128):
            # M=128, K=64 for gemm1; M=128, N=64 for gemm2, so a (128, 64)
            # buffer can serve as gemm1's A operand and gemm2's accumulator.
            both = T.auto_alloc((128, 64), "float16")
            c_acc = T.auto_alloc((128, 128), "float16")
            b1 = T.alloc_shared((128, 64), "float16")
            a2 = T.alloc_shared((128, 32), "float16")
            b2 = T.alloc_shared((64, 32), "float16")
            T.copy(B1, b1)
            T.copy(A2, a2)
            T.copy(B2, b2)
            T.clear(c_acc)
            T.clear(both)
            # 'both' is the A operand here ...
            T.gemm(both, b1, c_acc, transpose_B=True)
            # ... and the accumulator here: conflicting scope requirements.
            T.gemm(a2, b2, both, transpose_B=True)
            T.copy(c_acc, C)

    mod = tvm.IRModule.from_expr(main)
    with pytest.raises(Exception, match="cannot infer memory scope for buffer 'both'"):
        tilelang.transform.InferMemoryScope()(mod)


def _residual_auto_func():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            tmp = T.auto_alloc((128,), "float32")
            for i in T.Parallel(128):
                tmp[i] = A[i]
            for i in T.Parallel(128):
                B[i] = tmp[i]

    return tvm.IRModule.from_expr(main)


def test_verify_buffer_init_rejects_residual_auto():
    """VerifyBufferInit is the single gate for unresolved auto scopes: on
    pipelines with InferMemoryScope they cannot occur, on pipelines without it
    (metal/webgpu) they must fail loudly here."""
    with pytest.raises(Exception, match="InferMemoryScope"):
        tilelang.transform.VerifyBufferInit()(_residual_auto_func())


def test_auto_check_not_gated_by_init_check_optout():
    """The auto-scope gate is a hard correctness check, not a warning: it must
    fire even when the buffer-init warning is disabled by pass config."""
    mod = _residual_auto_func()
    config = {tilelang.PassConfigKey.TL_DISABLE_BUFFER_INIT_CHECK.value: True}
    with tvm.transform.PassContext(config=config), pytest.raises(Exception, match="InferMemoryScope"):
        tilelang.transform.VerifyBufferInit()(mod)


def test_reduce_rejects_auto_scope():

    def build():
        @T.prim_func
        def main(A: T.Tensor((128, 64), "float32"), B: T.Tensor((128,), "float32")):
            with T.Kernel(1, threads=128):
                src = T.auto_alloc((128, 64), "float32")
                dst = T.alloc_fragment((128,), "float32")
                T.copy(A, src)
                T.reduce_sum(src, dst, dim=1)
                T.copy(dst, B)

        return tvm.IRModule.from_expr(main)

    with pytest.raises(ValueError, match="auto"):
        build()


def test_print_rejects_auto_scope():

    def build():
        @T.prim_func
        def main(A: T.Tensor((128,), "float32")):
            with T.Kernel(1, threads=128):
                tmp = T.auto_alloc((128,), "float32")
                for i in T.Parallel(128):
                    tmp[i] = A[i]
                T.print(tmp)

        return tvm.IRModule.from_expr(main)

    with pytest.raises(ValueError, match="auto"):
        build()


def test_verify_buffer_init_checks_inferred_buffer(capfd):
    """Pipeline order: InferMemoryScope runs before VerifyBufferInit, so the
    init check applies to buffers under their inferred scopes."""

    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            tmp = T.auto_alloc((128,), "float32")
            for i in T.Parallel(128):
                B[i] = tmp[i]  # read before anything writes tmp
            for i in T.Parallel(128):
                tmp[i] = A[i]

    mod = tvm.IRModule.from_expr(main)
    mod = tilelang.transform.InferMemoryScope()(mod)
    tilelang.transform.VerifyBufferInit()(mod)
    assert "Buffer read before initialization" in capfd.readouterr().err


# ---------------------------------------------------------------------------
# pipeline-level equivalence and codegen tests
# ---------------------------------------------------------------------------

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER.value: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED.value: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK.value: True,
}


def _lower_source(func, target: str) -> str:
    with tvm.transform.PassContext(config=_PASS_CONFIGS), Target(target):
        artifact = tilelang.lower(func, target=target)
    return artifact.kernel_source


@tilelang.testing.requires_cuda
def test_auto_gemm_matches_explicit_gemm_cuda():
    """The auto-scope kernel must lower to the same CUDA source as the
    hand-written alloc_shared/alloc_fragment version."""
    auto_src = _lower_source(_gemm_auto(), "cuda")
    explicit_src = _lower_source(_gemm_explicit(), "cuda")
    assert auto_src == explicit_src


@tilelang.testing.requires_cuda
def test_auto_gemm_codegen_form():
    """A/B land in dynamic shared memory; C is a plain local array."""
    src = _lower_source(_gemm_auto(), "cuda")
    # A/B are carved out of the dynamic shared-memory arena.
    assert "extern __shared__" in src
    assert re.search(r"A_buf\s*=\s*\(\(void\*\)\(\(char\*\)buf_dyn_shmem", src), src
    assert re.search(r"B_buf\s*=\s*\(\(void\*\)\(\(char\*\)buf_dyn_shmem", src), src
    # The accumulator is a thread-local array, not shared.
    assert re.search(r"half_t\s+C_buf\[\d+\];", src), src


@tilelang.testing.requires_cuda
def test_elementwise_fragment_full_pipeline():
    """R6's local.fragment decision must survive LayoutInference and lower to a
    per-thread local array (vectorized), not shared memory."""

    @T.prim_func
    def main(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")):
        with T.Kernel(1, threads=128):
            tmp = T.auto_alloc((128, 128), "float32")
            for i, j in T.Parallel(128, 128):
                tmp[i, j] = A[i, j] + 1.0
            for i, j in T.Parallel(128, 128):
                B[i, j] = tmp[i, j]

    src = _lower_source(main, "cuda")
    assert re.search(r"float\s+tmp\[\d+\];", src), src
    assert "tmp = ((void*)((char*)buf_dyn_shmem" not in src


# ---------------------------------------------------------------------------
# CPU end-to-end numeric validation
# ---------------------------------------------------------------------------


def test_cpu_elementwise_numeric():
    """R7 local on CPU: serial-only auto buffer kernel matches torch."""

    @T.prim_func
    def main(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        with T.Kernel(1):
            tmp = T.auto_alloc((256,), "float32")
            for i in T.serial(256):
                tmp[i] = A[i] * 2.0
            for i in T.serial(256):
                B[i] = tmp[i] + 1.0

    compiled = tilelang.compile(main, target="c", out_idx=-1, execution_backend="cython")
    a = torch.randn(256)
    b = compiled(a)
    torch.testing.assert_close(b, a * 2.0 + 1.0)


def test_cpu_gemm_numeric():
    """Auto buffers in a small tiled gemm (copy staging + scalar compute)
    match torch on the CPU target."""
    m = n = k = 128
    bm, bn, bk = 32, 32, 32

    @T.prim_func
    def matmul(
        A: T.Tensor((m, k), "float32"),
        B: T.Tensor((k, n), "float32"),
        C: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(n // bn, m // bm) as (bx, by):
            A_local = T.auto_alloc((bm, bk), "float32")
            B_local = T.auto_alloc((bk, bn), "float32")
            C_local = T.auto_alloc((bm, bn), "float32")
            T.clear(C_local)
            for ko in T.serial(k // bk):
                T.copy(A[by * bm, ko * bk], A_local)
                T.copy(B[ko * bk, bx * bn], B_local)
                for i, j, kk in T.grid(bm, bn, bk):
                    C_local[i, j] += A_local[i, kk] * B_local[kk, j]
            T.copy(C_local, C[by * bm, bx * bn])

    compiled = tilelang.compile(matmul, target="c", out_idx=-1, execution_backend="cython")
    a = torch.randn(m, k)
    b = torch.randn(k, n)
    c = compiled(a, b)
    torch.testing.assert_close(c, a @ b, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    tilelang.testing.main()
