"""llvm-target CPU parallelization (``tl.cpu_parallel``) tests.

The CPU pipeline is shared by the ``c`` and ``llvm`` targets; with
``tl.cpu_parallel`` enabled the llvm side marks only the first non-unit
grid dim as kParallel, which TVM's LLVM codegen lowers to
``TVMBackendParallelLaunch`` (its own thread pool — no OpenMP flags are
injected on this target). These tests lock in correctness of that path.
"""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing
import tvm
from tilelang.transform import PassConfigKey
from tvm import tirx

M = N = K = 256
BLOCK_M = BLOCK_N = 64
BLOCK_K = 32


@T.prim_func
def gemm(
    A: T.Tensor((M, K), dtype="float32"),
    B: T.Tensor((K, N), dtype="float32"),
    C: T.Tensor((M, N), dtype="float32"),
):
    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=1) as (bx, by):
        A_shared = T.alloc_buffer((BLOCK_M, BLOCK_K), dtype="float32", scope="shared")
        B_shared = T.alloc_buffer((BLOCK_K, BLOCK_N), dtype="float32", scope="shared")
        C_local = T.alloc_buffer((BLOCK_M, BLOCK_N), dtype="float32", scope="local")
        T.clear(C_local)
        for ko in T.Pipelined(K // BLOCK_K, num_stages=1):
            T.copy(A[by * BLOCK_M, ko * BLOCK_K], A_shared)
            T.copy(B[ko * BLOCK_K, bx * BLOCK_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])


def _compile(pass_configs):
    return tilelang.compile(
        gemm,
        target="llvm",
        out_idx=-1,
        execution_backend="tvm_ffi",
        pass_configs=pass_configs,
    )


def _count_parallel_loops(mod):
    count = 0

    def visit(node):
        nonlocal count
        if isinstance(node, tirx.For) and node.kind == tirx.ForKind.PARALLEL:
            count += 1

    for _, f in mod.functions.items():
        tirx.stmt_functor.post_order_visit(f.body, visit)
    return count


def _lowered_parallel_loop_count(pass_config):
    """kParallel loops in the lowered module (host + device) — proof the
    grid nest was actually parallelized, not just numerically correct."""
    with tvm.target.Target("llvm"), tvm.transform.PassContext(config=pass_config):
        artifact = tilelang.lower(gemm, target="llvm")
    return _count_parallel_loops(artifact.host_mod) + _count_parallel_loops(artifact.device_mod)


@tilelang.testing.requires_llvm
def test_llvm_cpu_parallel_gemm_correctness():
    torch.manual_seed(0)
    assert _lowered_parallel_loop_count({"tl.cpu_parallel": True}) >= 1
    kernel = _compile({PassConfigKey.TL_CPU_PARALLEL: True})
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    torch.testing.assert_close(kernel(A, B), A @ B, rtol=1e-3, atol=1e-3)


@tilelang.testing.requires_llvm
def test_llvm_cpu_parallel_disabled_by_default():
    torch.manual_seed(0)
    assert _lowered_parallel_loop_count({}) == 0
    kernel = _compile(None)
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    torch.testing.assert_close(kernel(A, B), A @ B, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    tilelang.testing.main()
