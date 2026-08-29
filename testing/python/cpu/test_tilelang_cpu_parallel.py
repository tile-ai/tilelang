"""OpenMP CPU parallelization (``tl.cpu_parallel``) tests.

Covers the opt-in contract of the CPU OpenMP lowering:

- enabled: grid loops become ``#pragma omp parallel for [collapse(n)]`` in
  the generated C source, function-scope buffers are sunk into the parallel
  region (per-worker private copies), and results are exact;
- disabled (default): no OpenMP pragma in the generated source and results
  are unchanged (bit-identical serial lowering);
- ``tl.cpu_parallel_min_trip`` keeps small grids serial.
"""

import sys

import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey

import tvm
from tvm import tirx
from tvm.target import Target

M = N = K = 512
BLOCK_M = BLOCK_N = 128
BLOCK_K = 32


def make_gemm(M, N, K, BM, BN, BK):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype="float32"),
        B: T.Tensor((K, N), dtype="float32"),
        C: T.Tensor((M, N), dtype="float32"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=1) as (bx, by):
            A_shared = T.alloc_buffer((BM, BK), dtype="float32", scope="shared")
            B_shared = T.alloc_buffer((BK, BN), dtype="float32", scope="shared")
            C_local = T.alloc_buffer((BM, BN), dtype="float32", scope="local")
            T.clear(C_local)
            for ko in T.Pipelined(K // BK, num_stages=1):
                T.copy(A[by * BM, ko * BK], A_shared)
                T.copy(B[ko * BK, bx * BN], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * BM, bx * BN])

    return gemm


def _compile(pass_configs):
    return tilelang.compile(
        make_gemm(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K),
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs=pass_configs,
    )


def _run(kernel, A, B):
    return kernel(A, B)


def test_cpu_parallel_gemm_correctness():
    torch.manual_seed(0)
    kernel = _compile({PassConfigKey.TL_CPU_PARALLEL: True})
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    C = _run(kernel, A, B)
    torch.testing.assert_close(C, A @ B, rtol=1e-3, atol=1e-3)


def test_cpu_parallel_emits_pragma_and_sinks_allocs():
    kernel = _compile({PassConfigKey.TL_CPU_PARALLEL: True})
    source = kernel.get_kernel_source()
    assert "#pragma omp parallel for" in source
    # 2D grid: both dims parallelized for collapse.
    assert "collapse(2)" in source
    # Function-scope buffers must be sunk into the innermost parallelized
    # loop body (per-worker private copies) rather than staying shared at
    # function scope.
    assert source.index("float C_local") > source.index("for (int32_t by")


def test_cpu_parallel_unit_grid_dim_stays_in_chain():
    # A unit-extent middle grid dim must not cut the deeper dims off from
    # the parallel chain: M=128 gives grid (4, 1) and the collapse clause
    # still covers both dims.
    kernel = tilelang.compile(
        make_gemm(128, N, K, BLOCK_M, BLOCK_N, BLOCK_K),
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    source = kernel.get_kernel_source()
    assert "collapse(2)" in source

    torch.manual_seed(0)
    A = torch.randn(128, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    torch.testing.assert_close(kernel(A, B), A @ B, rtol=1e-3, atol=1e-3)


def test_cpu_parallel_disabled_by_default():
    kernel = _compile(None)
    source = kernel.get_kernel_source()
    assert "#pragma omp" not in source

    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    torch.testing.assert_close(_run(kernel, A, B), A @ B, rtol=1e-3, atol=1e-3)


def test_cpu_parallel_min_trip_gate():
    # Total grid trip count is 4x4=16; a threshold above that keeps the grid
    # serial (no pragma), while the switch itself stays on.
    kernel = _compile(
        {
            PassConfigKey.TL_CPU_PARALLEL: True,
            PassConfigKey.TL_CPU_PARALLEL_MIN_TRIP: 1024,
        }
    )
    source = kernel.get_kernel_source()
    assert "#pragma omp" not in source

    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    torch.testing.assert_close(_run(kernel, A, B), A @ B, rtol=1e-3, atol=1e-3)


def test_cpu_parallel_num_threads_clause():
    kernel = _compile(
        {
            PassConfigKey.TL_CPU_PARALLEL: True,
            PassConfigKey.TL_CPU_NUM_THREADS: 4,
        }
    )
    source = kernel.get_kernel_source()
    assert "num_threads(4)" in source


def test_cpu_parallel_default_off_injects_no_flags():
    # Default-off contract: the compile command stays free of OpenMP flags
    # (the injection channel is empty unless the switch is enabled).
    from tilelang.jit.adapter.libgen import cpu_openmp_flags

    assert cpu_openmp_flags(None) == []
    assert cpu_openmp_flags({}) == []
    assert cpu_openmp_flags({PassConfigKey.TL_DISABLE_VECTORIZE_256: True}) == []

    enabled = cpu_openmp_flags({PassConfigKey.TL_CPU_PARALLEL: True})
    assert "-O2" in enabled
    # With a discoverable libomp (torch bundle or Homebrew on the CI/macOS
    # hosts) the OpenMP flag must be present; a missing runtime legitimately
    # degrades to serial with only -O2.
    from tilelang.contrib.openmp import _find_libomp

    if sys.platform != "win32" and (sys.platform != "darwin" or _find_libomp() is not None):
        assert "-fopenmp" in enabled


def test_cpu_parallel_codegen_nested_parallel_keeps_pragma():
    # A kParallel loop reached through an IfThenElse inside another parallel
    # chain's body is NOT a collapse member and must keep its own pragma.
    # Regression for the chain-depth-counter bug where any kParallel seen
    # while printing a chain was wrongly suppressed.
    code = """
@I.ir_module
class Module:
    @T.prim_func
    def main():
        A = T.alloc_buffer((64,), "float32", scope="local")
        for bx in T.parallel(4):
            if bx == 0:
                for i in T.parallel(64):
                    A[bx * 16 + i] = 1.0
            for j in range(16):
                A[bx * 16 + j] = 2.0
"""
    from tilelang.cpu.codegen import build_c

    mod = tvm.script.from_source(code)
    mod = tirx.transform.BindTarget(Target("c"))(mod)
    source = build_c(mod, Target("c")).inspect_source()
    assert source.count("#pragma omp parallel for") == 2


if __name__ == "__main__":
    tilelang.testing.main()
