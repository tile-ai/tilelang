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


def make_gemm(M, N, K, BM, BN, BK, cpu_num_threads=None):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype="float32"),
        B: T.Tensor((K, N), dtype="float32"),
        C: T.Tensor((M, N), dtype="float32"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), cpu_num_threads=cpu_num_threads) as (bx, by):
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


def _compile(pass_configs, cpu_num_threads=None):
    return tilelang.compile(
        make_gemm(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, cpu_num_threads=cpu_num_threads),
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
    kernel = _compile({PassConfigKey.TL_CPU_PARALLEL: True}, cpu_num_threads=4)
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


def test_cpu_parallel_two_sequential_kernels():
    # Regression: alloc sinking must attribute uses to the nest they belong
    # to — the second kernel's scratch buffer used to be sunk into the first
    # nest, failing the C compile with "use of undeclared identifier". Both
    # sibling nests are parallelized independently.
    TILE = 128

    @T.prim_func
    def two_kernels(
        A: T.Tensor((M,), "float32"),
        B: T.Tensor((M,), "float32"),
        C: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M // TILE, M // TILE, threads=1) as (bx, by):
            buf1 = T.alloc_buffer((TILE,), "float32", scope="local")
            for i in T.serial(TILE):
                buf1[i] = A[bx * TILE + i] + 1.0
            for i in T.serial(TILE):
                B[bx * TILE + i] = buf1[i] + by * 0.0
        with T.Kernel(M // TILE, M // TILE, threads=1) as (bx2, by2):
            buf2 = T.alloc_buffer((TILE,), "float32", scope="local")
            for i in T.serial(TILE):
                buf2[i] = A[bx2 * TILE + i] * 2.0
            for i in T.serial(TILE):
                C[bx2 * TILE + i] = buf2[i] + by2 * 0.0

    kernel = tilelang.compile(
        two_kernels,
        target="c",
        out_idx=[-2, -1],
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    source = kernel.get_kernel_source()
    # Each nest becomes its own parallel region.
    assert source.count("#pragma omp parallel for") == 2

    torch.manual_seed(0)
    A = torch.randn(M, dtype=torch.float32)
    B, C = kernel(A)
    torch.testing.assert_close(B, A + 1.0, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(C, A * 2.0, rtol=1e-6, atol=1e-6)


def test_cpu_parallel_dynamic_extent():
    # Symbolic grid extents are parallelized too: the assume AttrStmt that
    # InjectAssumes wraps around symbolic-shape kernels is transparent to the
    # pass, and OpenMP handles runtime trip counts (the min_trip gate is off
    # by default).
    m = T.dynamic("m")

    @T.prim_func
    def dyn(A: T.Tensor((m,), "float32"), B: T.Tensor((m,), "float32")):
        with T.Kernel(T.ceildiv(m, 128), threads=1) as bx:
            for i in T.serial(128):
                if bx * 128 + i < m:
                    B[bx * 128 + i] = A[bx * 128 + i] * 2.0

    kernel = tilelang.compile(
        dyn,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    source = kernel.get_kernel_source()
    assert "#pragma omp parallel for" in source

    torch.manual_seed(0)
    A = torch.randn(1000, dtype=torch.float32)
    torch.testing.assert_close(kernel(A), A * 2.0, rtol=1e-6, atol=1e-6)


def test_cpu_parallel_opaque_use_stays_serial():
    # A buffer whose in-nest use is opaque (call_extern on its data var)
    # cannot be proven iteration-private; parallelizing with it shared would
    # race (my_sink mutates it), so the nest must stay serial.
    TILE = 128

    @T.prim_func
    def opaque_only(
        A: T.Tensor((M,), "float32"),
        B: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(
            M // TILE,
            M // TILE,
            threads=1,
            prelude='extern "C" void my_sink(float* p, int n) { for (int t = 0; t < n; ++t) p[t] += 1.0f; }\n',
        ) as (bx, by):
            buf = T.alloc_buffer((TILE,), "float32", scope="local")
            T.call_extern("void", "my_sink", buf.data, TILE)
            for i in T.serial(TILE):
                B[bx * TILE + i] = A[bx * TILE + i] + by * 0.0

    kernel = tilelang.compile(
        opaque_only,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    source = kernel.get_kernel_source()
    assert "#pragma omp" not in source

    torch.manual_seed(0)
    A = torch.randn(M, dtype=torch.float32)
    torch.testing.assert_close(kernel(A), A, rtol=1e-6, atol=1e-6)


def test_cpu_parallel_mutable_state_outside_nest_stays_serial():
    # Mixed case: a normal store inside the nest plus an opaque use outside
    # it. The buffer cannot be privatized (the outside use would dangle), and
    # sharing it across workers would race — the nest must stay serial.
    # (Sinking it used to break the C compile with an undeclared identifier.)
    TILE = 128

    @T.prim_func
    def mixed_use(
        A: T.Tensor((M,), "float32"),
        B: T.Tensor((M,), "float32"),
    ):
        buf = T.alloc_buffer((TILE,), "float32", scope="local")
        with T.Kernel(
            M // TILE,
            M // TILE,
            threads=1,
            prelude='extern "C" void my_sink(float* p, int n) { for (int t = 0; t < n; ++t) p[t] = 0.0f; }\n',
        ) as (bx, by):
            for i in T.serial(TILE):
                buf[i] = A[bx * TILE + i]
            for i in T.serial(TILE):
                B[bx * TILE + i] = buf[i] + by * 0.0
        T.call_extern("void", "my_sink", buf.data, TILE)

    kernel = tilelang.compile(
        mixed_use,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    source = kernel.get_kernel_source()
    assert "#pragma omp" not in source

    torch.manual_seed(0)
    A = torch.randn(M, dtype=torch.float32)
    torch.testing.assert_close(kernel(A), A, rtol=1e-6, atol=1e-6)


def test_cpu_parallel_outside_first_access_stays_serial():
    # Regression: an outside access that comes *before* the first in-nest
    # access used to leave no trace (min_depth got overwritten), so the
    # buffer was sunk into the nest and the outside store crashed the
    # pipeline with "used before definition". The nest must stay serial.
    TILE = 128

    @T.prim_func
    def outside_first(
        A: T.Tensor((M,), "float32"),
        B: T.Tensor((M,), "float32"),
    ):
        buf = T.alloc_buffer((TILE,), "float32", scope="local")
        buf[0] = 0.0  # outside access before the kernel nest
        with T.Kernel(M // TILE, M // TILE, threads=1) as (bx, by):
            for i in T.serial(TILE):
                buf[i] = A[bx * TILE + i]
            for i in T.serial(TILE):
                B[bx * TILE + i] = buf[i] + by * 0.0

    kernel = tilelang.compile(
        outside_first,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    source = kernel.get_kernel_source()
    assert "#pragma omp" not in source

    torch.manual_seed(0)
    A = torch.randn(M, dtype=torch.float32)
    torch.testing.assert_close(kernel(A), A, rtol=1e-6, atol=1e-6)


def test_cpu_parallel_readonly_shared_table_still_parallelizes():
    # Load-only sharing is race-free: a buffer initialized before the nest
    # and only read inside it must not block parallelization.
    TILE = 128

    @T.prim_func
    def table_read(
        A: T.Tensor((M,), "float32"),
        B: T.Tensor((M,), "float32"),
    ):
        tbl = T.alloc_buffer((TILE,), "float32", scope="local")
        for i in T.serial(TILE):
            tbl[i] = 2.0
        with T.Kernel(M // TILE, M // TILE, threads=1) as (bx, by):
            for i in T.serial(TILE):
                B[bx * TILE + i] = A[bx * TILE + i] * tbl[i] + by * 0.0

    kernel = tilelang.compile(
        table_read,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    source = kernel.get_kernel_source()
    assert "#pragma omp parallel for" in source

    torch.manual_seed(0)
    A = torch.randn(M, dtype=torch.float32)
    torch.testing.assert_close(kernel(A), A * 2.0, rtol=1e-6, atol=1e-6)


def test_cpu_parallel_address_of_use_stays_serial():
    # address_of wraps a BufferLoad, which would otherwise hide the
    # callee-mutated buffer from the opaque-use check; the nest must stay
    # serial.
    TILE = 128

    @T.prim_func
    def addr_of_use(
        A: T.Tensor((M,), "float32"),
        B: T.Tensor((M,), "float32"),
    ):
        buf = T.alloc_buffer((TILE,), "float32", scope="local")
        for i in T.serial(TILE):
            buf[i] = 0.0
        with T.Kernel(
            M // TILE,
            M // TILE,
            threads=1,
            prelude='extern "C" void writer(float* p) { p[0] += 1.0f; }\n',
        ) as (bx, by):
            T.call_extern("void", "writer", T.address_of(buf[0]))
            for i in T.serial(TILE):
                B[bx * TILE + i] = A[bx * TILE + i] + by * 0.0
        for i in T.serial(TILE):
            B[i] = B[i] + buf[i]

    kernel = tilelang.compile(
        addr_of_use,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    source = kernel.get_kernel_source()
    assert "#pragma omp" not in source

    torch.manual_seed(0)
    A = torch.randn(M, dtype=torch.float32)
    expected = A.clone()
    expected[0] += 16.0  # writer increments buf[0] once per grid iteration
    torch.testing.assert_close(kernel(A), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    tilelang.testing.main()
