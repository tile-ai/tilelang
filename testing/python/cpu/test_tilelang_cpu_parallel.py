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
        with T.Kernel(M // TILE, threads=1) as bx:
            buf1 = T.alloc_buffer((TILE,), "float32", scope="local")
            for i in T.serial(TILE):
                buf1[i] = A[bx * TILE + i] + 1.0
            for i in T.serial(TILE):
                B[bx * TILE + i] = buf1[i]
        with T.Kernel(M // TILE, threads=1) as bx2:
            buf2 = T.alloc_buffer((TILE,), "float32", scope="local")
            for i in T.serial(TILE):
                buf2[i] = A[bx2 * TILE + i] * 2.0
            for i in T.serial(TILE):
                C[bx2 * TILE + i] = buf2[i]

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
        with T.Kernel(M // TILE, threads=1) as bx:
            for i in T.serial(TILE):
                B[bx * TILE + i] = A[bx * TILE + i] * tbl[i]

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


def test_cpu_parallel_atomic_stays_serial():
    # Kernels calling atomic ops stay serial: atomics lower to plain
    # read-modify-write, which would race across workers in a parallel grid.
    N_ATOMIC = 200000

    @T.prim_func
    def atomic_sum(A: T.Tensor((N_ATOMIC,), "float32"), B: T.Tensor((1,), "float32")):
        B[0] = 0.0  # initialize the accumulator before the grid
        with T.Kernel(200, threads=1) as bx:
            for i in T.serial(1000):
                T.atomic_add(B[0], A[bx * 1000 + i])

    kernel = tilelang.compile(
        atomic_sum,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    assert "#pragma omp" not in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randn(N_ATOMIC, dtype=torch.float32)
    torch.testing.assert_close(kernel(A)[0], A.sum(), rtol=1e-4, atol=1e-3)


def test_cpu_parallel_cross_iteration_state_stays_serial():
    # A buffer carrying state across grid iterations (read-modify-write with
    # no per-iteration reset) must not be privatized; since it is also
    # mutated inside, the nest stays serial.
    N_RANK = 512

    @T.prim_func
    def rank(A: T.Tensor((N_RANK,), "float32"), B: T.Tensor((N_RANK,), "float32")):
        with T.Kernel(N_RANK // 128, threads=1) as bx:
            acc = T.alloc_buffer((1,), "float32", scope="local")
            for i in T.serial(128):
                acc[0] = 0.0 if (bx == 0 and i == 0) else acc[0] + 1.0
                B[bx * 128 + i] = acc[0] + A[bx * 128 + i] * 0.0

    kernel = tilelang.compile(
        rank,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    assert "#pragma omp" not in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randn(N_RANK, dtype=torch.float32)
    torch.testing.assert_close(kernel(A), torch.arange(N_RANK, dtype=torch.float32), rtol=1e-6, atol=1e-6)


def test_cpu_parallel_param_overlapping_store_stays_serial():
    # A store to a parameter buffer at an iteration-invariant address with an
    # iteration-dependent value is a definite overlapping write: the nest
    # must stay serial.
    TILE = 128

    @T.prim_func
    def overlapping(A: T.Tensor((M,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(M // TILE, M // TILE, threads=1) as (bx, by):
            for i in T.serial(TILE):
                B[0] = A[bx * TILE + i] + by * 0.0

    kernel = tilelang.compile(
        overlapping,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    assert "#pragma omp" not in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randn(M, dtype=torch.float32)
    # Serial semantics: the last write (bx=3, i=127) wins.
    torch.testing.assert_close(kernel(A)[0], A[M - 1], rtol=1e-6, atol=1e-6)


def test_cpu_parallel_region_atomic_stays_serial():
    # Region-form atomics (tl.tileop.atomic*) must also be marked: they
    # lower to serial RMW loops, which would race across workers.
    N_ATOMIC = 256

    @T.prim_func
    def region_atomic(A: T.Tensor((N_ATOMIC,), "float32"), B: T.Tensor((N_ATOMIC,), "float32")):
        for i in T.serial(N_ATOMIC):
            B[i] = 0.0
        with T.Kernel(N_ATOMIC, threads=1):
            T.atomic_add(B, A)

    kernel = tilelang.compile(
        region_atomic,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    assert "#pragma omp" not in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randn(N_ATOMIC, dtype=torch.float32)
    torch.testing.assert_close(kernel(A), N_ATOMIC * A, rtol=1e-4, atol=1e-3)


def test_cpu_parallel_partial_init_stays_serial():
    # A partial write (state[0]) must not count as a whole-buffer
    # per-iteration reset for state[1], which accumulates across grid
    # iterations: the nest stays serial.
    N_PI = 256
    BLOCK_PI = 32

    @T.prim_func
    def partial_init(A: T.Tensor((N_PI,), "float32"), B: T.Tensor((N_PI,), "float32")):
        with T.Kernel(N_PI // BLOCK_PI, threads=1) as bx:
            state = T.alloc_buffer((2,), "float32", scope="local")
            state[0] = 1.0
            if bx == 0:
                state[1] = 0.0
            for i in T.serial(BLOCK_PI):
                state[1] = state[1] + A[bx * BLOCK_PI + i]
                B[bx * BLOCK_PI + i] = state[1]

    kernel = tilelang.compile(
        partial_init,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    assert "#pragma omp" not in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randn(N_PI, dtype=torch.float32)
    torch.testing.assert_close(kernel(A), torch.cumsum(A, 0), rtol=1e-4, atol=1e-3)


def test_cpu_parallel_colliding_affine_store_stays_serial():
    # B[bx % 2] += ... — the address varies with the grid var but is not
    # injective (two blocks collide on each slot): the nest stays serial.
    N_COL = 256
    BLOCK_COL = 32

    @T.prim_func
    def collide(A: T.Tensor((N_COL,), "float32"), B: T.Tensor((2,), "float32")):
        B[0] = 0.0
        B[1] = 0.0
        with T.Kernel(N_COL // BLOCK_COL, threads=1) as bx:
            for i in T.serial(BLOCK_COL):
                B[bx % 2] = B[bx % 2] + A[bx * BLOCK_COL + i]

    kernel = tilelang.compile(
        collide,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    assert "#pragma omp" not in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randn(N_COL, dtype=torch.float32)
    expected = torch.stack([A.reshape(-1, BLOCK_COL)[0::2].sum(), A.reshape(-1, BLOCK_COL)[1::2].sum()])
    torch.testing.assert_close(kernel(A), expected, rtol=1e-4, atol=1e-3)


def test_cpu_parallel_shared_rmw_no_grid_var_stays_serial():
    # B[0] += A[i] — the value carries no grid var, but a shared RMW on an
    # iteration-invariant address is still a race: the nest stays serial.
    N_RMW = 256
    BLOCK_RMW = 32

    @T.prim_func
    def shared_rmw(A: T.Tensor((N_RMW,), "float32"), B: T.Tensor((1,), "float32")):
        B[0] = 0.0
        with T.Kernel(N_RMW // BLOCK_RMW, threads=1):
            for i in T.serial(BLOCK_RMW):
                B[0] = B[0] + A[i]

    kernel = tilelang.compile(
        shared_rmw,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    assert "#pragma omp" not in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randn(N_RMW, dtype=torch.float32)
    expected = A[:BLOCK_RMW].sum() * (N_RMW // BLOCK_RMW)
    torch.testing.assert_close(kernel(A)[0], expected, rtol=1e-4, atol=1e-3)


def test_cpu_parallel_extern_write_to_param_stays_serial():
    # A call_extern that writes a parameter buffer through a pointer is an
    # unanalyzable opaque use; the nest must stay serial.
    N_EXT = 4096

    @T.prim_func
    def extern_write(
        A: T.Tensor((N_EXT,), "float32"),
        B: T.Tensor((1,), "float32"),
    ):
        B[0] = 0.0
        with T.Kernel(
            N_EXT,
            threads=1,
            prelude="static inline void writer(float* p) { *p += 1.0f; }\n",
        ) as _bx:
            T.call_extern("void", "writer", T.address_of(B[0]))

    kernel = tilelang.compile(
        extern_write,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    assert "#pragma omp" not in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randn(N_EXT, dtype=torch.float32)
    torch.testing.assert_close(kernel(A)[0], torch.tensor(float(N_EXT)))


def test_cpu_parallel_extern_bare_data_var_stays_serial():
    # Same as above, but the buffer reaches the callee as a bare data var
    # (no BufferLoad/Store inside the nest at all); still an opaque write.
    N_BV = 4096

    @T.prim_func
    def extern_bare(
        A: T.Tensor((N_BV,), "float32"),
        B: T.Tensor((1,), "float32"),
    ):
        B[0] = 0.0
        with T.Kernel(
            N_BV,
            threads=1,
            prelude="static inline void writer(float* p) { *p += 1.0f; }\n",
        ) as _bx:
            T.call_extern("void", "writer", B.data)

    kernel = tilelang.compile(
        extern_bare,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    assert "#pragma omp" not in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randn(N_BV, dtype=torch.float32)
    torch.testing.assert_close(kernel(A)[0], torch.tensor(float(N_BV)))


def test_cpu_parallel_address_of_write_range_stays_serial():
    # The callee writes past the addressed element (p[0] and p[1]), so
    # adjacent iterations' write sets overlap even though the start
    # addresses are injective; the nest must stay serial.
    N_AW = 4096

    @T.prim_func
    def address_of_range(
        A: T.Tensor((N_AW,), "float32"),
        B: T.Tensor((N_AW + 1,), "float32"),
    ):
        with T.Kernel(
            N_AW,
            threads=1,
            prelude="static inline void writer(float* p) {\n    p[0] += 1.0f;\n    p[1] += 1.0f;\n}\n",
        ) as bx:
            T.call_extern("void", "writer", T.address_of(B[bx]))

    kernel = tilelang.compile(
        address_of_range,
        target="c",
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    assert "#pragma omp" not in kernel.get_kernel_source()

    A = torch.zeros(N_AW, dtype=torch.float32)
    B = torch.zeros(N_AW + 1, dtype=torch.float32)
    kernel(A, B)
    torch.testing.assert_close(B.sum(), torch.tensor(float(2 * N_AW)))


def test_cpu_parallel_cross_store_collision_stays_serial():
    # B[bx] and B[bx+1] are each injective on their own, but iteration bx and
    # bx+1 collide on B[bx+1]: the nest must stay serial.
    N_CS = 256

    @T.prim_func
    def cross_store(A: T.Tensor((N_CS,), "float32"), B: T.Tensor((N_CS,), "float32")):
        with T.Kernel(N_CS, threads=1) as bx:
            if bx + 1 < N_CS:
                B[bx + 1] = A[bx]
            B[bx] = A[bx]

    kernel = tilelang.compile(
        cross_store,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    assert "#pragma omp" not in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randn(N_CS, dtype=torch.float32)
    torch.testing.assert_close(kernel(A), A, rtol=1e-6, atol=1e-6)


def test_cpu_parallel_loop_carried_dependency_stays_serial():
    # B[bx+1] = B[bx] + A[bx]: the store address is injective, but the load
    # reads a value written by another iteration — a loop-carried dependency.
    N_LC = 256

    @T.prim_func
    def loop_carried(A: T.Tensor((N_LC,), "float32"), B: T.Tensor((N_LC,), "float32")):
        B[0] = 0.0
        with T.Kernel(N_LC - 1, threads=1) as bx:
            B[bx + 1] = B[bx] + A[bx]

    kernel = tilelang.compile(
        loop_carried,
        target="c",
        out_idx=-1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    assert "#pragma omp" not in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randn(N_LC, dtype=torch.float32)
    expected = torch.zeros(N_LC, dtype=torch.float32)
    expected[1:] = torch.cumsum(A, 0)[:-1]
    torch.testing.assert_close(kernel(A), expected, rtol=1e-4, atol=1e-3)


def test_cpu_parallel_zero_trip_reset_stays_serial():
    # A store inside a possibly zero-trip dynamic loop is not a provable
    # per-iteration reset: the buffer is not iteration-private, so the nest
    # stays serial.

    @T.prim_func
    def zero_trip(
        A: T.Tensor((256,), "float32"),
        B: T.Tensor((256,), "float32"),
        n: T.int32,
    ):
        with T.Kernel(256, threads=1) as bx:
            s = T.alloc_buffer((1,), "float32", scope="local")
            for _t in T.serial(n):
                s[0] = 0.0
            s[0] = s[0] + A[bx]
            B[bx] = s[0]

    kernel = tilelang.compile(
        zero_trip,
        target="c",
        out_idx=1,
        execution_backend="cython",
        pass_configs={PassConfigKey.TL_CPU_PARALLEL: True},
    )
    assert "#pragma omp" not in kernel.get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
