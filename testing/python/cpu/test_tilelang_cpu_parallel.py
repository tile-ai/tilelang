"""OpenMP CPU grid lowering, private scratch buffers, and serial fallback."""

import pytest
import torch

import tilelang
import tilelang.cpu.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey

import tvm
from tvm import tirx
from tvm.target import Target

M = N = 512
K = 64
BLOCK_M = BLOCK_N = 128
BLOCK_K = 32


def _make_gemm(rows, cpu_num_threads):
    @T.prim_func
    def gemm(
        A: T.Tensor((rows, K), dtype=T.float32),
        B: T.Tensor((K, N), dtype=T.float32),
        C: T.Tensor((rows, N), dtype=T.float32),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(rows, BLOCK_M), cpu_num_threads=cpu_num_threads) as (bx, by):
            A_shared = T.alloc_buffer((BLOCK_M, BLOCK_K), dtype=T.float32, scope="shared")
            B_shared = T.alloc_buffer((BLOCK_K, BLOCK_N), dtype=T.float32, scope="shared")
            C_local = T.alloc_buffer((BLOCK_M, BLOCK_N), dtype=T.float32, scope="local")
            T.clear(C_local)
            for ko in T.Pipelined(K // BLOCK_K, num_stages=1):
                T.copy(A[by * BLOCK_M, ko * BLOCK_K], A_shared)
                T.copy(B[ko * BLOCK_K, bx * BLOCK_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])

    return gemm


def _compile_parallel(func, out_idx=-1):
    return tilelang.compile(func, target="c", out_idx=out_idx, pass_configs={PassConfigKey.TL_CPU_PARALLEL: True})


@pytest.mark.parametrize(
    "rows,pass_configs,cpu_num_threads,parallel",
    [
        pytest.param(M, {PassConfigKey.TL_CPU_PARALLEL: True}, 4, True, id="parallel"),
        pytest.param(BLOCK_M, {PassConfigKey.TL_CPU_PARALLEL: True}, None, True, id="unit_grid_dim"),
        pytest.param(M, None, None, False, id="default_off"),
        pytest.param(
            M,
            {PassConfigKey.TL_CPU_PARALLEL: True, PassConfigKey.TL_CPU_PARALLEL_MIN_TRIP: 1024},
            None,
            False,
            id="min_trip",
        ),
    ],
)
def test_cpu_parallel_gemm_correctness(rows, pass_configs, cpu_num_threads, parallel):
    kernel = tilelang.compile(
        _make_gemm(rows, cpu_num_threads),
        target="c",
        out_idx=-1,
        pass_configs=pass_configs,
    )
    source = kernel.get_kernel_source()
    if parallel:
        pragma = "#pragma omp parallel for collapse(2)"
        if cpu_num_threads is not None:
            pragma += f" num_threads({cpu_num_threads})"
        assert pragma in source
        assert source.index("float C_local") > source.index("for (int32_t by")
    else:
        assert "#pragma omp" not in source
    if cpu_num_threads is None:
        assert "num_threads" not in source

    torch.manual_seed(0)
    A = torch.randn(rows, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    torch.testing.assert_close(kernel(A, B), A @ B, rtol=1e-3, atol=1e-3)


def test_cpu_parallel_codegen_nested_parallel_keeps_pragma():
    # A parallel loop inside a conditional needs its own pragma.
    @T.prim_func
    def main():
        A = T.alloc_buffer((64,), T.float32, scope="local")
        for bx in T.parallel(4):
            if bx == 0:
                for i in T.parallel(64):
                    A[bx * 16 + i] = 1.0
            for j in range(16):
                A[bx * 16 + j] = 2.0

    from tilelang.cpu.codegen import build_c

    mod = tvm.IRModule.from_expr(main)
    mod = tirx.transform.BindTarget(Target("c"))(mod)
    source = build_c(mod, Target("c")).inspect_source()
    assert source.count("#pragma omp parallel for") == 2


@pytest.mark.parametrize("mode", ["parallel", "default_off", "min_trip"])
def test_cpu_parallel_two_sequential_kernels(mode):
    TILE = 128

    @T.prim_func
    def two_kernels(
        A: T.Tensor((M,), T.float32),
        B: T.Tensor((M,), T.float32),
        C: T.Tensor((M,), T.float32),
    ):
        with T.Kernel(M // TILE, cpu_num_threads=2) as bx:
            buf1 = T.alloc_buffer((TILE,), T.float32, scope="local")
            for i in T.serial(TILE):
                buf1[i] = A[bx * TILE + i] + 1.0
            for i in T.serial(TILE):
                B[bx * TILE + i] = buf1[i]
        with T.Kernel(M // TILE, cpu_num_threads=3) as bx2:
            buf2 = T.alloc_buffer((TILE,), T.float32, scope="local")
            for i in T.serial(TILE):
                buf2[i] = A[bx2 * TILE + i] * 2.0
            for i in T.serial(TILE):
                C[bx2 * TILE + i] = buf2[i]

    pass_configs = None if mode == "default_off" else {PassConfigKey.TL_CPU_PARALLEL: True}
    if mode == "min_trip":
        pass_configs[PassConfigKey.TL_CPU_PARALLEL_MIN_TRIP] = 1024
    kernel = tilelang.compile(two_kernels, target="c", out_idx=[-2, -1], pass_configs=pass_configs)
    source = kernel.get_kernel_source()
    if mode == "parallel":
        assert source.count("#pragma omp parallel for") == 2
        assert source.count("num_threads(2)") == 1
        assert source.count("num_threads(3)") == 1
    else:
        assert "#pragma omp" not in source

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
    def dyn(A: T.Tensor((m,), T.float32), B: T.Tensor((m,), T.float32)):
        with T.Kernel(T.ceildiv(m, 128)) as bx:
            for i in T.serial(128):
                if bx * 128 + i < m:
                    B[bx * 128 + i] = A[bx * 128 + i] * 2.0

    kernel = _compile_parallel(dyn)
    source = kernel.get_kernel_source()
    assert "#pragma omp parallel for" in source

    torch.manual_seed(0)
    A = torch.randn(1000, dtype=torch.float32)
    torch.testing.assert_close(kernel(A), A * 2.0, rtol=1e-6, atol=1e-6)


def test_cpu_parallel_mutable_state_outside_nest_stays_serial():
    # Mixed case: a normal store inside the nest plus an opaque use outside
    # it. The buffer cannot be privatized (the outside use would dangle), and
    # sharing it across workers would race — the nest must stay serial.
    TILE = 128

    @T.prim_func
    def mixed_use(
        A: T.Tensor((M,), T.float32),
        B: T.Tensor((M,), T.float32),
    ):
        buf = T.alloc_buffer((TILE,), T.float32, scope="local")
        with T.Kernel(
            M // TILE,
            prelude='extern "C" void my_sink(float* p, int n) { for (int t = 0; t < n; ++t) p[t] = 0.0f; }\n',
        ) as bx:
            for i in T.serial(TILE):
                buf[i] = A[bx * TILE + i]
            for i in T.serial(TILE):
                B[bx * TILE + i] = buf[i]
        T.call_extern("void", "my_sink", buf.data, TILE)

    kernel = _compile_parallel(mixed_use)
    source = kernel.get_kernel_source()
    assert "#pragma omp" not in source

    torch.manual_seed(0)
    A = torch.randn(M, dtype=torch.float32)
    torch.testing.assert_close(kernel(A), A, rtol=1e-6, atol=1e-6)


def test_cpu_parallel_outside_first_access_stays_serial():
    # An outside access prevents privatization regardless of access order.
    TILE = 128

    @T.prim_func
    def outside_first(
        A: T.Tensor((M,), T.float32),
        B: T.Tensor((M,), T.float32),
    ):
        buf = T.alloc_buffer((TILE,), T.float32, scope="local")
        buf[0] = 0.0  # outside access before the kernel nest
        with T.Kernel(M // TILE) as bx:
            for i in T.serial(TILE):
                buf[i] = A[bx * TILE + i]
            for i in T.serial(TILE):
                B[bx * TILE + i] = buf[i]

    kernel = _compile_parallel(outside_first)
    source = kernel.get_kernel_source()
    assert "#pragma omp" not in source

    torch.manual_seed(0)
    A = torch.randn(M, dtype=torch.float32)
    torch.testing.assert_close(kernel(A), A, rtol=1e-6, atol=1e-6)


def test_cpu_parallel_readonly_shared_table_still_parallelizes():
    # Load-only sharing is race-free: a buffer initialized before the nest
    # and only read inside it must not block parallelization. A pure math call
    # must also remain eligible after the opaque-call safety check.
    TILE = 128

    @T.prim_func
    def table_read(
        A: T.Tensor((M,), T.float32),
        B: T.Tensor((M,), T.float32),
    ):
        tbl = T.alloc_buffer((TILE,), T.float32, scope="local")
        for i in T.serial(TILE):
            tbl[i] = 2.0
        with T.Kernel(M // TILE) as bx:
            for i in T.serial(TILE):
                B[bx * TILE + i] = T.sqrt(A[bx * TILE + i] * A[bx * TILE + i]) * tbl[i]

    kernel = _compile_parallel(table_read)
    source = kernel.get_kernel_source()
    assert "#pragma omp parallel for" in source

    torch.manual_seed(0)
    A = torch.randn(M, dtype=torch.float32)
    torch.testing.assert_close(kernel(A), A.abs() * 2.0, rtol=1e-6, atol=1e-6)


def test_cpu_parallel_atomic_stays_serial():
    # Kernels calling atomic ops stay serial: atomics lower to plain
    # read-modify-write, which would race across workers in a parallel grid.
    N_ATOMIC = 200000

    @T.prim_func
    def atomic_sum(A: T.Tensor((N_ATOMIC,), T.float32), B: T.Tensor((1,), T.float32)):
        B[0] = 0.0  # initialize the accumulator before the grid
        with T.Kernel(200) as bx:
            for i in T.serial(1000):
                T.atomic_add(B[0], A[bx * 1000 + i])

    kernel = _compile_parallel(atomic_sum)
    assert "#pragma omp" not in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randn(N_ATOMIC, dtype=torch.float32)
    torch.testing.assert_close(kernel(A)[0], A.sum(), rtol=1e-4, atol=1e-3)


def test_cpu_parallel_param_overlapping_store_stays_serial():
    # A store to a parameter buffer at an iteration-invariant address with an
    # iteration-dependent value is a definite overlapping write: the nest
    # must stay serial.
    TILE = 128

    @T.prim_func
    def overlapping(A: T.Tensor((M,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(M // TILE, M // TILE) as (bx, by):
            for i in T.serial(TILE):
                B[0] = A[bx * TILE + i] + by * 0.0

    kernel = _compile_parallel(overlapping)
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
    def region_atomic(A: T.Tensor((N_ATOMIC,), T.float32), B: T.Tensor((N_ATOMIC,), T.float32)):
        for i in T.serial(N_ATOMIC):
            B[i] = 0.0
        with T.Kernel(N_ATOMIC):
            T.atomic_add(B, A)

    kernel = _compile_parallel(region_atomic)
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
    def partial_init(A: T.Tensor((N_PI,), T.float32), B: T.Tensor((N_PI,), T.float32)):
        with T.Kernel(N_PI // BLOCK_PI) as bx:
            state = T.alloc_buffer((2,), T.float32, scope="local")
            state[0] = 1.0
            if bx == 0:
                state[1] = 0.0
            for i in T.serial(BLOCK_PI):
                state[1] = state[1] + A[bx * BLOCK_PI + i]
                B[bx * BLOCK_PI + i] = state[1]

    kernel = _compile_parallel(partial_init)
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
    def collide(A: T.Tensor((N_COL,), T.float32), B: T.Tensor((2,), T.float32)):
        B[0] = 0.0
        B[1] = 0.0
        with T.Kernel(N_COL // BLOCK_COL) as bx:
            for i in T.serial(BLOCK_COL):
                B[bx % 2] = B[bx % 2] + A[bx * BLOCK_COL + i]

    kernel = _compile_parallel(collide)
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
    def shared_rmw(A: T.Tensor((N_RMW,), T.float32), B: T.Tensor((1,), T.float32)):
        B[0] = 0.0
        with T.Kernel(N_RMW // BLOCK_RMW):
            for i in T.serial(BLOCK_RMW):
                B[0] = B[0] + A[i]

    kernel = _compile_parallel(shared_rmw)
    assert "#pragma omp" not in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randn(N_RMW, dtype=torch.float32)
    expected = A[:BLOCK_RMW].sum() * (N_RMW // BLOCK_RMW)
    torch.testing.assert_close(kernel(A)[0], expected, rtol=1e-4, atol=1e-3)


def test_cpu_parallel_address_of_write_range_stays_serial():
    # The callee writes past the addressed element (p[0] and p[1]), so
    # adjacent iterations' write sets overlap even though the start
    # addresses are injective; the nest must stay serial.
    N_AW = 4096

    @T.prim_func
    def address_of_range(B: T.Tensor((N_AW + 1,), T.float32)):
        with T.Kernel(
            N_AW,
            prelude='extern "C" void writer(float* p) {\n    p[0] += 1.0f;\n    p[1] += 1.0f;\n}\n',
        ) as bx:
            T.call_extern("void", "writer", T.address_of(B[bx]))

    kernel = _compile_parallel(address_of_range, out_idx=None)
    assert "#pragma omp" not in kernel.get_kernel_source()

    B = torch.zeros(N_AW + 1, dtype=torch.float32)
    kernel(B)
    torch.testing.assert_close(B.sum(), torch.tensor(float(2 * N_AW)))


def test_cpu_parallel_cross_store_collision_stays_serial():
    # B[bx] and B[bx+1] are each injective on their own, but iteration bx and
    # bx+1 collide on B[bx+1]: the nest must stay serial.
    N_CS = 256

    @T.prim_func
    def cross_store(A: T.Tensor((N_CS,), T.float32), B: T.Tensor((N_CS,), T.float32)):
        with T.Kernel(N_CS) as bx:
            if bx + 1 < N_CS:
                B[bx + 1] = A[bx]
            B[bx] = A[bx]

    kernel = _compile_parallel(cross_store)
    assert "#pragma omp" not in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randn(N_CS, dtype=torch.float32)
    torch.testing.assert_close(kernel(A), A, rtol=1e-6, atol=1e-6)


def test_cpu_parallel_loop_carried_dependency_stays_serial():
    # B[bx+1] = B[bx] + A[bx]: the store address is injective, but the load
    # reads a value written by another iteration — a loop-carried dependency.
    N_LC = 256

    @T.prim_func
    def loop_carried(A: T.Tensor((N_LC,), T.float32), B: T.Tensor((N_LC,), T.float32)):
        B[0] = 0.0
        with T.Kernel(N_LC - 1) as bx:
            B[bx + 1] = B[bx] + A[bx]

    kernel = _compile_parallel(loop_carried)
    assert "#pragma omp" not in kernel.get_kernel_source()

    torch.manual_seed(0)
    A = torch.randn(N_LC, dtype=torch.float32)
    expected = torch.zeros(N_LC, dtype=torch.float32)
    expected[1:] = torch.cumsum(A, 0)[:-1]
    torch.testing.assert_close(kernel(A), expected, rtol=1e-4, atol=1e-3)


if __name__ == "__main__":
    tilelang.testing.main()
