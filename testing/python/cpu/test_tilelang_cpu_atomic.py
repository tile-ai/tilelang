"""CPU atomic op tests (target="c", execution_backend="cython").

Two lowering paths are covered:
- tile-region: T.atomic_add/atomic_max/atomic_min on buffers/regions, lowered
  by src/cpu/op/atomic_add.cc / atomic_reduce.cc to plain serial RMW loops;
- scalar elem-op: T.atomic_add/max/min/or/load/store/addx2/addx4 on addressed
  elements, rewritten by tl.transform.LowerCPUAtomics to plain RMW.

On serial CPU, atomics degenerate to plain read-modify-write: memory_order is
accepted but ignored, and return_prev yields the pre-update value.
"""

import functools

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang import tvm


def _compile_c(func, out_idx):
    with tvm.target.Target("c"):
        return tilelang.compile(
            func,
            out_idx=out_idx,
            target="c",
            target_host="c",
            execution_backend="cython",
        )


def _make_input(shape, dtype, seed=7):
    tdtype = getattr(torch, dtype)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if tdtype in (torch.int32, torch.int64):
        return torch.randint(-50, 50, shape, dtype=tdtype, generator=gen)
    return torch.randn(shape, dtype=tdtype, generator=gen)


@pytest.mark.parametrize("dtype", ["float32", "int32", "float16"])
def test_cpu_atomic_add_scalar_parallel(dtype):
    """Scalar atomic_add into a global cell from a T.Parallel loop."""
    N = 128

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), Init: T.Tensor((1,), dtype), B: T.Tensor((1,), dtype)):
        with T.Kernel(1):
            B[0] = Init[0]
            for i in T.Parallel(N):
                T.atomic_add(B[0], A[i])

    kernel = _compile_c(main, out_idx=[2])
    A = _make_input((N,), dtype)
    Init = torch.zeros((1,), dtype=getattr(torch, dtype))
    out = kernel(A, Init)
    expected = A.sum() + Init
    if dtype == "float16":
        torch.testing.assert_close(out, expected.reshape((1,)), rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(out, expected.reshape((1,)))


def test_cpu_atomic_add_scalar_serial_memory_order():
    """Serial loop + memory_order (accepted, ignored) + codegen assertion."""
    N = 64
    dtype = "float32"

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), Init: T.Tensor((1,), dtype), B: T.Tensor((1,), dtype)):
        with T.Kernel(1):
            B[0] = Init[0]
            for i in T.serial(N):
                T.atomic_add(B[0], A[i], memory_order="seq_cst")

    kernel = _compile_c(main, out_idx=[2])
    # The intrinsic must be gone from the generated source (plain RMW).
    assert "atomic_add_elem_op" not in kernel.get_kernel_source()

    A = _make_input((N,), dtype)
    Init = torch.full((1,), 1.0, dtype=torch.float32)
    out = kernel(A, Init)
    torch.testing.assert_close(out, A.sum() + Init, rtol=1e-4, atol=1e-4)


def test_cpu_atomic_add_scalar_return_prev():
    """return_prev yields the pre-update value (exclusive prefix sum)."""
    N = 8
    dtype = "float32"

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        Init: T.Tensor((1,), dtype),
        B: T.Tensor((1,), dtype),
        P: T.Tensor((N,), dtype),
    ):
        with T.Kernel(1):
            B[0] = Init[0]
            for i in T.serial(N):
                P[i] = T.atomic_add(B[0], A[i], return_prev=True)

    kernel = _compile_c(main, out_idx=[2, 3])
    A = _make_input((N,), dtype)
    Init = torch.full((1,), 1.0, dtype=torch.float32)
    out_b, out_p = kernel(A, Init)
    expected_p = Init + torch.cumsum(A, dim=0) - A
    torch.testing.assert_close(out_p, expected_p, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_b, A.sum() + Init, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("op", ["max", "min"])
def test_cpu_atomic_max_min_scalar_return_prev(op):
    """return_prev with max/min combine: exclusive running prefix."""
    N = 8
    dtype = "float32"
    emit = T.atomic_max if op == "max" else T.atomic_min

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        Init: T.Tensor((1,), dtype),
        B: T.Tensor((1,), dtype),
        P: T.Tensor((N,), dtype),
    ):
        with T.Kernel(1):
            B[0] = Init[0]
            for i in T.serial(N):
                P[i] = emit(B[0], A[i], return_prev=True)

    kernel = _compile_c(main, out_idx=[2, 3])
    A = _make_input((N,), dtype)
    Init = torch.zeros((1,), dtype=torch.float32)
    out_b, out_p = kernel(A, Init)

    running = Init.clone()
    expected_p = torch.empty_like(A)
    combine = torch.maximum if op == "max" else torch.minimum
    for i in range(N):
        expected_p[i] = running[0]
        running = combine(running, A[i : i + 1])
    torch.testing.assert_close(out_p, expected_p, rtol=1e-4, atol=1e-4)
    expected_b = combine(Init, A.max() if op == "max" else A.min())
    torch.testing.assert_close(out_b, expected_b, rtol=1e-4, atol=1e-4)


def test_cpu_atomic_add_scalar_int_value_cast():
    """An int value added to a float buffer is cast to the dst dtype."""
    dtype = "float32"

    @T.prim_func
    def main(Init: T.Tensor((1,), dtype), B: T.Tensor((1,), dtype)):
        with T.Kernel(1):
            B[0] = Init[0]
            T.atomic_add(B[0], 3)

    kernel = _compile_c(main, out_idx=[1])
    Init = torch.full((1,), 1.5, dtype=torch.float32)
    out = kernel(Init)
    torch.testing.assert_close(out, Init + 3.0)


@pytest.mark.parametrize(
    "op,dtype",
    [("max", "float32"), ("min", "float32"), ("max", "float16"), ("min", "int32")],
)
def test_cpu_atomic_max_min_scalar(op, dtype):
    N = 64
    emit = T.atomic_max if op == "max" else T.atomic_min

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), Init: T.Tensor((1,), dtype), B: T.Tensor((1,), dtype)):
        with T.Kernel(1):
            B[0] = Init[0]
            for i in T.Parallel(N):
                emit(B[0], A[i])

    kernel = _compile_c(main, out_idx=[2])
    A = _make_input((N,), dtype)
    base = A.max() if op == "max" else A.min()
    Init = (base - 1).reshape((1,)) if op == "max" else (base + 1).reshape((1,))
    out = kernel(A, Init)
    expected = torch.maximum(A.max(), Init) if op == "max" else torch.minimum(A.min(), Init)
    if dtype == "float16":
        torch.testing.assert_close(out, expected.reshape((1,)), rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(out, expected.reshape((1,)))


def test_cpu_atomic_or_scalar():
    N = 32
    dtype = "int32"

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), Init: T.Tensor((1,), dtype), B: T.Tensor((1,), dtype)):
        with T.Kernel(1):
            B[0] = Init[0]
            for i in T.Parallel(N):
                T.atomic_or(B[0], A[i])

    kernel = _compile_c(main, out_idx=[2])
    A = _make_input((N,), dtype)
    Init = torch.zeros((1,), dtype=torch.int32)
    out = kernel(A, Init)
    expected = functools.reduce(torch.bitwise_or, list(A.unbind()), Init[0].clone())
    torch.testing.assert_close(out, expected.reshape((1,)))


def test_cpu_atomic_load_store():
    dtype = "float32"

    @T.prim_func
    def main(A: T.Tensor((1,), dtype), B: T.Tensor((1,), dtype)):
        with T.Kernel(1):
            x = T.atomic_load(A[0], memory_order="acquire")
            T.atomic_store(B[0], x + T.float32(1.0), memory_order="release")

    kernel = _compile_c(main, out_idx=[1])
    A = torch.full((1,), 2.5, dtype=torch.float32)
    out = kernel(A)
    torch.testing.assert_close(out, A + 1.0)


def test_cpu_atomic_addx2_fp16():
    """x2 vector atomic add, full-buffer form, expanded per element."""
    dtype = "float16"

    @T.prim_func
    def main(Val: T.Tensor((2,), dtype), Init: T.Tensor((2,), dtype), Dst: T.Tensor((2,), dtype)):
        with T.Kernel(1):
            for i in T.serial(2):
                Dst[i] = Init[i]
            T.atomic_addx2(Dst, Val)

    kernel = _compile_c(main, out_idx=[2])
    Val = _make_input((2,), dtype)
    Init = _make_input((2,), dtype, seed=11)
    out = kernel(Val, Init)
    torch.testing.assert_close(out, Init + Val, rtol=1e-2, atol=1e-2)


def test_cpu_atomic_addx4_fp32_slice():
    """x4 vector atomic add, slice form (region min offsets)."""
    dtype = "float32"

    @T.prim_func
    def main(Val: T.Tensor((4,), dtype), Init: T.Tensor((4,), dtype), Dst: T.Tensor((4,), dtype)):
        with T.Kernel(1):
            for i in T.serial(4):
                Dst[i] = Init[i]
            T.atomic_addx4(Dst[0:4], Val[0:4])

    kernel = _compile_c(main, out_idx=[2])
    Val = _make_input((4,), dtype)
    Init = _make_input((4,), dtype, seed=11)
    out = kernel(Val, Init)
    torch.testing.assert_close(out, Init + Val, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_cpu_atomic_add_region_global(dtype):
    """Tile-region atomic_add with a global dst buffer."""
    M, N = 4, 8

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Init: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(1):
            for i, j in T.grid(M, N):
                B[i, j] = Init[i, j]
            T.atomic_add(B, A)

    kernel = _compile_c(main, out_idx=[2])
    A = _make_input((M, N), dtype)
    Init = _make_input((M, N), dtype, seed=11)
    out = kernel(A, Init)
    torch.testing.assert_close(out, Init + A, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("op", ["max", "min"])
def test_cpu_atomic_max_min_region_local_dst(op):
    """Tile-region atomic_max/atomic_min with a local dst buffer."""
    M, N = 4, 8
    dtype = "float32"
    emit = T.atomic_max if op == "max" else T.atomic_min

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), Init: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(1):
            dst = T.alloc_local((M, N), dtype)
            for i, j in T.grid(M, N):
                dst[i, j] = Init[i, j]
            emit(dst, A)
            for i, j in T.grid(M, N):
                B[i, j] = dst[i, j]

    kernel = _compile_c(main, out_idx=[2])
    A = _make_input((M, N), dtype)
    Init = _make_input((M, N), dtype, seed=11)
    out = kernel(A, Init)
    expected = torch.maximum(Init, A) if op == "max" else torch.minimum(Init, A)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_cpu_atomic_add_region_scalar_value():
    """Tile-region atomic_add with a scalar src value (src_value path)."""
    M, N = 4, 8
    dtype = "float32"

    @T.prim_func
    def main(Init: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(1):
            for i, j in T.grid(M, N):
                B[i, j] = Init[i, j]
            T.atomic_add(B, 2)

    kernel = _compile_c(main, out_idx=[1])
    Init = _make_input((M, N), dtype)
    out = kernel(Init)
    torch.testing.assert_close(out, Init + 2.0, rtol=1e-4, atol=1e-4)


def test_cpu_atomic_add_region_sliced_dst():
    """Sliced dst region: dst_range carries a non-zero min offset."""
    M, N = 4, 8
    dtype = "float32"

    @T.prim_func
    def main(A: T.Tensor((M - 2, N), dtype), Init: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(1):
            for i, j in T.grid(M, N):
                B[i, j] = Init[i, j]
            T.atomic_add(B[1 : M - 1, :], A)

    kernel = _compile_c(main, out_idx=[2])
    A = _make_input((M - 2, N), dtype)
    Init = _make_input((M, N), dtype, seed=11)
    out = kernel(A, Init)
    expected = Init.clone()
    expected[1 : M - 1, :] += A
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_cpu_atomic_add_region_memory_order():
    """memory_order annotation on the tile-region path is accepted/ignored."""
    M, N = 4, 8
    dtype = "float32"

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(1):
            for i, j in T.grid(M, N):
                B[i, j] = 0.0
            T.atomic_add(B, A, memory_order="acquire")

    kernel = _compile_c(main, out_idx=[1])
    A = _make_input((M, N), dtype)
    out = kernel(A)
    torch.testing.assert_close(out, A, rtol=1e-4, atol=1e-4)


def test_cpu_atomic_add_use_tma_rejected():
    M, N = 4, 8
    dtype = "float32"

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype)):
        with T.Kernel(1):
            T.atomic_add(B, A, use_tma=True)

    with pytest.raises(Exception, match="use_tma"):
        _compile_c(main, out_idx=[1])


def test_cpu_atomic_addx2_return_prev_rejected():
    """Vector return_prev is rejected with a readable error on CPU."""

    @T.prim_func
    def main(
        Dst: T.Tensor((2,), "float16"),
        Val: T.Tensor((2,), "float16"),
        Prev: T.Tensor((2,), "float16"),
    ):
        with T.Kernel(1):
            Prev[0:2] = T.atomic_addx2(Dst[0:2], Val[0:2], return_prev=True)

    with pytest.raises(Exception, match="return_prev"):
        _compile_c(main, out_idx=[2])


def test_cpu_atomic_multiple_return_prev_one_statement():
    """Two return_prev atomics in one statement: both Bind/Store prefixes are
    spliced, in order, before the statement (pending_prefix_ isolation)."""
    N = 8
    dtype = "float32"

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B0: T.Tensor((1,), dtype),
        B1: T.Tensor((1,), dtype),
        P: T.Tensor((N,), dtype),
    ):
        with T.Kernel(1):
            B0[0] = 0.0
            B1[0] = 100.0
            for i in T.serial(N):
                P[i] = T.atomic_add(B0[0], A[i], return_prev=True) + T.atomic_add(B1[0], A[i], return_prev=True)

    kernel = _compile_c(main, out_idx=[1, 2, 3])
    A = _make_input((N,), dtype)
    out_b0, out_b1, out_p = kernel(A)

    prefix = torch.cumsum(A, dim=0) - A  # sum(A[:i])
    torch.testing.assert_close(out_p, prefix + (100.0 + prefix), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_b0, A.sum().reshape((1,)), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_b1, 100.0 + A.sum().reshape((1,)), rtol=1e-4, atol=1e-4)
