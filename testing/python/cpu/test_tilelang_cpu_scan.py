"""Serial CPU scan lowering and region semantics."""

import pytest
import torch

import tilelang
import tilelang.cpu.language as T


_TARGETS = [("c", "cython"), ("llvm", "tvm_ffi")]
_INPUT = torch.tensor([3.0, -1.0, 2.0, -5.0, 4.0, 0.0, -2.0, 7.0, -6.0, 1.0, 5.0, -3.0, 2.0, -4.0, 6.0, -7.0, 0.5])


def _compile(func, target, execution_backend):
    return tilelang.compile(
        func,
        target=target,
        target_host="c" if target == "c" else None,
        execution_backend=execution_backend,
        out_idx=-1,
    )


def _scan_reference(a, op, dim, reverse):
    if reverse:
        flipped = a.flip(dims=[dim])
        if op == "cumsum":
            return flipped.cumsum(dim=dim).flip(dims=[dim])
        return flipped.cummax(dim=dim).values.flip(dims=[dim])
    if op == "cumsum":
        return a.cumsum(dim=dim)
    return a.cummax(dim=dim).values


@pytest.mark.parametrize("target,execution_backend", _TARGETS)
@pytest.mark.parametrize("op", ["cumsum", "cummax"])
@pytest.mark.parametrize("reverse", [False, True])
def test_cpu_scan_1d_in_place(target, execution_backend, op, reverse):
    n = len(_INPUT)

    @T.prim_func
    def main(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
        with T.Kernel(1):
            buf = T.alloc_local((n,), T.float32)
            for i in T.serial(n):
                buf[i] = A[i]
            if op == "cumsum":
                T.cumsum(buf, dim=0, reverse=reverse)
            else:
                T.cummax(buf, dim=0, reverse=reverse)
            for i in T.serial(n):
                B[i] = buf[i]

    kernel = _compile(main, target, execution_backend)
    torch.testing.assert_close(kernel(_INPUT), _scan_reference(_INPUT, op, 0, reverse))


@pytest.mark.parametrize("target,execution_backend", _TARGETS)
@pytest.mark.parametrize("op", ["cumsum", "cummax"])
def test_cpu_scan_dynamic_extent(target, execution_backend, op):
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
        with T.Kernel(1):
            for i in T.serial(n):
                B[i] = A[i]
            if op == "cumsum":
                T.cumsum(B, dim=0, reverse=True)
            else:
                T.cummax(B, dim=0, reverse=True)

    kernel = _compile(main, target, execution_backend)
    for size in (1, len(_INPUT)):
        a = _INPUT[:size]
        torch.testing.assert_close(kernel(a), _scan_reference(a, op, 0, True))


@pytest.mark.parametrize("target,execution_backend", _TARGETS)
@pytest.mark.parametrize("op", ["cumsum", "cummax"])
@pytest.mark.parametrize("dim", [0, 1])
def test_cpu_scan_strided_offset_out_of_place(target, execution_backend, op, dim):
    m, n, src_pitch, dst_pitch = 2, 3, 5, 6
    reverse = dim == 1

    @T.prim_func
    def main(A: T.Tensor((m, n), T.float32), B: T.Tensor((m, n), T.float32)):
        with T.Kernel(1):
            src = T.alloc_local((m, src_pitch), T.float32)
            dst = T.alloc_local((m, dst_pitch), T.float32)
            for i, j in T.grid(m, src_pitch):
                src[i, j] = -10.0
            for i, j in T.grid(m, dst_pitch):
                dst[i, j] = -20.0
            for i, j in T.grid(m, n):
                src[i, j + 1] = A[i, j]
            if op == "cumsum":
                T.cumsum(src[0:m, 1 : n + 1], dst[0:m, 2 : n + 2], dim=dim, reverse=reverse)
            else:
                T.cummax(src[0:m, 1 : n + 1], dst[0:m, 2 : n + 2], dim=dim, reverse=reverse)
            for i, j in T.grid(m, n):
                B[i, j] = dst[i, j + 2]

    a = _INPUT[: m * n].reshape(m, n)
    kernel = _compile(main, target, execution_backend)
    torch.testing.assert_close(kernel(a), _scan_reference(a, op, dim, reverse))


@pytest.mark.parametrize("target,execution_backend", _TARGETS)
@pytest.mark.parametrize("op", ["cumsum", "cummax"])
def test_cpu_scan_3d_shared(target, execution_backend, op):
    shape = (2, 3, 4)

    @T.prim_func
    def main(A: T.Tensor(shape, T.float32), B: T.Tensor(shape, T.float32)):
        with T.Kernel(1):
            buf = T.alloc_shared(shape, T.float32)
            for i, j, k in T.grid(*shape):
                buf[i, j, k] = A[i, j, k]
            if op == "cumsum":
                T.cumsum(buf, dim=1, reverse=True)
            else:
                T.cummax(buf, dim=1, reverse=True)
            for i, j, k in T.grid(*shape):
                B[i, j, k] = buf[i, j, k]

    a = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(shape) - 5
    kernel = _compile(main, target, execution_backend)
    torch.testing.assert_close(kernel(a), _scan_reference(a, op, 1, True))


@pytest.mark.parametrize("op", ["cumsum", "cummax"])
def test_cpu_scan_empty_region(op):
    n = 4

    @T.prim_func
    def main(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
        with T.Kernel(1):
            buf = T.alloc_local((n,), T.float32)
            for i in T.serial(n):
                buf[i] = A[i]
            if op == "cumsum":
                T.cumsum(buf[0:0], dim=0)
            else:
                T.cummax(buf[0:0], dim=0)
            for i in T.serial(n):
                B[i] = buf[i]

    a = _INPUT[:n]
    kernel = _compile(main, "c", "cython")
    torch.testing.assert_close(kernel(a), a)


def test_cpu_cumsum_casts_to_destination_dtype():
    n = 4

    @T.prim_func
    def main(A: T.Tensor((n,), T.int32), B: T.Tensor((n,), T.float32)):
        with T.Kernel(1):
            src = T.alloc_local((n,), T.int32)
            dst = T.alloc_local((n,), T.float32)
            for i in T.serial(n):
                src[i] = A[i]
            T.cumsum(src, dst, dim=0)
            for i in T.serial(n):
                B[i] = dst[i]

    a = torch.tensor([1, -2, 3, 4], dtype=torch.int32)
    kernel = _compile(main, "c", "cython")
    torch.testing.assert_close(kernel(a), a.cumsum(dim=0).to(torch.float32))


def test_cpu_scan_rejects_non_identical_regions_on_same_buffer():
    n = 8

    @T.prim_func
    def main(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
        with T.Kernel(1):
            buf = T.alloc_local((n,), T.float32)
            for i in T.serial(n):
                buf[i] = A[i]
            T.cumsum(buf[0:4], buf[4:8], dim=0)
            for i in T.serial(n):
                B[i] = buf[i]

    with pytest.raises(Exception, match="non-identical source/destination regions"):
        _compile(main, "c", "cython")
