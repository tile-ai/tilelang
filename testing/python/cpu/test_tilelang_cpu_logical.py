"""Serial CPU lowering for T.any_of and T.all_of."""

import pytest
import torch

import tilelang
import tilelang.cpu.language as T


_TARGETS = [("c", "cython")]


def _compile(func, target, execution_backend, out_idx=-1):
    return tilelang.compile(
        func,
        target=target,
        target_host="c" if target == "c" else None,
        execution_backend=execution_backend,
        out_idx=out_idx,
    )


def _emit_logical(op, buf):
    return T.any_of(buf) if op == "any_of" else T.all_of(buf)


def _reference(a, op):
    return torch.any(a) if op == "any_of" else torch.all(a)


@pytest.mark.parametrize("target,execution_backend", _TARGETS)
@pytest.mark.parametrize("op", ["any_of", "all_of"])
@pytest.mark.parametrize(
    "values",
    [
        [False, False, False, False, False, False, False, False],
        [False, False, False, True, False, False, False, False],
        [True, True, True, True, True, True, True, True],
    ],
    ids=["all_false", "single_true", "all_true"],
)
def test_cpu_logical_full_buffer(target, execution_backend, op, values):
    shape = (2, 4)

    @T.prim_func
    def main(A: T.Tensor(shape, "bool"), B: T.Tensor((1,), "bool")):
        with T.Kernel(1):
            buf = T.alloc_local(shape, "bool")
            for i, j in T.grid(*shape):
                buf[i, j] = A[i, j]
            if op == "any_of":
                if T.any_of(buf):
                    B[0] = True
                else:
                    B[0] = False
            else:
                if T.all_of(buf):
                    B[0] = True
                else:
                    B[0] = False

    a = torch.tensor(values, dtype=torch.bool).reshape(shape)
    kernel = _compile(main, target, execution_backend)
    assert kernel(a).item() == _reference(a, op).item()


@pytest.mark.parametrize("target,execution_backend", _TARGETS)
@pytest.mark.parametrize("op", ["any_of", "all_of"])
def test_cpu_logical_last_dimension_region(target, execution_backend, op):
    n = 8

    @T.prim_func
    def main(A: T.Tensor((n,), "bool"), B: T.Tensor((1,), "bool")):
        with T.Kernel(1):
            buf = T.alloc_local((n,), "bool")
            for i in T.serial(n):
                buf[i] = A[i]
            B[0] = _emit_logical(op, buf[2:5])

    a = torch.tensor([False, False, True, True, True, False, False, False])
    kernel = _compile(main, target, execution_backend)
    assert kernel(a).item() == _reference(a[2:5], op).item()


@pytest.mark.parametrize("target,execution_backend", _TARGETS)
@pytest.mark.parametrize("op", ["any_of", "all_of"])
def test_cpu_logical_shared_buffer(target, execution_backend, op):
    n = 4

    @T.prim_func
    def main(A: T.Tensor((n,), "bool"), B: T.Tensor((1,), "bool")):
        with T.Kernel(1):
            buf = T.alloc_shared((n,), "bool")
            for i in T.serial(n):
                buf[i] = A[i]
            B[0] = _emit_logical(op, buf)

    a = torch.tensor([False, True, True, True])
    kernel = _compile(main, target, execution_backend)
    assert kernel(a).item() == _reference(a, op).item()


@pytest.mark.parametrize("target,execution_backend", _TARGETS)
@pytest.mark.parametrize("op", ["any_of", "all_of"])
def test_cpu_logical_dynamic_extent(target, execution_backend, op):
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.Tensor((n,), "bool"), B: T.Tensor((1,), "bool")):
        with T.Kernel(1):
            B[0] = _emit_logical(op, A)

    kernel = _compile(main, target, execution_backend)
    for a in (
        torch.tensor([False]),
        torch.tensor([False, True, False, True]),
        torch.tensor([True, True, True]),
    ):
        assert kernel(a).item() == _reference(a, op).item()


@pytest.mark.parametrize("target,execution_backend", _TARGETS)
@pytest.mark.parametrize("op", ["any_of", "all_of"])
def test_cpu_logical_empty_region(target, execution_backend, op):
    n = 4

    @T.prim_func
    def main(A: T.Tensor((n,), "bool"), B: T.Tensor((1,), "bool")):
        with T.Kernel(1):
            buf = T.alloc_local((n,), "bool")
            for i in T.serial(n):
                buf[i] = A[i]
            B[0] = _emit_logical(op, buf[0:0])

    a = torch.ones(n, dtype=torch.bool)
    kernel = _compile(main, target, execution_backend)
    assert kernel(a).item() == (op == "all_of")


@pytest.mark.parametrize("target,execution_backend", _TARGETS)
@pytest.mark.parametrize("op", ["any_of", "all_of"])
def test_cpu_logical_numeric_truthiness(target, execution_backend, op):
    n = 4

    @T.prim_func
    def main(A: T.Tensor((n,), "int32"), B: T.Tensor((1,), "bool")):
        with T.Kernel(1):
            buf = T.alloc_local((n,), "int32")
            for i in T.serial(n):
                buf[i] = A[i]
            B[0] = _emit_logical(op, buf)

    a = torch.tensor([0, 0, 2, 0], dtype=torch.int32)
    kernel = _compile(main, target, execution_backend)
    assert kernel(a).item() == _reference(a, op).item()


@pytest.mark.parametrize("target,execution_backend", _TARGETS)
def test_cpu_logical_while_condition_is_reevaluated(target, execution_backend):
    @T.prim_func
    def main(B: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            buf = T.alloc_local((1,), "bool")
            count = T.alloc_local((1,), "int32")
            buf[0] = True
            count[0] = 0
            with T.While(T.any_of(buf) and count[0] < 3):
                buf[0] = False
                count[0] += 1
            B[0] = count[0]

    kernel = _compile(main, target, execution_backend, out_idx=None)
    b = torch.empty(1, dtype=torch.int32, device="cpu")
    kernel(b)
    assert b.item() == 1


@pytest.mark.parametrize("target,execution_backend", _TARGETS)
@pytest.mark.parametrize("op", ["any_of", "all_of"])
def test_cpu_logical_single_element(target, execution_backend, op):
    n = 1

    @T.prim_func
    def main(A: T.Tensor((n,), "bool"), B: T.Tensor((1,), "bool")):
        with T.Kernel(1):
            buf = T.alloc_local((n,), "bool")
            buf[0] = A[0]
            B[0] = _emit_logical(op, buf)

    a = torch.tensor([True])
    kernel = _compile(main, target, execution_backend)
    assert kernel(a).item() is True


@pytest.mark.parametrize("target,execution_backend", _TARGETS)
@pytest.mark.parametrize("op", ["any_of", "all_of"])
def test_cpu_logical_single_element_inside_grid(target, execution_backend, op):
    n = 4

    @T.prim_func
    def main(A: T.Tensor((n,), "bool"), B: T.Tensor((n,), "bool")):
        with T.Kernel(n) as bx:
            buf = T.alloc_local((1,), "bool")
            buf[0] = A[bx]
            B[bx] = _emit_logical(op, buf)

    a = torch.ones(n, dtype=torch.bool)
    kernel = _compile(main, target, execution_backend)
    torch.testing.assert_close(kernel(a), a)


@pytest.mark.parametrize("target,execution_backend", _TARGETS)
@pytest.mark.parametrize("op", ["any_of", "all_of"])
def test_cpu_logical_static_empty_buffer(target, execution_backend, op):
    @T.prim_func
    def main(A: T.Tensor((0,), "bool"), B: T.Tensor((1,), "bool")):
        with T.Kernel(1):
            B[0] = _emit_logical(op, A)

    kernel = _compile(main, target, execution_backend)
    assert kernel(torch.empty(0, dtype=torch.bool)).item() == (op == "all_of")


@pytest.mark.parametrize("target,execution_backend", _TARGETS)
@pytest.mark.parametrize("op", ["any_of", "all_of"])
def test_cpu_logical_nan_truthiness(target, execution_backend, op):
    n = 4

    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((1,), "bool")):
        with T.Kernel(1):
            B[0] = _emit_logical(op, A)

    a = torch.full((n,), float("nan"), dtype=torch.float32)
    kernel = _compile(main, target, execution_backend)
    assert kernel(a).item() is True
