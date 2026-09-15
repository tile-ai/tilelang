"""Loop induction bindings must not assign to older mutable scalar bindings."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def make_kernel(kind, live=False):
    @T.prim_func
    def kernel(A: T.Tensor((5,), "int32")):
        with T.Kernel(1, threads=1):
            if live:
                d = T.alloc_var("int32", init=19)
                saved = d
            else:
                for j in T.serial(2):
                    d = T.alloc_var("int32", init=j)
                    A[j] = d
            if kind == "serial":
                for d in T.serial(4):
                    A[d] = d
            elif kind == "parallel":
                for d in T.Parallel(4):
                    A[d] = d
            elif kind == "unroll":
                for d in T.unroll(4):
                    A[d] = d
            elif kind == "stepped":
                for d in T.serial(0, 4, step=2):
                    A[d] = d
            else:
                for d, j in T.grid(2, 2):
                    A[d * 2 + j] = d * 2 + j
            if live:
                A[4] = saved

    return kernel


@pytest.mark.parametrize("kind", ["serial", "parallel", "unroll", "stepped", "grid"])
@pytest.mark.parametrize("live", [False, True])
def test_loop_target_is_not_buffer_store(kind, live):
    kernel = make_kernel(kind, live)
    stores = []
    tvm.tirx.stmt_functor.post_order_visit(
        kernel.body, lambda node: stores.append(node) if isinstance(node, tvm.tirx.BufferStore) else None
    )
    # Only initialization writes d; introducing another loop cannot write it.
    local_stores = [node for node in stores if node.buffer.scope() == "local.var"]
    assert len(local_stores) == 1


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("kind", ["serial", "parallel", "unroll", "stepped", "grid"])
@pytest.mark.parametrize("live", [False, True])
def test_loop_target_execution(kind, live):
    kernel = tilelang.compile(make_kernel(kind, live))
    result = torch.full((5,), -1, dtype=torch.int32, device="cuda")
    kernel(result)
    expected = [0, 1, 2, 3, 19 if live else -1]
    if kind == "stepped":
        expected[1] = -1 if live else 1
        expected[3] = -1
    torch.testing.assert_close(result.cpu(), torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0)


def test_loop_target_does_not_disable_scope_checks():
    with pytest.raises(RuntimeError, match="outside its defining region"):

        @T.prim_func
        def kernel(A: T.Tensor((1,), "int32")):
            with T.Kernel(1, threads=1):
                for d in T.serial(2):
                    A[0] = d
                A[0] = d


def test_expired_local_read_still_rejected():
    with pytest.raises(RuntimeError, match="outside its defining region"):

        @T.prim_func
        def kernel(A: T.Tensor((1,), "int32")):
            with T.Kernel(1, threads=1):
                for j in T.serial(2):
                    value = T.alloc_var("int32", init=j)
                A[0] = value


@tilelang.testing.requires_cuda
def test_regular_mutable_assignment_and_tuple_swap():
    @T.prim_func
    def kernel(A: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=1):
            a = T.alloc_var("int32", init=1)
            b = T.alloc_var("int32", init=2)
            for j in T.serial(3):
                a = a + j
            a, b = b, a
            A[0] = a
            A[1] = b

    result = torch.empty((2,), device="cuda", dtype=torch.int32)
    tilelang.compile(kernel)(result)
    torch.testing.assert_close(result.cpu(), torch.tensor([2, 4], dtype=torch.int32), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
