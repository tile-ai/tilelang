"""SIMT GM atomics: contended updates and returned previous values."""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "target",
    ["ascend", pytest.param("pto", marks=pytest.mark.pto)],
)
@pytest.mark.parametrize("name,initial", [("add", 7.0), ("max", -1.0), ("min", 256.0)])
@pytest.mark.parametrize("return_previous", [False, True], ids=["contended", "return-previous"])
def test_gm_atomic(name, initial, return_previous, target):
    operation = getattr(T, "atomic_" + name)
    blocks, threads = (1, 1) if return_previous else (2, 64)

    @T.prim_func
    def main(C: T.Tensor((1,), "float32"), Previous: T.Tensor((1,), "float32")):
        with T.Kernel(blocks) as bx, T.SimtVF(threads=threads):
            for tx in T.Parallel(threads):
                value = T.float32(bx * threads + tx + 1)
                if return_previous:
                    Previous[0] = operation(C[0], value, return_prev=True)
                else:
                    operation(C[0], value)

    counter = torch.full((1,), initial, device="npu")
    previous = torch.full_like(counter, -999)
    tilelang.compile(main, target=target)(counter, previous)
    count = blocks * threads
    expected = {"add": initial + count * (count + 1) / 2, "max": float(count), "min": 1.0}[name]
    assert counter.item() == expected
    assert previous.item() == (initial if return_previous else -999)


@tilelang.testing.requires_ascend
@pytest.mark.pto
@pytest.mark.parametrize(
    "name,dtype,initial,value",
    [
        ("add", "int64", 7, 3),
        ("max", "uint32", 7, 2**31 + 5),
        ("min", "int32", 7, -5),
    ],
)
def test_pto_atomic_integer(name, dtype, initial, value):
    operation = getattr(T, "atomic_" + name)

    @T.prim_func
    def main(C: T.Tensor((1,), dtype)):
        with T.Kernel(1), T.SimtVF(threads=1):
            operation(C[0], getattr(T, dtype)(value))

    counter = torch.tensor([initial], dtype=getattr(torch, dtype), device="npu")
    tilelang.compile(main, target="pto")(counter)
    expected = {"add": initial + value, "max": max(initial, value), "min": min(initial, value)}[name]
    assert counter.item() == expected


@tilelang.testing.requires_ascend
@pytest.mark.pto
def test_pto_atomic_max_uses_pointer_signedness():
    @T.prim_func
    def main(C: T.Tensor((1,), "uint32"), V: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            value_ub = T.alloc_shared((1,), "int32")
            T.copy(V, value_ub)
            with T.SimtVF(threads=1):
                T.atomic_max(C[0], value_ub[0])

    counter = torch.tensor([7], dtype=torch.uint32, device="npu")
    value = torch.tensor([-1], dtype=torch.int32, device="npu")
    tilelang.compile(main, target="pto")(counter, value)
    assert counter.item() == 2**32 - 1
