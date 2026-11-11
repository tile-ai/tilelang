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
        ("add", "int64", 1 << 40, -7),
        ("min", "int32", 1024, -7),
        ("max", "uint32", (1 << 31) + 1024, 7),
        ("max", "uint64", (1 << 63) + 1024, 7),
    ],
)
def test_pto_integer_atomic_previous(name, dtype, initial, value):
    operation = getattr(T, "atomic_" + name)

    @T.prim_func
    def main(C: T.Tensor((1,), dtype), Previous: T.Tensor((1,), dtype)):
        with T.Kernel(1), T.SimtVF(threads=1):
            for _tx in T.Parallel(1):
                Previous[0] = operation(C[0], T.cast(value, dtype), return_prev=True)

    counter = torch.tensor([initial], dtype=getattr(torch, dtype), device="npu")
    # NPU zeros_like does not support uint32/uint64; upload CPU data instead.
    previous = torch.tensor([0], dtype=getattr(torch, dtype), device="npu")
    tilelang.compile(main, target="pto")(counter, previous)
    expected = {"add": initial + value, "min": min(initial, value), "max": max(initial, value)}[name]
    assert counter.cpu().item() == expected
    assert previous.cpu().item() == initial


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


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "target",
    ["ascend", pytest.param("pto", marks=pytest.mark.pto)],
)
@pytest.mark.parametrize(
    "value_dtype,value,expected",
    [("uint32", 0xFFFFFFFF, 7 + 0xFFFFFFFF), ("float32", 3.5, 10)],
)
def test_atomic_add_converts_operand(value_dtype, value, expected, target):
    @T.prim_func
    def main(C: T.Tensor((1,), "int64"), V: T.Tensor((1,), value_dtype)):
        with T.Kernel(1):
            value_ub = T.alloc_shared((1,), value_dtype)
            T.copy(V, value_ub)
            with T.SimtVF(threads=1):
                T.atomic_add(C[0], value_ub[0])

    counter = torch.tensor([7], dtype=torch.int64, device="npu")
    operand = torch.tensor([value], dtype=getattr(torch, value_dtype), device="npu")
    tilelang.compile(main, target=target)(counter, operand)
    assert counter.cpu().item() == expected
