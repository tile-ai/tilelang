"""SIMT GM atomics: contended updates and returned previous values."""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("name,initial", [("add", 7.0), ("max", -1.0), ("min", 256.0)])
@pytest.mark.parametrize("return_previous", [False, True], ids=["contended", "return-previous"])
def test_gm_atomic(name, initial, return_previous):
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
    tilelang.compile(main, target="ascend")(counter, previous)
    count = blocks * threads
    expected = {"add": initial + count * (count + 1) / 2, "max": float(count), "min": 1.0}[name]
    assert counter.item() == expected
    assert previous.item() == (initial if return_previous else -999)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", ["int32", "uint32", "int64", "uint64"])
@pytest.mark.parametrize("name", ["add", "min", "max"])
def test_pto_integer_atomic_previous(dtype, name):
    operation = getattr(T, "atomic_" + name)
    signed = dtype.startswith("int")
    initial = (1 << 40) if "64" in dtype else 1024
    if not signed:
        initial = (1 << (63 if "64" in dtype else 31)) + 1024
    value = -7 if signed else 7

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
