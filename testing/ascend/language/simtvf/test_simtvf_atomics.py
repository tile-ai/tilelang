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
