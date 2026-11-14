"""RNG vector-local lane offsets in the PTO codegen path."""

import re

import pytest
import torch

import tilelang
import tilelang.ascend.language as T
import tilelang.testing


def _rng_vector_local(distribution, unrolled):
    @T.prim_func
    def main(reference: T.Tensor((4,), T.float32), result: T.Tensor((4,), T.float32)):
        with T.Kernel(1):
            reference_ub = T.alloc_shared((4,), T.float32)
            result_ub = T.alloc_shared((4,), T.float32)
            with T.SimtVF(threads=1):
                vals = T.alloc_local((2,), "float32x2")
                T.rng_init(42, seq=7, off=12)
                for i in T.serial(4):
                    reference_ub[i] = T.rng_rand_float(dist=distribution)
                T.rng_init(42, seq=7, off=12)
                if unrolled:
                    for slot in T.unroll(2):
                        vals[slot] = T.Broadcast(T.rng_rand_float(dist=distribution), 2)
                else:
                    vals[0] = T.Broadcast(T.rng_rand_float(dist=distribution), 2)
                    vals[1] = T.Broadcast(T.rng_rand_float(dist=distribution), 2)
                result_ub[T.Ramp(0, 1, 2)] = vals[0]
                result_ub[T.Ramp(2, 1, 2)] = vals[1]
            T.copy(reference_ub, reference)
            T.copy(result_ub, result)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("distribution", ["uniform", "normal"])
@pytest.mark.parametrize("unrolled", [False, True], ids=["literal", "unrolled"])
def test_pto_rng_vector_local_lane_offsets(distribution, unrolled):
    source = tilelang.lower(_rng_vector_local(distribution, unrolled), target="pto", enable_device_compile=False).kernel_source
    assert "vals = pto.alloc_buffer((4,), pto.f32)" in source
    assert "vals[" not in source
    if unrolled:
        assert ", vals, slot * 2)" in source
        assert re.search(r", vals, \(?slot \* 2\)? \+ 1\)", source)
    else:
        for offset in range(4):
            assert f", vals, {offset})" in source


@tilelang.testing.requires_ascend
@pytest.mark.pto
@pytest.mark.parametrize("distribution", ["uniform", "normal"])
@pytest.mark.parametrize("unrolled", [False, True], ids=["literal", "unrolled"])
def test_pto_rng_vector_local_advances_each_lane(distribution, unrolled):
    kernel = tilelang.compile(_rng_vector_local(distribution, unrolled), target="pto", out_idx=[0, 1])
    reference, result = kernel()
    torch.npu.synchronize()
    torch.testing.assert_close(result, reference, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
