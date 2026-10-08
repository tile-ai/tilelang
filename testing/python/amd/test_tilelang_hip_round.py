import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _round_kernel(dtype, lanes):
    size = 64 * lanes

    @T.prim_func
    def main(
        A: T.Tensor((size,), dtype),
        B: T.Tensor((size,), dtype),
        C: T.Tensor((size,), dtype),
    ):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            for lane in T.vectorized(lanes):
                index = tx * lanes + lane
                B[index] = T.round(A[index])
                C[index] = T.round(A[index], rounding_mode="ties-to-even")

    return main


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype,intrinsic", [("float32", "nearbyintf"), ("float64", "nearbyint")])
@pytest.mark.parametrize("lanes", [1, 2])
def test_round_ties_to_even_codegen(dtype, intrinsic, lanes):
    if tvm.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True) is None:
        pytest.skip("TileLang HIP codegen is not enabled")

    target = tvm.target.Target({"kind": "hip", "mcpu": "gfx942"})
    with tvm.transform.PassContext(), target:
        artifact = tilelang.lower(_round_kernel(dtype, lanes), target=target, enable_device_compile=False)

    source = artifact.kernel_source
    assert f"{intrinsic}(" in source
    assert not re.search(r"\broundf?\(", source)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("lanes", [1, 2])
def test_round_ties_to_even(dtype, lanes):
    # Runtime inputs keep constant folding from hiding the device intrinsic.
    values = [
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        3.5,
        -3.5,
        4.5,
        -4.5,
        0.25,
        -0.25,
        0.75,
        -0.75,
        2.5 - 2**-20,
        2.5 + 2**-20,
        -2.5 - 2**-20,
        -2.5 + 2**-20,
        2**40 + 1.5,
        -(2**40 + 1.5),
        float("inf"),
        float("-inf"),
        float("nan"),
    ]
    size = 64 * lanes
    data = (values * ((size + len(values) - 1) // len(values)))[:size]
    source = torch.tensor(data, dtype=getattr(torch, dtype), device="cuda")
    kernel = tilelang.compile(_round_kernel(dtype, lanes), out_idx=[1, 2], target="hip")
    default, explicit = kernel(source)
    expected = torch.round(source)
    zeros = expected == 0

    for actual in (default, explicit):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


if __name__ == "__main__":
    tilelang.testing.main()
