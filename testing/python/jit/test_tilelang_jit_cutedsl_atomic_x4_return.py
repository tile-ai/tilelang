"""CuTeDSL x4 atomic add returning the previous values, including 16-bit forms.

float32 uses one ``atom.global.add.v4.f32``; float16/bfloat16 have no x4 atomic
and are two packed per-pair atomics, mirroring the CUDA backend. Writing 16 bytes
into the 8-byte destination of a 16-bit type would corrupt the neighbouring
elements, so both the old and the new values are checked here.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _require_cutedsl():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")


@tilelang.jit(target="cutedsl")
def atomic_addx4_prev_kernel(dtype, guard=0):
    """Atomic x4 add with return_prev; ``guard`` pads the buffer so an over-wide
    store would show up as a wrong tail value."""

    @T.prim_func
    def kernel(
        Dst: T.Tensor((4 + guard,), dtype),
        Val: T.Tensor((4,), dtype),
        Prev: T.Tensor((4,), dtype),
    ):
        with T.Kernel(1, threads=1):
            Prev[0:4] = T.atomic_addx4(Dst[0], Val[0], return_prev=True)

    return kernel


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_cutedsl_atomic_addx4_return_prev(dtype):
    _require_cutedsl()
    torch_dtype = getattr(torch, dtype)
    dst = torch.arange(1, 9, device="cuda", dtype=torch_dtype)
    val = (torch.arange(1, 5, device="cuda", dtype=torch_dtype) * 10).to(torch_dtype)
    prev = torch.zeros(4, device="cuda", dtype=torch_dtype)
    tail_before = dst[4:].clone()

    atomic_addx4_prev_kernel(dtype, guard=4)(dst, val, prev)

    expected_prev = torch.arange(1, 5, device="cuda", dtype=torch_dtype)
    torch.testing.assert_close(prev, expected_prev, atol=0, rtol=0)
    expected_dst = torch.cat([expected_prev + val, tail_before])
    torch.testing.assert_close(dst, expected_dst, atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
