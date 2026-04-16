import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit
def _issue2042_vector_cast(M: int):
    assert M % 256 == 0

    @T.prim_func
    def kernel(
        x: T.Tensor[(M,), "float32"],  # noqa: F821
        y: T.Tensor[(M,), "float8_e8m0fnu"],  # noqa: F821
    ):
        with T.Kernel(1, threads=128):
            T.copy(x, y)

    return kernel


@tilelang.jit
def _issue2042_scalar_cast(M: int):
    @T.prim_func
    def kernel(
        x: T.Tensor[(M,), "float32"],  # noqa: F821
        y: T.Tensor[(M,), "float8_e8m0fnu"],  # noqa: F821
    ):
        with T.Kernel(M, threads=1) as pid:
            y[pid] = T.Cast("float8_e8m0fnu", x[pid])

    return kernel


def _make_issue2042_inputs(M: int) -> torch.Tensor:
    # Cover the cases where rounding-to-posinf diverges from PyTorch:
    # midpoint ties, sub-min values, negative inputs, and overflow.
    base = torch.tensor(
        [
            0.0,
            2**-128,
            2**-127,
            0.0056,
            0.0057,
            0.0061,
            0.49,
            0.5,
            0.51,
            0.74999,
            0.75,
            0.75001,
            1.49999,
            1.5,
            1.50001,
            2.99999,
            3.0,
            3.00001,
            -1.5,
            -3.0,
            2.5e38,
            3.0e38,
            float("inf"),
            float("nan"),
        ],
        device="cuda",
        dtype=torch.float32,
    )
    repeats = (M + base.numel() - 1) // base.numel()
    return base.repeat(repeats)[:M].contiguous()


def _raw_fp8_bytes(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.uint8)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 9)
def test_issue_2042_fp8_e8m0_cast_matches_torch():
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch.float8_e8m0fnu requires torch >= 2.8")

    M = 256
    x = _make_issue2042_inputs(M)
    ref = x.to(torch.float8_e8m0fnu)

    vector_kernel = _issue2042_vector_cast(M)
    scalar_kernel = _issue2042_scalar_cast(M)

    vector_code = vector_kernel.get_kernel_source()
    scalar_code = scalar_kernel.get_kernel_source()
    assert "__tl_cvt_float2_to_e8m0x2" in vector_code
    assert "__tl_cvt_float_to_e8m0" in scalar_code

    y_vector = torch.empty_like(ref)
    y_scalar = torch.empty_like(ref)
    vector_kernel(x, y_vector)
    scalar_kernel(x, y_scalar)

    assert torch.equal(_raw_fp8_bytes(ref), _raw_fp8_bytes(y_vector))
    assert torch.equal(_raw_fp8_bytes(ref), _raw_fp8_bytes(y_scalar))


if __name__ == "__main__":
    tilelang.testing.main()
