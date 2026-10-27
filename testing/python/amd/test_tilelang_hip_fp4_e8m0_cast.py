"""Numeric casts to the software FP4 E2M1 and E8M0 types of the HIP templates.

``T.cast(x, T.float4_e2m1fn)`` is printed as ``(fp4_e2_t)x``. Without a float
constructor that cast truncated ``x`` to an integer and kept it as raw E2M1
bits (1.0 came back as 0.5, 6.0 as 4.0); ``T.float8_e8m0fnu`` had no type at
all. The FP4 types exist on gfx950, E8M0 on every ROCm target.
"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.language as T

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _arch():
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def _roundtrip(n, dtype):
    @T.prim_func
    def kernel(X: T.Tensor((n,), T.float32), Y: T.Tensor((n,), T.float32)):
        with T.Kernel(1, threads=n):
            for i in T.Parallel(n):
                Y[i] = T.cast(T.cast(X[i], dtype), T.float32)

    return tilelang.compile(kernel, target={"kind": "hip", "mcpu": _arch()})


def _reference_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest E2M1 value, ties to the even code, saturate at 6; NaN of either sign gives +6."""
    values = torch.tensor(FP4_VALUES, dtype=torch.float32)
    mag = x.abs().clamp(max=6.0)
    diff = (mag[:, None] - values[None, :]).abs()
    best = diff.min(dim=1, keepdim=True).values
    # Codes are ordered by value; among equally close codes take the even one.
    candidates = diff == best
    even = candidates & (torch.arange(8) % 2 == 0)
    code = torch.where(even.any(dim=1), even.int().argmax(dim=1), candidates.int().argmax(dim=1))
    return torch.where(x.isnan(), torch.full_like(x, 6.0), torch.copysign(values[code], x))


@tilelang.testing.requires_cdna
def test_float_to_fp4_rounds_to_nearest_even():
    if _arch() != "gfx950":
        pytest.skip("the FP4 E2M1 types exist on gfx950 only")
    ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    exact = list(FP4_VALUES)
    other = [0.1, 0.3, 0.6, 0.9, 1.1, 2.2, 2.9, 4.4, 5.5, 7.0, 100.0]
    vals = ties + exact + other
    nans = [float("nan"), -float("nan")]
    x = torch.tensor(vals + [-v for v in vals] + nans + [0.0] * (64 - 2 * len(vals) - 2), dtype=torch.float32)

    y = torch.empty_like(x, device="cuda")
    _roundtrip(x.numel(), T.float4_e2m1fn)(x.cuda(), y)

    torch.testing.assert_close(y.cpu(), _reference_fp4(x), rtol=0, atol=0)


@tilelang.testing.requires_cdna
def test_float_to_e8m0_rounds_up_to_a_power_of_two():
    # As __nv_fp8_e8m0(float) (cudaRoundPosInf, __NV_SATFINITE): powers of two
    # are exact, other magnitudes round up, NaN stays NaN.
    x = torch.tensor([2.0**e for e in range(-20, 21)] + [3.0, 0.75, 1000.0, float("nan")], dtype=torch.float32)
    expected = torch.where(x.isnan(), x, torch.exp2(torch.ceil(torch.log2(x))))
    # E8M0 has no sign. Results below 2^-127 (zero, subnormals up to 2^-127) clip
    # to it, and results above 2^127 (large finite values, infinities) saturate to it.
    edge = [0.0, -0.0, 2.0**-149, 2.0**-127 + 2.0**-149, -3.0, 1.5 * 2.0**127, float("inf"), -float("inf")]
    edge_expected = [2.0**-127] * 3 + [2.0**-126, 4.0] + [2.0**127] * 3
    x = torch.cat([x, torch.tensor(edge, dtype=torch.float32)])
    expected = torch.cat([expected, torch.tensor(edge_expected, dtype=torch.float32)])

    y = torch.empty_like(x, device="cuda")
    _roundtrip(x.numel(), T.float8_e8m0fnu)(x.cuda(), y)

    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0, equal_nan=True)


if __name__ == "__main__":
    tilelang.testing.main()
