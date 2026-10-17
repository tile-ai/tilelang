from fractions import Fraction

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

pytest.importorskip("cutlass")


def test_truncdiv_python_integers():
    from tilelang.contrib.cutedsl.math import truncdiv

    for a in [-(2**63), -7, -1, 0, 1, 7, 2**63 - 1]:
        for b in [-3, 3]:
            assert truncdiv(a, b) == int(Fraction(a, b))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("target", ["cuda", "cutedsl"])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64"])
@pytest.mark.parametrize("lanes", [1, 4])
def test_integer_division_rounding(target, dtype, lanes):
    n = 32 * lanes

    @T.prim_func
    def kernel(A: T.Tensor((n,), dtype), C: T.Tensor((5, n), dtype)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(n):
                C[0, i] = T.truncdiv(A[i], 3)
                C[1, i] = T.floordiv(A[i], 3)
                C[2, i] = T.ceildiv(A[i], 3)
                C[3, i] = T.truncmod(A[i], 3)
                C[4, i] = T.floormod(A[i], 3)

    a = torch.tensor([-7, -5, -1, 0, 1, 3, 5, 7] * (n // 8), dtype=getattr(torch, dtype), device="cuda")
    result = tilelang.compile(kernel, target=target, out_idx=-1)(a)
    expected = torch.stack(
        [
            torch.div(a, 3, rounding_mode="trunc"),
            torch.div(a, 3, rounding_mode="floor"),
            torch.div(a + 2, 3, rounding_mode="floor"),
            torch.fmod(a, 3),
            torch.remainder(a, 3),
        ]
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(result[0] * 3 + result[3], a, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64"])
@pytest.mark.parametrize("lanes", [1, 4])
def test_truncdiv_runtime_divisors_and_extrema(dtype, lanes):
    torch_dtype = getattr(torch, dtype)
    info = torch.iinfo(torch_dtype)
    values = [-7, 7, -7, info.min, info.max, info.min, info.max - 2, 0] * (4 * lanes)
    divisors = [3, -3, -3, 3, 3, -3, -7, -1] * (4 * lanes)
    n = len(values)

    @T.prim_func
    def kernel(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((2, n), dtype)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(n):
                C[0, i] = T.truncdiv(A[i], B[i])
                C[1, i] = T.truncmod(A[i], B[i])

    a = torch.tensor(values, dtype=torch_dtype, device="cuda")
    b = torch.tensor(divisors, dtype=torch_dtype, device="cuda")
    quotients, remainders = tilelang.compile(kernel, target="cutedsl", out_idx=-1)(a, b).tolist()
    for x, y, q, r in zip(values, divisors, quotients, remainders):
        expected = int(Fraction(x, y))
        assert q == expected, (x, y, q, expected)
        assert q * y + r == x


if __name__ == "__main__":
    tilelang.testing.main()
