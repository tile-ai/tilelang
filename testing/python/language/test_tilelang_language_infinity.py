import torch
import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.jit(out_idx=-1)
def get_inf_kernel(dtype: str):
    @T.prim_func
    def main(A: T.Tensor((32,), dtype)):
        with T.Kernel(1, threads=32):
            T.fill(A, T.infinity(dtype))

    return main


def _test_infinity(dtype: str):
    kernel = get_inf_kernel(dtype)
    output = kernel()

    assert torch.all(output == torch.inf), f"check failed for {dtype=}"


@tilelang.testing.requires_cuda
def test_infinity():
    _test_infinity(T.float16)
    _test_infinity(T.bfloat16)
    _test_infinity(T.float32)
    _test_infinity(T.float64)
    _test_infinity(T.float8_e5m2)


@tilelang.jit(out_idx=[1])
def cast_to_fp8_kernel(dtype: str):
    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), C: T.Tensor((8,), dtype)):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(8):
                C[i] = T.cast(A[i], dtype)

    return main


# Over-range fp32, the +/-inf endpoints, and the largest finite e5m2 value.
_OVER_RANGE = [65504.0, 1e5, 1e6, 120000.0, float("inf"), -1e5, float("-inf"), 57344.0]


@tilelang.testing.requires_cuda
def test_cast_to_e5m2_reaches_infinity():
    """A type that can hold an infinity has to produce one from an over-range value.

    `T.infinity("float8_e5m2")` stores an infinity and `torch.float8_e5m2` converts
    an over-range fp32 to one, but the cast between them clamped to the largest
    finite value 57344 -- including for `inf` itself, which needs no rounding.
    """
    values = torch.tensor(_OVER_RANGE, dtype=torch.float32, device="cuda")
    out = cast_to_fp8_kernel("float8_e5m2")(values)

    torch.testing.assert_close(out.view(torch.uint8), values.to(torch.float8_e5m2).view(torch.uint8), rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_cast_to_e4m3fn_still_saturates():
    """Control: e4m3fn has no infinity encoding, so over-range still clamps."""

    values = torch.tensor(_OVER_RANGE, dtype=torch.float32, device="cuda")
    out = cast_to_fp8_kernel("float8_e4m3fn")(values)

    torch.testing.assert_close(out.view(torch.uint8), values.to(torch.float8_e4m3fn).view(torch.uint8), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
