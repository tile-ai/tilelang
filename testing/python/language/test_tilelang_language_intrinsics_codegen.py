import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
def test_language_ldg_codegen():
    N = 128

    @T.prim_func
    def main(
        x: T.Tensor((N,), T.float32),
        y: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(N, threads=32) as pid:
            # Explicitly request read-only cache load for x[pid]
            y[pid] = T.__ldg(x[pid]) + 1.0

    # Compile for CUDA and retrieve generated CUDA source
    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    src = kernel.get_kernel_source()
    print(src)
    # Assert that codegen uses __ldg on CUDA backend
    # We look for the intrinsic call with address-of argument
    assert "__ldg(" in src, "Expected __ldg call in generated CUDA source"
    assert "__ldg(&" in src or "__ldg(&(" in src, "Expected address-of form in __ldg call"


def run_ldg_roundtrip(dtype, N=128):
    @T.prim_func
    def main(
        x: T.Tensor((N,), dtype),
        y: T.Tensor((N,), dtype),
    ):
        with T.Kernel(N, threads=32) as pid:
            y[pid] = T.__ldg(x[pid])

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    # Match the address-of form emitted by CUDA codegen.
    assert "__ldg(&(" in kernel.get_kernel_source()

    x = torch.randn(N, device="cuda").to(dtype.as_torch())
    assert torch.equal(kernel(x), x)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16])
def test_language_ldg_roundtrip(dtype):
    run_ldg_roundtrip(dtype)


# (op, lowered h* intrinsic, torch reference, input index), in the row order
# half_math_kernel writes.
HALF_MATH_OPS = (
    ("exp", "hexp", torch.exp, 0),
    ("exp2", "hexp2", torch.exp2, 0),
    ("exp10", "hexp10", lambda x: torch.pow(10.0, x), 0),
    ("log", "hlog", torch.log, 0),
    ("log2", "hlog2", torch.log2, 0),
    ("log10", "hlog10", torch.log10, 0),
    ("sin", "hsin", torch.sin, 1),
    ("cos", "hcos", torch.cos, 1),
    ("floor", "hfloor", torch.floor, 1),
    ("ceil", "hceil", torch.ceil, 1),
    ("round", "hrint", torch.round, 1),
    ("trunc", "htrunc", torch.trunc, 1),
    ("sinh", "hsinh", torch.sinh, 1),
    ("cosh", "hcosh", torch.cosh, 1),
    ("tanh", "htanh", torch.tanh, 1),
    ("atan", "hatan", torch.atan, 1),
    ("erf", "herf", torch.erf, 1),
    ("nearbyint", "hnearbyint", torch.round, 1),
)


def half_math_kernel(dtype, N):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
        B: T.Tensor((len(HALF_MATH_OPS), N), dtype),
    ):
        with T.Kernel(1, threads=N):
            i = T.get_thread_binding()
            # Keep the temporaries: they copy-initialize, a store only assigns.
            exp_value = T.exp(A[i])
            exp2_value = T.exp2(A[i])
            exp10_value = T.exp10(A[i])
            log_value = T.log(A[i])
            log2_value = T.log2(A[i])
            log10_value = T.log10(A[i])
            sin_value = T.sin(C[i])
            cos_value = T.cos(C[i])
            floor_value = T.floor(C[i])
            ceil_value = T.ceil(C[i])
            round_value = T.round(C[i])
            trunc_value = T.trunc(C[i])
            sinh_value = T.sinh(C[i])
            cosh_value = T.cosh(C[i])
            tanh_value = T.tanh(C[i])
            atan_value = T.atan(C[i])
            erf_value = T.erf(C[i])
            nearbyint_value = T.nearbyint(C[i])
            B[0, i] = exp_value
            B[1, i] = exp2_value
            B[2, i] = exp10_value
            B[3, i] = log_value
            B[4, i] = log2_value
            B[5, i] = log10_value
            B[6, i] = sin_value
            B[7, i] = cos_value
            B[8, i] = floor_value
            B[9, i] = ceil_value
            B[10, i] = round_value
            B[11, i] = trunc_value
            B[12, i] = sinh_value
            B[13, i] = cosh_value
            B[14, i] = tanh_value
            B[15, i] = atan_value
            B[16, i] = erf_value
            B[17, i] = nearbyint_value

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16])
def test_half_math_intrinsics(dtype):
    N = 128
    kernel = tilelang.compile(half_math_kernel(dtype, N), target="cuda")
    body = kernel.get_kernel_source()
    body = body[body.rindex("__global__") :]

    torch_dtype = dtype.as_torch()
    # [0.3, 4.3) keeps exp10 finite in float16; the mixed-sign input avoids
    # integers and rounding ties.
    a = (torch.arange(N, device="cuda") * 0.03125 + 0.3).to(torch_dtype)
    c = ((torch.arange(N, device="cuda") - 63.5) * 0.0625).to(torch_dtype)
    b = torch.empty(len(HALF_MATH_OPS), N, device="cuda", dtype=torch_dtype)
    kernel(a, c, b)

    inputs = (a, c)
    for row, (name, intrinsic, ref_fn, which) in enumerate(HALF_MATH_OPS):
        assert body.count(f"{intrinsic}(") == 1
        ref = ref_fn(inputs[which].float()).to(torch_dtype)
        torch.testing.assert_close(
            b[row].float(),
            ref.float(),
            atol=2e-2,
            rtol=2e-2,
            msg=lambda base, name=name: f"T.{name} mismatch\n{base}",
        )


# Same bridging as HALF_MATH_OPS, but asin, acos, atanh and acosh have
# restricted domains, so this group takes inputs of its own. (op, lowered h*
# intrinsic, torch reference, input index), in the row order
# domain_math_kernel writes.
DOMAIN_MATH_OPS = (
    ("asin", "hasin", torch.asin, 0),
    ("acos", "hacos", torch.acos, 0),
    ("atanh", "hatanh", torch.atanh, 0),
    ("asinh", "hasinh", torch.asinh, 0),
    ("log1p", "hlog1p", torch.log1p, 0),
    ("acosh", "hacosh", torch.acosh, 1),
)


def domain_math_kernel(dtype, N):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
        B: T.Tensor((len(DOMAIN_MATH_OPS), N), dtype),
    ):
        with T.Kernel(1, threads=N):
            i = T.get_thread_binding()
            # Keep the temporaries: they copy-initialize, a store only assigns.
            asin_value = T.asin(A[i])
            acos_value = T.acos(A[i])
            atanh_value = T.atanh(A[i])
            asinh_value = T.asinh(A[i])
            log1p_value = T.log1p(A[i])
            acosh_value = T.acosh(C[i])
            B[0, i] = asin_value
            B[1, i] = acos_value
            B[2, i] = atanh_value
            B[3, i] = asinh_value
            B[4, i] = log1p_value
            B[5, i] = acosh_value

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16])
def test_domain_math_intrinsics(dtype):
    N = 128
    kernel = tilelang.compile(domain_math_kernel(dtype, N), target="cuda")
    body = kernel.get_kernel_source()
    body = body[body.rindex("__global__") :]

    torch_dtype = dtype.as_torch()
    idx = torch.arange(N, device="cuda", dtype=torch.float32)
    # Exact in both dtypes and inside (-1, 1), keeping every result finite.
    a = ((idx - 63.5) * 0.015625).to(torch_dtype)
    # acosh needs [1, inf).
    c = (idx * 0.03125 + 1.0).to(torch_dtype)
    b = torch.empty(len(DOMAIN_MATH_OPS), N, device="cuda", dtype=torch_dtype)
    kernel(a, c, b)

    inputs = (a, c)
    for row, (name, intrinsic, ref_fn, which) in enumerate(DOMAIN_MATH_OPS):
        assert body.count(f"{intrinsic}(") == 1
        ref = ref_fn(inputs[which].float()).to(torch_dtype)
        torch.testing.assert_close(
            b[row].float(),
            ref.float(),
            atol=2e-2,
            rtol=2e-2,
            msg=lambda base, name=name: f"T.{name} mismatch\n{base}",
        )


# (op, lowered h* intrinsic, torch reference, dtypes it can reach codegen on).
# bfloat16 T.pow is rejected before codegen by an upstream dtype check, so its
# bridge is not reachable yet (tile-ai/tilelang#2571).
BINARY_MATH_OPS = (
    ("fmod", "hfmod", torch.fmod, (T.float16, T.bfloat16)),
    ("pow", "hpow", torch.pow, (T.float16,)),
    ("atan2", "hatan2", torch.atan2, (T.float16, T.bfloat16)),
    ("hypot", "hhypot", torch.hypot, (T.float16, T.bfloat16)),
    ("copysign", "hcopysign", torch.copysign, (T.float16, T.bfloat16)),
)


def binary_math_kernel(op, dtype, N, rhs_dtype=None, rhs=None):
    fn = getattr(T, op)
    rhs_dtype = dtype if rhs_dtype is None else rhs_dtype

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        C: T.Tensor((N,), rhs_dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(1, threads=N):
            i = T.get_thread_binding()
            # Keep the temporary: it copy-initializes, a store only assigns.
            value = fn(A[i], C[i] if rhs is None else rhs)
            B[i] = value

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "name, intrinsic, ref_fn, dtype, rhs_mode",
    [
        (n, h, f, d, mode)
        for n, h, f, dtypes in BINARY_MATH_OPS
        for d in dtypes
        for mode in (("same", "float32", "constant") if n in ("atan2", "hypot", "copysign") else ("same",))
    ],
)
def test_binary_math_intrinsics(name, intrinsic, ref_fn, dtype, rhs_mode):
    N = 128
    rhs_dtype = dtype if rhs_mode == "same" else T.float32
    rhs = -1.0001 if rhs_mode == "constant" else None
    kernel = tilelang.compile(binary_math_kernel(name, dtype, N, rhs_dtype, rhs), target="cuda")
    body = kernel.get_kernel_source()
    body = body[body.rindex("__global__") :]
    assert body.count(f"{intrinsic}(") == 1

    torch_dtype = dtype.as_torch()
    idx = torch.arange(N, device="cuda", dtype=torch.float32)
    if name == "fmod":
        # Both operands take both signs and |A| usually exceeds |C|, so the
        # remainder is non-trivial and the dividend's sign is exercised. Every
        # value is exact in float16 and bfloat16.
        a = ((idx - 64) * 0.25).to(torch_dtype)
        c = (((idx % 5) - 2.5) * 1.5).to(torch_dtype)
    elif name == "pow":
        # [0.5, 1.5) base and [0.5, 2.5) exponent keep pow finite in float16.
        a = (idx * 0.0078125 + 0.5).to(torch_dtype)
        c = (idx * 0.015625 + 0.5).to(torch_dtype)
    else:
        # A crosses zero and C takes both signs at a comparable magnitude: atan2
        # sees all four quadrants, hypot cannot pass as |A|, copysign must flip.
        a = ((idx - 64) * 0.25).to(torch_dtype)
        c = (((idx % 9) - 4) * 2.0).to(torch_dtype)
    if rhs_mode == "float32":
        # Keep precision that would be lost by converting C to the output dtype.
        c = c.float() + 0.0001
    elif rhs_mode == "constant":
        c = torch.full((N,), rhs, device="cuda", dtype=torch.float32)
    b = torch.empty(N, device="cuda", dtype=torch_dtype)
    kernel(a, c, b)

    ref = ref_fn(a.float(), c.float()).to(torch_dtype)
    torch.testing.assert_close(
        b.float(),
        ref.float(),
        atol=2e-2,
        rtol=2e-2,
        msg=lambda base: f"T.{name} mismatch\n{base}",
    )


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16])
@pytest.mark.parametrize("rhs_dtype", [None, T.float32, T.float64])
def test_copysign_special_values(dtype, rhs_dtype):
    rhs_dtype = dtype if rhs_dtype is None else rhs_dtype

    def special_values(dtype):
        # Construct encodings directly so input conversion cannot canonicalize
        # the signs or payloads of quiet and signaling NaNs.
        int_dtype, width, inf, quiet = {
            T.float16: (torch.int16, 16, 0x7C00, 0x0200),
            T.bfloat16: (torch.int16, 16, 0x7F80, 0x0040),
            T.float32: (torch.int32, 32, 0x7F800000, 0x00400000),
            T.float64: (torch.int64, 64, 0x7FF0000000000000, 0x0008000000000000),
        }[dtype]
        magnitudes = [0, 1, inf - 1, inf, inf | 1, inf | quiet, inf | quiet | 1, quiet]
        bits = [value + sign for value in magnitudes for sign in (0, -(1 << (width - 1)))]
        return torch.tensor(bits * 2, device="cuda", dtype=int_dtype).view(dtype.as_torch()), int_dtype

    a, _ = special_values(dtype)
    c, rhs_int_dtype = special_values(rhs_dtype)
    # Use different magnitudes and signs so returning either input cannot pass.
    c = c.roll(3)
    b = torch.empty_like(a)
    kernel = tilelang.compile(
        binary_math_kernel("copysign", dtype, a.numel(), rhs_dtype),
        target="cuda",
        pass_configs={"tl.enable_fast_math": True},
    )
    kernel(a, c, b)

    # copysign must preserve A's magnitude/payload and take C's sign, even for
    # NaNs, signed zeros and subnormals under fast math.
    assert torch.equal(b.view(torch.int16) & 0x7FFF, a.view(torch.int16) & 0x7FFF)
    assert torch.equal(b.view(torch.int16) < 0, c.view(rhs_int_dtype) < 0)


def ldexp_kernel(dtype, N):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        C: T.Tensor((N,), T.int32),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(1, threads=N):
            i = T.get_thread_binding()
            # Keep the temporary: it copy-initializes, a store only assigns.
            value = T.ldexp(A[i], C[i])
            B[i] = value

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16])
def test_ldexp_intrinsic(dtype):
    N = 128
    kernel = tilelang.compile(ldexp_kernel(dtype, N), target="cuda")
    body = kernel.get_kernel_source()
    body = body[body.rindex("__global__") :]
    assert body.count("hldexp(") == 1

    torch_dtype = dtype.as_torch()
    idx = torch.arange(N, device="cuda", dtype=torch.float32)
    # Exponents of both signs. Scaling exact multiples of 0.25 by 2**n is exact
    # in both dtypes, so the result has to match exactly.
    a = ((idx - 64) * 0.25).to(torch_dtype)
    n = ((idx % 7) - 3).to(torch.int32)
    b = torch.empty(N, device="cuda", dtype=torch_dtype)
    kernel(a, n, b)

    ref = torch.ldexp(a.float(), n).to(torch_dtype)
    torch.testing.assert_close(b.float(), ref.float(), atol=0, rtol=0)


# Fast math can flush bf16 values and wider targets near zero. Check it for
# both output dtypes when the target is wider than the result.
@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "dtype, fast_math, rhs_dtype, rhs",
    [(T.float16, False, None, None), (T.bfloat16, False, None, None), (T.bfloat16, True, None, None)]
    + [(d, False, T.float32, 1.0001) for d in (T.float16, T.bfloat16)]
    + [(d, fast, r, None) for d in (T.float16, T.bfloat16) for fast in (False, True) for r in (T.float32, T.float64)],
)
def test_nextafter_intrinsic(dtype, fast_math, rhs_dtype, rhs):
    N = 128
    kernel = tilelang.compile(
        binary_math_kernel("nextafter", dtype, N, rhs_dtype, rhs),
        target="cuda",
        pass_configs={"tl.enable_fast_math": fast_math},
    )
    body = kernel.get_kernel_source()
    body = body[body.rindex("__global__") :]
    assert body.count("hnextafter(") == 1

    torch_dtype = dtype.as_torch()
    idx = torch.arange(N, device="cuda", dtype=torch.float32)
    # Steps in both directions on both sides of zero, including x == y.
    a = ((idx - 64) * 0.25).to(torch_dtype)
    c = (((idx % 9) - 4) * 2.0).to(torch_dtype)
    # The edges the encoding step has to get right: signed zeros, the
    # smallest subnormal, the largest finite value, infinities and NaN.
    inf, nan = float("inf"), float("nan")
    big = torch.finfo(torch_dtype).max
    tiny = torch.tensor([1], dtype=torch.int16).view(torch_dtype).item()
    edge_a = [0.0, -0.0, 0.0, -0.0, tiny, -tiny, big, -big, inf, -inf, nan, 1.0]
    edge_c = [1.0, -1.0, -0.0, 0.0, -1.0, 1.0, inf, -inf, 0.0, 0.0, 1.0, nan]
    a[: len(edge_a)] = torch.tensor(edge_a, dtype=torch_dtype)
    c[: len(edge_c)] = torch.tensor(edge_c, dtype=torch_dtype)
    if rhs is not None:
        # For A == 1, rounding the literal to 16 bits would incorrectly stay at 1.
        c = torch.full((N,), rhs, device="cuda", dtype=rhs_dtype.as_torch())
    elif rhs_dtype is not None:
        # One step in the wider dtype usually rounds back to A. Include steps on
        # both sides of signed zeros, subnormals, finite limits and infinities.
        direction = torch.where(idx % 2 == 0, inf, -inf).to(rhs_dtype.as_torch())
        c = torch.nextafter(a.to(rhs_dtype.as_torch()), direction)
        c[2], c[3] = -0.0, 0.0
        c[10], c[11] = 1.0, nan
    b = torch.empty(N, device="cuda", dtype=torch_dtype)
    kernel(a, c, b)

    if rhs_dtype is None:
        ref = torch.nextafter(a, c)
    else:
        # Choose direction before rounding C, then ask torch to take a 16-bit
        # step. Preserve equality (including signed zero) and NaN semantics.
        wide_a, wide_c = a.double(), c.double()
        direction = torch.where(wide_a < wide_c, inf, -inf).to(torch_dtype)
        ref = torch.nextafter(a, direction)
        ref = torch.where(wide_a == wide_c, c.to(torch_dtype), ref)
        ref = torch.where(torch.isnan(a) | torch.isnan(c), nan, ref)
    # Compare encodings: the sign of a zero result is part of the answer.
    non_nan = ~torch.isnan(ref)
    assert torch.equal(torch.isnan(b), ~non_nan)
    assert torch.equal(b.view(torch.int16)[non_nan], ref.view(torch.int16)[non_nan])


if __name__ == "__main__":
    tilelang.testing.main()
