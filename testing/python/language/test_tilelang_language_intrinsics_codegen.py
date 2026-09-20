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


# (op, lowered h* intrinsic, torch reference, dtypes it can reach codegen on).
# bfloat16 T.pow is rejected before codegen by an upstream dtype check, so its
# bridge is not reachable yet (tile-ai/tilelang#2571).
BINARY_MATH_OPS = (
    ("fmod", "hfmod", torch.fmod, (T.float16, T.bfloat16)),
    ("pow", "hpow", torch.pow, (T.float16,)),
)


def binary_math_kernel(op, dtype, N):
    fn = getattr(T, op)

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(1, threads=N):
            i = T.get_thread_binding()
            # Keep the temporary: it copy-initializes, a store only assigns.
            value = fn(A[i], C[i])
            B[i] = value

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "name, intrinsic, ref_fn, dtype",
    [(n, h, f, d) for n, h, f, dtypes in BINARY_MATH_OPS for d in dtypes],
)
def test_binary_math_intrinsics(name, intrinsic, ref_fn, dtype):
    N = 128
    kernel = tilelang.compile(binary_math_kernel(name, dtype, N), target="cuda")
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
    else:
        # [0.5, 1.5) base and [0.5, 2.5) exponent keep pow finite in float16.
        a = (idx * 0.0078125 + 0.5).to(torch_dtype)
        c = (idx * 0.015625 + 0.5).to(torch_dtype)
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


if __name__ == "__main__":
    tilelang.testing.main()
