import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tvm import tirx


FASTMATH_INTRINSICS = [
    T.__exp,
    T.__exp10,
    T.__log,
    T.__log2,
    T.__log10,
    T.__tan,
    T.__cos,
    T.__sin,
]


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float8_e4m3fn",
        "float8_e5m2",
    ],
)
@pytest.mark.parametrize("intrinsic", FASTMATH_INTRINSICS)
def test_fastmath_rejects_unsupported_dtype(intrinsic, dtype):
    value = tirx.Var("value", dtype)

    with pytest.raises(
        TypeError,
        match=rf"T\.{intrinsic.__name__} only supports floating-point inputs, "
        rf"but got {re.escape(dtype)}",
    ):
        intrinsic(value)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("intrinsic", FASTMATH_INTRINSICS)
def test_fastmath_accepts_float_dtype_at_frontend(intrinsic, dtype):
    value = tirx.Var("value", dtype)
    result = intrinsic(value)

    assert isinstance(result, tirx.Call)
    assert result.dtype == dtype


# Rows written by the kernels below, in order. The "pos" domain keeps the log
# family defined and exp10 finite in float16; "sym" is mixed-sign and stays
# clear of tan's pole.
FASTMATH_ROWS = (
    ("__exp", torch.exp, "pos"),
    ("__exp10", lambda x: torch.pow(10.0, x), "pos"),
    ("__log", torch.log, "pos"),
    ("__log2", torch.log2, "pos"),
    ("__log10", torch.log10, "pos"),
    ("__sin", torch.sin, "sym"),
    ("__cos", torch.cos, "sym"),
    ("__tan", torch.tan, "sym"),
)

MIXED_ROWS = (
    ("log", torch.log, "pos"),
    ("sqrt", torch.sqrt, "pos"),
    ("sin", torch.sin, "sym"),
    ("cos", torch.cos, "sym"),
    ("tanh", torch.tanh, "sym"),
    ("__exp", torch.exp, "pos"),
)

TOLERANCE = 2e-2


def _fastmath_kernel(dtype, N):
    @T.prim_func
    def main(
        P: T.Tensor((N,), dtype),
        S: T.Tensor((N,), dtype),
        B: T.Tensor((len(FASTMATH_ROWS), N), dtype),
    ):
        with T.Kernel(1, threads=N):
            i = T.get_thread_binding()
            # Materialize each result: copy-initialization rejects a
            # float-returning call, while a store only needs an assignment.
            exp_value = T.__exp(P[i])
            exp10_value = T.__exp10(P[i])
            log_value = T.__log(P[i])
            log2_value = T.__log2(P[i])
            log10_value = T.__log10(P[i])
            sin_value = T.__sin(S[i])
            cos_value = T.__cos(S[i])
            tan_value = T.__tan(S[i])
            B[0, i] = exp_value
            B[1, i] = exp10_value
            B[2, i] = log_value
            B[3, i] = log2_value
            B[4, i] = log10_value
            B[5, i] = sin_value
            B[6, i] = cos_value
            B[7, i] = tan_value

    return main


def _mixed_kernel(dtype, N):
    @T.prim_func
    def main(
        P: T.Tensor((N,), dtype),
        S: T.Tensor((N,), dtype),
        B: T.Tensor((len(MIXED_ROWS), N), dtype),
    ):
        with T.Kernel(1, threads=N):
            i = T.get_thread_binding()
            log_value = T.log(P[i])
            sqrt_value = T.sqrt(P[i])
            sin_value = T.sin(S[i])
            cos_value = T.cos(S[i])
            tanh_value = T.tanh(S[i])
            # This single fast-math op pulls math.h into the whole kernel,
            # rebinding every half-style name above to a cutlass::fast_* macro.
            exp_value = T.__exp(P[i])
            B[0, i] = log_value
            B[1, i] = sqrt_value
            B[2, i] = sin_value
            B[3, i] = cos_value
            B[4, i] = tanh_value
            B[5, i] = exp_value

    return main


def _run(kernel_fn, rows, dtype):
    N = 128
    kernel = tilelang.compile(kernel_fn(dtype, N), target="cuda")
    torch_dtype = getattr(torch, dtype)
    p = (torch.arange(N, device="cuda") * 0.03125 + 0.3).to(torch_dtype)
    s = ((torch.arange(N, device="cuda") - 63.5) * 0.02).to(torch_dtype)
    b = torch.empty(len(rows), N, device="cuda", dtype=torch_dtype)
    kernel(p, s, b)

    inputs = {"pos": p, "sym": s}
    for row, (name, ref_fn, domain) in enumerate(rows):
        ref = ref_fn(inputs[domain].float())
        torch.testing.assert_close(
            b[row].float(),
            ref.to(torch_dtype).float(),
            atol=TOLERANCE,
            rtol=TOLERANCE,
            msg=lambda base, name=name: f"T.{name} mismatch\n{base}",
        )


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_fastmath_16bit_intrinsics(dtype):
    _run(_fastmath_kernel, FASTMATH_ROWS, dtype)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_plain_math_alongside_fastmath(dtype):
    _run(_mixed_kernel, MIXED_ROWS, dtype)
