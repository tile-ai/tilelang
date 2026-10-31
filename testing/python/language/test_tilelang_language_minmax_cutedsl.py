"""CuTeDSL min/max must match the CUDA target bit for bit: NaN handling, the
order of signed zeros and the NaN-propagating reductions."""

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


def _compile_both(func, **kwargs):
    _require_cutedsl()
    return (
        tilelang.compile(func, target="cuda", **kwargs),
        tilelang.compile(func, target="cutedsl", **kwargs),
    )


_BITS = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
    torch.int32: torch.int32,
}


def _assert_same_bits(got, expected):
    assert torch.equal(got.view(_BITS[got.dtype]), expected.view(_BITS[expected.dtype])), f"{got}\n!=\n{expected}"


def _special_pairs(torch_dtype, n):
    nan, inf = float("nan"), float("inf")
    pairs = [
        (nan, 1.0),
        (1.0, nan),
        (nan, nan),
        (nan, -inf),
        (0.0, -0.0),
        (-0.0, 0.0),
        (-0.0, -0.0),
        (inf, -inf),
        (-inf, 3.0),
        (1.0, 2.0),
        (-3.0, 2.0),
        (2.5, 2.5),
    ]
    if torch_dtype in (torch.float32, torch.float64):
        tiny = torch.finfo(torch_dtype).tiny / 4
        pairs += [(tiny, -tiny), (-tiny, 0.0)]
    a = torch.randn(n, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)
    for i, (x, y) in enumerate(pairs):
        a[i], b[i] = x, y
    return a.to(torch_dtype).cuda(), b.to(torch_dtype).cuda()


def _elementwise_minmax(n, dtype):
    @T.prim_func
    def main(
        A: T.Tensor((n,), dtype),
        B: T.Tensor((n,), dtype),
        Mx: T.Tensor((n,), dtype),
        Mn: T.Tensor((n,), dtype),
    ):
        with T.Kernel(1, threads=n):
            tx = T.get_thread_binding()
            Mx[tx] = T.max(A[tx], B[tx])
            Mn[tx] = T.min(A[tx], B[tx])

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64", "int32"])
def test_elementwise_max_min_matches_cuda(dtype):
    n = 32
    torch_dtype = getattr(torch, dtype)
    cuda_kernel, cutedsl_kernel = _compile_both(_elementwise_minmax(n, dtype), out_idx=[2, 3])
    if torch_dtype is torch.int32:
        a = torch.randint(-100, 100, (n,), dtype=torch.int32, device="cuda")
        b = torch.randint(-100, 100, (n,), dtype=torch.int32, device="cuda")
    else:
        a, b = _special_pairs(torch_dtype, n)
    for got, expected in zip(cutedsl_kernel(a, b), cuda_kernel(a, b)):
        _assert_same_bits(got, expected)


def _reduce(reduce_fn, length, dtype, *, clear, nan_propagate):
    kwargs = {} if nan_propagate is None else {"nan_propagate": nan_propagate}

    @T.prim_func
    def main(a: T.Tensor((length,), dtype), out: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=32):
            frag = T.alloc_fragment((length,), dtype)
            out_frag = T.alloc_fragment((1,), dtype)
            T.copy(a, frag)
            out_frag[0] = -1.0
            reduce_fn(frag, out_frag, clear=clear, **kwargs)
            T.copy(out_frag, out)

    return main


def _reduce_inputs(torch_dtype, length):
    nan = float("nan")
    cases = {
        "nan": torch.arange(length, dtype=torch.float32),
        "all_negative_zero": torch.full((length,), -0.0),
        "mixed_zero": torch.where(torch.arange(length) % 2 == 0, 0.0, -0.0),
        "random": torch.randn(length),
        "all_nan": torch.full((length,), nan),
    }
    cases["nan"][7] = nan
    cases["nan"][40] = nan
    return {k: v.to(torch_dtype).cuda() for k, v in cases.items()}


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("reduce_name", ["reduce_max", "reduce_min", "reduce_absmax"])
@pytest.mark.parametrize("clear", [True, False])
@pytest.mark.parametrize("nan_propagate", [False, True])
def test_half_reduce_matches_cuda(dtype, reduce_name, clear, nan_propagate):
    length = 64
    func = _reduce(getattr(T, reduce_name), length, dtype, clear=clear, nan_propagate=nan_propagate)
    cuda_kernel, cutedsl_kernel = _compile_both(func, out_idx=-1)
    nan_op = "Nan" if nan_propagate else ""
    op = "Min" if reduce_name == "reduce_min" else "Max"
    assert f"tl.{op}Op{nan_op}" in cutedsl_kernel.get_kernel_source()
    for name, a in _reduce_inputs(getattr(torch, dtype), length).items():
        got, expected = cutedsl_kernel(a), cuda_kernel(a)
        assert torch.equal(got.view(torch.int16), expected.view(torch.int16)), f"{name}: {got} != {expected}"


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("reduce_name", ["reduce_max", "reduce_min", "reduce_absmax"])
def test_float_reduce_matches_cuda(dtype, reduce_name):
    length = 64
    func = _reduce(getattr(T, reduce_name), length, dtype, clear=True, nan_propagate=None)
    cuda_kernel, cutedsl_kernel = _compile_both(func, out_idx=-1)
    for name, a in _reduce_inputs(getattr(torch, dtype), length).items():
        got, expected = cutedsl_kernel(a), cuda_kernel(a)
        assert torch.equal(got.view(_BITS[got.dtype]), expected.view(_BITS[got.dtype])), f"{name}: {got} != {expected}"


if __name__ == "__main__":
    tilelang.testing.main()
