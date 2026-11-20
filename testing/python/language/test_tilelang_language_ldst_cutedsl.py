"""Explicit-width and read-only-cache loads and stores on CuTeDSL.

Mirrors test_tilelang_language_ldg.py, test_tilelang_language_ldst.py,
issue/test_tilelang_issue_3025.py and the ``T.__ldg`` cases of
test_tilelang_language_intrinsics_codegen.py with ``target="cutedsl"``:

T.ldg32/64/128/256  – tl::load_global_N[_conditional]; zeros when the predicate is false
T.stg32/64/128/256  – tl::store_global_N[_conditional]; no store when the predicate is false
T.lds32/64/128      – tl::load_shared_N
T.sts32/64/128      – tl::store_shared_N
T.__ldg             – __ldg (ld.global.nc)
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


N = 128
SENTINEL = 7.0


def _require_cutedsl():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")


def _compile(func, **kwargs):
    _require_cutedsl()
    return tilelang.compile(func, target="cutedsl", **kwargs)


def _first_half(x, fill):
    ref = torch.full_like(x, fill)
    ref[: x.numel() // 2] = x[: x.numel() // 2]
    return ref


# ---------------------------------------------------------------------------
# ldg32 / ldg64 / ldg128 / ldg256
# ---------------------------------------------------------------------------


def kernel_ldg(width, predicated, n=N):
    lanes = width // 32
    vec = T.float32 if lanes == 1 else f"float32x{lanes}"
    ldg = getattr(T, f"ldg{width}")

    @T.prim_func
    def main(X: T.Tensor((n,), T.float32), Y: T.Tensor((n,), T.float32)):
        with T.Kernel(n // lanes, threads=32) as pid:
            if predicated:
                value = ldg(X[pid * lanes : pid * lanes + lanes], pred=pid < n // lanes // 2)
            else:
                value = ldg(X[pid * lanes : pid * lanes + lanes])
            Y[pid * lanes : pid * lanes + lanes] = T.reinterpret(value, vec)

    return main


def run_ldg(width, predicated, n=N):
    kernel = _compile(kernel_ldg(width, predicated, n))
    src = kernel.get_kernel_source()
    assert f"tl.ldg{width}(" in src, src
    x = torch.randn(n, dtype=torch.float32, device="cuda")
    # A false predicate must zero-fill, so start from a non-zero sentinel.
    y = torch.full((n,), SENTINEL, dtype=torch.float32, device="cuda")
    kernel(x, y)
    expected = _first_half(x, 0.0) if predicated else x
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("width", [32, 64, 128])
@pytest.mark.parametrize("predicated", [False, True])
def test_ldg(width, predicated):
    run_ldg(width, predicated)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@pytest.mark.parametrize("predicated", [False, True])
def test_ldg256(predicated):
    run_ldg(256, predicated, n=256)


def kernel_ldg_int64_predicate():
    @T.prim_func
    def main(X: T.Tensor((N,), T.float32), P: T.Tensor((N,), T.int64), Y: T.Tensor((N,), T.float32)):
        with T.Kernel(N, threads=32) as pid:
            Y[pid] = T.reinterpret(T.ldg32(X[pid], pred=P[pid]), T.float32)

    return main


@tilelang.testing.requires_cuda
def test_ldg_int64_predicate():
    """The ``bool pred`` parameter tests the whole 64-bit value against zero."""
    kernel = _compile(kernel_ldg_int64_predicate())
    x = torch.randn(N, dtype=torch.float32, device="cuda")
    p = torch.zeros(N, dtype=torch.int64, device="cuda")
    p[::2] = 1 << 32
    y = torch.full((N,), SENTINEL, dtype=torch.float32, device="cuda")
    kernel(x, p, y)
    expected = torch.where(p != 0, x, torch.zeros_like(x))
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# stg32 / stg64 / stg128 / stg256
# ---------------------------------------------------------------------------


def kernel_stg(width, predicated, n=N):
    lanes = width // 32
    ldg = getattr(T, f"ldg{width}")
    stg = getattr(T, f"stg{width}")

    @T.prim_func
    def main(X: T.Tensor((n,), T.float32), Y: T.Tensor((n,), T.float32)):
        with T.Kernel(n // lanes, threads=32) as pid:
            value = ldg(X[pid * lanes : pid * lanes + lanes])
            if predicated:
                stg(Y[pid * lanes : pid * lanes + lanes], value, pred=pid < n // lanes // 2)
            else:
                stg(Y[pid * lanes : pid * lanes + lanes], value)

    return main


def kernel_stg32_reinterpret(predicated):
    @T.prim_func
    def main(X: T.Tensor((N,), T.float32), Y: T.Tensor((N,), T.float32)):
        with T.Kernel(N, threads=32) as pid:
            value = T.reinterpret(X[pid], T.uint32)
            if predicated:
                T.stg32(Y[pid], value, pred=pid < N // 2)
            else:
                T.stg32(Y[pid], value)

    return main


def run_stg(width, predicated, n=N, func=None):
    kernel = _compile(func if func is not None else kernel_stg(width, predicated, n))
    src = kernel.get_kernel_source()
    assert f"tl.stg{width}(" in src, src
    x = torch.randn(n, dtype=torch.float32, device="cuda")
    y = torch.full((n,), SENTINEL, dtype=torch.float32, device="cuda")
    kernel(x, y)
    expected = _first_half(x, SENTINEL) if predicated else x
    torch.testing.assert_close(y, expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("width", [32, 64, 128])
@pytest.mark.parametrize("predicated", [False, True])
def test_stg(width, predicated):
    run_stg(width, predicated)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("predicated", [False, True])
def test_stg32_reinterpreted_value(predicated):
    run_stg(32, predicated, func=kernel_stg32_reinterpret(predicated))


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
@pytest.mark.parametrize("predicated", [False, True])
def test_stg256(predicated):
    run_stg(256, predicated, n=256)


def kernel_ldg256_stg256_roundtrip():
    @T.prim_func
    def main(X: T.Tensor((256,), T.float32), Y: T.Tensor((256,), T.float32)):
        with T.Kernel(32, threads=32) as pid:
            value = T.ldg256(X[pid * 8 : pid * 8 + 8])
            T.stg256(Y[pid * 8 : pid * 8 + 8], value)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_lt(10, 0)
def test_ldg256_stg256_fallback_on_pre_sm100():
    """Below sm_100 the 256-bit accesses split into two 128-bit ones."""
    kernel = _compile(kernel_ldg256_stg256_roundtrip())
    x = torch.randn(256, dtype=torch.float32, device="cuda")
    y = torch.empty(256, dtype=torch.float32, device="cuda")
    kernel(x, y)
    torch.testing.assert_close(y, x, rtol=0, atol=0)


def kernel_stg256_float_value():
    @T.prim_func
    def main(X: T.Tensor((256,), T.float32), Y: T.Tensor((256,), T.float32)):
        with T.Kernel(32, threads=32) as pid:
            value = T.reinterpret(T.ldg256(X[pid * 8 : pid * 8 + 8]), T.float32x8)
            T.stg256(Y[pid * 8 : pid * 8 + 8], value)

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_stg256_float_value():
    """stg256 stores the bits of any 32-byte value, like ``store_global_256<T>``."""
    kernel = _compile(kernel_stg256_float_value())
    x = torch.randn(256, dtype=torch.float32, device="cuda")
    y = torch.empty(256, dtype=torch.float32, device="cuda")
    kernel(x, y)
    torch.testing.assert_close(y, x, rtol=0, atol=0)


def kernel_stg32_int_value():
    @T.prim_func
    def main(A: T.Tensor((N,), T.int32), B: T.Tensor((N,), T.int64), Y: T.Tensor((2, N), T.int32)):
        with T.Kernel(N, threads=32) as pid:
            T.stg32(Y[0, pid], A[pid])
            T.stg32(Y[1, pid], B[pid])

    return main


@tilelang.testing.requires_cuda
def test_stg32_converts_value_to_uint32():
    """``store_global_32(void *, uint32_t)`` converts integer values to uint32_t."""
    kernel = _compile(kernel_stg32_int_value())
    a = torch.randint(-(2**31), 2**31 - 1, (N,), dtype=torch.int32, device="cuda")
    b = torch.randint(-(2**62), 2**62, (N,), dtype=torch.int64, device="cuda")
    y = torch.zeros((2, N), dtype=torch.int32, device="cuda")
    kernel(a, b, y)
    torch.testing.assert_close(y[0], a, rtol=0, atol=0)
    low_words = ((b & 0xFFFFFFFF) + 2**31) % 2**32 - 2**31
    torch.testing.assert_close(y[1], low_words.to(torch.int32), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# lds / sts
# ---------------------------------------------------------------------------


def kernel_lds_sts(width):
    lanes = width // 32
    ldg = getattr(T, f"ldg{width}")
    stg = getattr(T, f"stg{width}")
    lds = getattr(T, f"lds{width}")
    sts = getattr(T, f"sts{width}")

    @T.prim_func
    def main(X: T.Tensor((N,), T.float32), Y: T.Tensor((N,), T.float32)):
        with T.Kernel(1, threads=N // lanes):
            tx = T.get_thread_binding()
            S = T.alloc_shared((N,), T.float32)
            # Read back another thread's slot so the round trip goes through shared memory.
            peer = (tx + 1) % (N // lanes)
            sts(S[tx * lanes : tx * lanes + lanes], ldg(X[tx * lanes : tx * lanes + lanes]))
            T.sync_threads()
            stg(Y[tx * lanes : tx * lanes + lanes], lds(S[peer * lanes : peer * lanes + lanes]))

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("width", [32, 64, 128])
def test_lds_sts(width):
    lanes = width // 32
    kernel = _compile(kernel_lds_sts(width))
    src = kernel.get_kernel_source()
    assert f"tl.sts{width}(" in src, src
    assert f"tl.lds{width}(" in src, src
    x = torch.randn(N, dtype=torch.float32, device="cuda")
    y = torch.empty(N, dtype=torch.float32, device="cuda")
    kernel(x, y)
    torch.testing.assert_close(y, torch.roll(x, -lanes), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Declared strides (issue 3025)
# ---------------------------------------------------------------------------


M_ROWS = 4
N_COLS = 3
PITCH = 8


def kernel_strided_store():
    @T.prim_func
    def main(dst: T.StridedTensor((M_ROWS, N_COLS), (PITCH, 1), "int32")):
        with T.Kernel(1, threads=1):
            for row in T.serial(M_ROWS):
                T.stg32(dst[row, 0], T.Cast("uint32", row + 1))

    return main


def kernel_strided_load():
    @T.prim_func
    def main(
        src: T.StridedTensor((M_ROWS, N_COLS), (PITCH, 1), "int32"),
        out: T.Tensor((M_ROWS,), "int32"),
    ):
        with T.Kernel(1, threads=1):
            for row in T.serial(M_ROWS):
                out[row] = T.reinterpret(T.ldg32(src[row, 0]), "int32")

    return main


@tilelang.testing.requires_cuda
def test_strided_stg32_uses_declared_stride():
    kernel = _compile(kernel_strided_store())
    physical = torch.zeros(M_ROWS * PITCH, dtype=torch.int32, device="cuda")
    view = physical.as_strided((M_ROWS, N_COLS), (PITCH, 1))
    kernel(view)
    expected = torch.arange(1, M_ROWS + 1, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(view[:, 0], expected, rtol=0, atol=0)
    untouched = physical.view(M_ROWS, PITCH)[:, 1:]
    assert torch.count_nonzero(untouched) == 0


@tilelang.testing.requires_cuda
def test_strided_ldg32_uses_declared_stride():
    physical = torch.arange(M_ROWS * PITCH, dtype=torch.int32, device="cuda")
    view = physical.as_strided((M_ROWS, N_COLS), (PITCH, 1))
    out = torch.empty(M_ROWS, dtype=torch.int32, device="cuda")
    kernel = _compile(kernel_strided_load())
    kernel(view, out)
    torch.testing.assert_close(out, view[:, 0], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# __ldg
# ---------------------------------------------------------------------------


def kernel_ldg_plus_one():
    @T.prim_func
    def main(x: T.Tensor((N,), T.float32), y: T.Tensor((N,), T.float32)):
        with T.Kernel(N, threads=32) as pid:
            y[pid] = T.__ldg(x[pid]) + 1.0

    return main


@tilelang.testing.requires_cuda
def test_language_ldg_codegen():
    kernel = _compile(kernel_ldg_plus_one(), out_idx=[1])
    assert "tl.__ldg(" in kernel.get_kernel_source()
    x = torch.randn(N, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(kernel(x), x + 1.0, rtol=0, atol=0)


def kernel_ldg_roundtrip(dtype):
    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
        with T.Kernel(N, threads=32) as pid:
            y[pid] = T.__ldg(x[pid])

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "dtype",
    [T.float16, T.bfloat16, T.float32, T.float64, T.int8, T.uint8, T.int16, T.int32, T.int64],
)
def test_language_ldg_roundtrip(dtype):
    kernel = _compile(kernel_ldg_roundtrip(dtype), out_idx=[1])
    assert "tl.__ldg(" in kernel.get_kernel_source()
    torch_dtype = dtype.as_torch()
    if torch_dtype.is_floating_point:
        x = torch.randn(N, device="cuda").to(torch_dtype)
    else:
        info = torch.iinfo(torch_dtype)
        x = torch.randint(info.min, info.max, (N,), dtype=torch_dtype, device="cuda")
    assert torch.equal(kernel(x), x)


if __name__ == "__main__":
    tilelang.testing.main()
