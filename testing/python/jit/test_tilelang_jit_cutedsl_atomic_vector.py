"""CuTeDSL x2/x4 atomic adds, mirroring ``src/tl_templates/cuda/atomic.h``.

On sm_90+ float32 x2/x4 and 16-bit x4 are single ``atom.global`` vector
atomics; below sm_90 (vector ``atom`` needs sm_90) they fall back to scalar or
packed pair atomics. 16-bit x2 is always one packed ``f16x2``/``bf16x2`` atomic
(a CAS loop for bf16 below sm_90), which also works on shared memory. 16-bit
x4-with-return is two packed pair atomics on every architecture.

Every destination carries a guard tail: an over-wide atomic (16 bytes into an
8-byte 16-bit x4 destination) would show up as a changed tail value. The
contended tests use small integers, so every sum is exact even in bfloat16.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

DTYPES = ["float32", "float16", "bfloat16"]
GUARD = 4
THREADS = 32


def _require_cutedsl():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except (ImportError, ModuleNotFoundError, RuntimeError, AssertionError) as err:
        pytest.skip(f"CuTeDSL is not available: {err}")


@tilelang.jit(target="cutedsl")
def atomic_addx4_prev_kernel(dtype):
    @T.prim_func
    def kernel(
        Dst: T.Tensor((4 + GUARD,), dtype),
        Val: T.Tensor((4,), dtype),
        Prev: T.Tensor((4,), dtype),
    ):
        with T.Kernel(1, threads=1):
            Prev[0:4] = T.atomic_addx4(Dst[0], Val[0], return_prev=True)

    return kernel


@tilelang.jit(target="cutedsl")
def atomic_addx2_prev_kernel(dtype):
    @T.prim_func
    def kernel(
        Dst: T.Tensor((2 + GUARD,), dtype),
        Val: T.Tensor((2,), dtype),
        Prev: T.Tensor((2,), dtype),
    ):
        with T.Kernel(1, threads=1):
            Prev[0:2] = T.atomic_addx2(Dst[0], Val[0], return_prev=True)

    return kernel


@tilelang.jit(target="cutedsl")
def atomic_addx2_prev_shared_kernel(dtype):
    """Contended x2 atomics with return on a shared-memory destination."""

    @T.prim_func
    def kernel(
        Init: T.Tensor((2 + GUARD,), dtype),
        Val: T.Tensor((2,), dtype),
        Out: T.Tensor((2 + GUARD,), dtype),
        Prev: T.Tensor((THREADS, 2), dtype),
    ):
        with T.Kernel(1, threads=THREADS):
            tx = T.get_thread_binding()
            S = T.alloc_shared((2 + GUARD,), dtype)
            if tx < 2 + GUARD:
                S[tx] = Init[tx]
            T.sync_threads()
            Prev[tx, 0:2] = T.atomic_addx2(S[0], Val[0], return_prev=True)
            T.sync_threads()
            if tx < 2 + GUARD:
                Out[tx] = S[tx]

    return kernel


@tilelang.jit(target="cutedsl")
def atomic_addxn_global_kernel(dtype, width):
    """Every thread adds the same ``width`` values to the same destination."""
    atomic_addxn = T.atomic_addx2 if width == 2 else T.atomic_addx4

    @T.prim_func
    def kernel(Dst: T.Tensor((width + GUARD,), dtype), Val: T.Tensor((width,), dtype)):
        with T.Kernel(1, threads=THREADS):
            atomic_addxn(Dst[0], Val[0])

    return kernel


@tilelang.jit(target="cutedsl")
def atomic_addx2_shared_kernel(dtype):
    """Contended x2 atomics on a shared-memory destination."""

    @T.prim_func
    def kernel(Init: T.Tensor((2 + GUARD,), dtype), Val: T.Tensor((2,), dtype), Out: T.Tensor((2 + GUARD,), dtype)):
        with T.Kernel(1, threads=THREADS):
            tx = T.get_thread_binding()
            S = T.alloc_shared((2 + GUARD,), dtype)
            if tx < 2 + GUARD:
                S[tx] = Init[tx]
            T.sync_threads()
            T.atomic_addx2(S[0], Val[0])
            T.sync_threads()
            if tx < 2 + GUARD:
                Out[tx] = S[tx]

    return kernel


def _values(width, dtype):
    torch_dtype = getattr(torch, dtype)
    dst = torch.arange(1, width + GUARD + 1, device="cuda", dtype=torch_dtype)
    val = torch.arange(1, width + 1, device="cuda", dtype=torch_dtype)
    expected = dst.clone()
    expected[:width] += THREADS * val
    return dst, val, expected


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_cutedsl_atomic_addx4_return_prev(dtype):
    _require_cutedsl()
    torch_dtype = getattr(torch, dtype)
    dst = torch.arange(1, 4 + GUARD + 1, device="cuda", dtype=torch_dtype)
    val = torch.arange(1, 5, device="cuda", dtype=torch_dtype) * 10
    prev = torch.zeros(4, device="cuda", dtype=torch_dtype)
    tail_before = dst[4:].clone()

    atomic_addx4_prev_kernel(dtype)(dst, val, prev)

    expected_prev = torch.arange(1, 5, device="cuda", dtype=torch_dtype)
    torch.testing.assert_close(prev, expected_prev, atol=0, rtol=0)
    torch.testing.assert_close(dst, torch.cat([expected_prev + val, tail_before]), atol=0, rtol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_cutedsl_atomic_addx2_return_prev(dtype):
    _require_cutedsl()
    torch_dtype = getattr(torch, dtype)
    dst = torch.arange(1, 2 + GUARD + 1, device="cuda", dtype=torch_dtype)
    val = torch.tensor([10, 20], device="cuda", dtype=torch_dtype)
    prev = torch.zeros(2, device="cuda", dtype=torch_dtype)
    tail_before = dst[2:].clone()

    kernel = atomic_addx2_prev_kernel(dtype)
    assert "tl.AtomicAddx2Ret(" in kernel.get_kernel_source()
    kernel(dst, val, prev)

    expected_prev = torch.tensor([1, 2], device="cuda", dtype=torch_dtype)
    torch.testing.assert_close(prev, expected_prev, atol=0, rtol=0)
    torch.testing.assert_close(dst, torch.cat([expected_prev + val, tail_before]), atol=0, rtol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cutedsl_atomic_addx2_return_prev_shared(dtype):
    _require_cutedsl()
    init, val, expected = _values(2, dtype)
    out = torch.zeros_like(init)
    prev = torch.zeros(THREADS, 2, device="cuda", dtype=init.dtype)
    atomic_addx2_prev_shared_kernel(dtype)(init, val, out, prev)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    # The pair is updated as one word, so the returned pairs are exactly the
    # THREADS successive states init + k * val, in some order.
    steps = torch.arange(THREADS, device="cuda", dtype=init.dtype)[:, None]
    states = init[:2][None, :] + steps * val[None, :]
    order = torch.argsort(prev[:, 0].float())
    torch.testing.assert_close(prev[order], states, atol=0, rtol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("dtype", DTYPES)
def test_cutedsl_atomic_addxn_global(dtype, width):
    _require_cutedsl()
    dst, val, expected = _values(width, dtype)
    atomic_addxn_global_kernel(dtype, width)(dst, val)
    torch.testing.assert_close(dst, expected, atol=0, rtol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cutedsl_atomic_addx2_shared(dtype):
    _require_cutedsl()
    init, val, expected = _values(2, dtype)
    out = torch.zeros_like(init)
    atomic_addx2_shared_kernel(dtype)(init, val, out)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
