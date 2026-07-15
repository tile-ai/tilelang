"""Unsigned TileIR reduction and elementwise numerical tests."""

import tilelang
import tilelang.language as T

from tileir_test_utils import _setup_gpu, _skip_if_tileir_toolchain_unavailable, skip_no_cuda_tile


def _reduce_max_uint32_prim_func(m: int = 16, n: int = 32, dim: int = 1):
    """`T.reduce_max` over a uint32 fragment (mirrors
    `test_tileir_structured_lowering_skips_reduce_identity_combine`'s
    frag/accum shape in test_tileir_structured_reduce_atomic.py)."""

    @T.prim_func
    def kern(A: T.Tensor((m, n), "uint32"), B: T.Tensor((m,), "uint32")):
        with T.Kernel(1, threads=128):
            frag = T.alloc_fragment((m, n), "uint32")
            accum = T.alloc_fragment((m,), "uint32")
            T.copy(A, frag)
            T.reduce_max(frag, accum, dim=dim, clear=True)
            T.copy(accum, B)

    return kern


def _mixed_top_bit_rows(m: int, n: int, torch, device: str = "cuda"):
    """Build an (m, n) int64 tensor whose rows INTERLEAVE top-bit-clear and
    top-bit-set (>= 2**31) uint32 bit patterns (mirrors
    `test_cummax_numerical_2d_uint32_src`'s construction above), asserting
    every row has both classes so an unsigned-vs-signed max genuinely
    diverges."""
    torch.manual_seed(0)
    vals_2d = torch.empty(m, n, device=device, dtype=torch.int64)
    half = n // 2
    for row in range(m):
        clear_vals = torch.arange(row * n, row * n + half, device=device, dtype=torch.int64)
        set_vals = torch.arange(row * half, row * half + half, device=device, dtype=torch.int64) + 2**31
        vals_2d[row, 0::2] = clear_vals
        vals_2d[row, 1::2] = set_vals
    has_clear = (vals_2d < 2**31).any(dim=1)
    has_set = (vals_2d >= 2**31).any(dim=1)
    assert (has_clear & has_set).all(), f"Row(s) missing a value class: clear={has_clear.tolist()}, set={has_set.tolist()}"
    return vals_2d


@skip_no_cuda_tile
def test_reduce_max_uint32_numerical_mixed_top_bit():
    """uint32 `T.reduce_max` with mixed top-bit-set/clear values per row must
    match the UNSIGNED running max, bit-exact (forcing SIGNED comparison
    reproduces the wrong, signed-max answer on this exact test)."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    m, n = 16, 32
    vals_2d = _mixed_top_bit_rows(m, n, torch)
    a_uint_bits = vals_2d.to(torch.int32)

    kernel = tilelang.compile(_reduce_max_uint32_prim_func(m, n, dim=1), execution_backend="tileir")
    b_uint_bits = torch.empty(m, device="cuda", dtype=torch.int32)
    kernel(a_uint_bits, b_uint_bits)

    a_uint64 = a_uint_bits.to(torch.int64) & 0xFFFFFFFF
    ref_uint64 = a_uint64.max(dim=1).values
    ref = (ref_uint64 & 0xFFFFFFFF).to(torch.int32)
    torch.testing.assert_close(b_uint_bits, ref, rtol=0, atol=0)


def _elementwise_max_uint32_prim_func(n: int = 256):
    """Elementwise `T.max(A[idx], B[idx])` over uint32 GLOBAL buffers."""

    @T.prim_func
    def kern(A: T.Tensor((n,), "uint32"), B: T.Tensor((n,), "uint32"), C: T.Tensor((n,), "uint32")):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as bx:
            for i in T.Parallel(128):
                idx = bx * 128 + i
                if idx < n:
                    C[idx] = T.max(A[idx], B[idx])

    return kern


@skip_no_cuda_tile
def test_elementwise_max_uint32_numerical_mixed_top_bit():
    """uint32 elementwise `T.max` with mixed top-bit-set/clear values,
    ALTERNATING which operand (A vs B) carries the top-bit-set value so
    neither "lhs always wins" nor "rhs always wins" could accidentally look
    correct under a signed comparison due to operand-position bias. Must
    match the UNSIGNED max, bit-exact."""
    _skip_if_tileir_toolchain_unavailable()
    torch, *_ = _setup_gpu()

    n = 256
    idx = torch.arange(n, device="cuda", dtype=torch.int64)
    small = idx
    large = idx + 2**31
    even = idx % 2 == 0
    a_vals = torch.where(even, small, large)
    b_vals = torch.where(even, large, small)
    # Both are non-negative int64 here (true magnitudes), so a plain int64
    # max IS the correct unsigned-bit-pattern reference.
    ref_uint64 = torch.maximum(a_vals, b_vals)

    a = (a_vals & 0xFFFFFFFF).to(torch.int32)
    b = (b_vals & 0xFFFFFFFF).to(torch.int32)
    c = torch.zeros(n, device="cuda", dtype=torch.int32)

    kernel = tilelang.compile(_elementwise_max_uint32_prim_func(n), execution_backend="tileir")
    kernel(a, b, c)

    ref = (ref_uint64 & 0xFFFFFFFF).to(torch.int32)
    torch.testing.assert_close(c, ref, rtol=0, atol=0)
