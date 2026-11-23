"""End-to-end NPU numeric acceptance for non-divisible Parallel extents.

The kernels use ``T.Parallel`` extents that are not multiples of the
SIMD_VF lanes, so VectorizeParallelToPTO emits the per-unit ``remaining``
scalar, the per-chunk dynamic create_mask, and the masked tail store.
Checks cover:

- padded output UB + sentinel preservation past E (lane discipline of
  the predicated store);
- compact output UB of exactly E elements (no out-of-bounds write);
- 2D per-row reset and per-row sentinel (row independence);
- mixed divisible/non-divisible units in one VF;
- the three lane widths and the dtypes of the design acceptance matrix.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["TILELANG_DISABLE_DATA_RACE_CHECK"] = "1"

import tilelang
import tilelang.ascend.language as T


def _require_npu_runtime():
    if not hasattr(torch, "npu"):
        pytest.skip("torch.npu is unavailable in the current environment")

    try:
        _ = torch.randn(1, device=torch.device("npu"))
        torch.npu.synchronize()
    except Exception as err:
        pytest.skip(f"NPU runtime is unavailable in the current environment: {err}")


@pytest.fixture(scope="module")
def npu():
    _require_npu_runtime()
    return torch.device("npu")


SENTINEL = -12345.0

TORCH_DT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _ceil_to_lanes(e, lanes):
    return (e + lanes - 1) // lanes * lanes


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _k1d(E, out_extent, lanes, dtype):
    """1D scale-by-2 with padded or compact output UB (out_extent = P or
    E); B_init pre-fills the output so the sentinel check observes which
    lanes the masked store actually wrote."""

    P = _ceil_to_lanes(E, lanes)

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        B_init: T.Tensor((out_extent,), dtype),
        B: T.Tensor((out_extent,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((out_extent,), dtype)
            T.copy(A, a)
            T.copy(B_init, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    b[i] = a[i] + a[i]
            T.copy(b, B)

    return main


def _k2d(M, N, lanes, dtype):
    """2D row scale-by-2: j is vectorized and non-divisible. Both the
    source and the output UB use the padded row stride P: the tail
    vload reads [row*N, row*N + L) which crosses into the next row's
    head, so the buffer must be readable through the padded range AND
    the load base (row stride) must stay vector-aligned — a logical
    stride of N elements would make later rows' vload bases unaligned
    (the design keeps alignment a caller responsibility, same as tail
    readability)."""

    P = _ceil_to_lanes(N, lanes)

    @T.prim_func
    def main(
        A: T.Tensor((M * P,), dtype),
        B_init: T.Tensor((M * P,), dtype),
        B: T.Tensor((M * P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((M * P,), dtype)
            b = T.alloc_shared((M * P,), dtype)
            T.copy(A, a)
            T.copy(B_init, b)
            with T.SimdVF(lanes=lanes), T.Parallel(M, N) as (i, j):
                b[i * P + j] = a[i * P + j] + a[i * P + j]
            T.copy(b, B)

    return main


def _k_mixed(E_div, E_non, lanes, dtype):
    """Divisible unit then non-divisible unit in one VF."""

    Pd = _ceil_to_lanes(E_div, lanes)
    Pn = _ceil_to_lanes(E_non, lanes)

    @T.prim_func
    def main(
        A1: T.Tensor((Pd,), dtype),
        B1_init: T.Tensor((Pd,), dtype),
        B1: T.Tensor((Pd,), dtype),
        A2: T.Tensor((Pn,), dtype),
        B2_init: T.Tensor((Pn,), dtype),
        B2: T.Tensor((Pn,), dtype),
    ):
        with T.Kernel(1):
            a1 = T.alloc_shared((Pd,), dtype)
            b1 = T.alloc_shared((Pd,), dtype)
            a2 = T.alloc_shared((Pn,), dtype)
            b2 = T.alloc_shared((Pn,), dtype)
            T.copy(A1, a1)
            T.copy(B1_init, b1)
            T.copy(A2, a2)
            T.copy(B2_init, b2)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E_div):
                    b1[i] = a1[i] + a1[i]
                for i in T.Parallel(E_non):
                    b2[i] = a2[i] + a2[i]
            T.copy(b1, B1)
            T.copy(b2, B2)

    return main


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128, 256])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_1d_padded_sentinel(npu, lanes, dtype):
    """Padded output UB: [0,E) holds 2*A, [E,P) keeps the sentinel."""
    E = 150 if lanes <= 128 else 150  # tail shape independent of lanes
    P = _ceil_to_lanes(E, lanes)
    a = (torch.randn(P) * 2).to(TORCH_DT[dtype])
    b_init = torch.full((P,), SENTINEL, dtype=TORCH_DT[dtype])
    kernel = tilelang.compile(_k1d(E, P, lanes, dtype), target="pto", out_idx=-1)
    out = kernel(a.to(npu), b_init.to(npu))
    torch.npu.synchronize()
    got = out.cpu()
    assert torch.equal(got[0:E], (a[0:E] + a[0:E])), f"{dtype} L={lanes}: values"
    assert bool((got[E:P] == SENTINEL).all()), f"{dtype} L={lanes}: sentinel"


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128, 256])
def test_1d_compact_output(npu, lanes):
    """Compact output UB of exactly E elements: no write past E."""
    E = 100 if lanes == 64 else 150
    P = _ceil_to_lanes(E, lanes)
    a = (torch.randn(P) * 2).to(torch.float32)
    b_init = torch.full((E,), SENTINEL, dtype=torch.float32)
    kernel = tilelang.compile(_k1d(E, E, lanes, "float32"), target="pto", out_idx=-1)
    out = kernel(a.to(npu), b_init.to(npu))
    torch.npu.synchronize()
    got = out.cpu()
    assert torch.equal(got, a[0:E] + a[0:E])


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128, 256])
def test_2d_row_reset_sentinel(npu, lanes):
    """(4, 100) j-vectorized: every row's [0,N) is correct and each row's
    [N,P) tail keeps its own sentinel — rows are independent (per-row
    remaining reset) and no row writes past its own E lanes."""
    M, N = 4, 100
    P = _ceil_to_lanes(N, lanes)
    a_rows = torch.randn(M, N) * 2
    a = torch.zeros(M, P)
    a[:, 0:N] = a_rows  # padding lanes read as zeros; never stored (mask)
    b_init = torch.full((M * P,), SENTINEL, dtype=torch.float32)
    kernel = tilelang.compile(_k2d(M, N, lanes, "float32"), target="pto", out_idx=-1)
    out = kernel(a.to(npu).flatten(), b_init.to(npu).flatten())
    torch.npu.synchronize()
    got = out.cpu().reshape(M, P)
    expect_row = a_rows + a_rows
    assert torch.equal(got[:, 0:N], expect_row), "row values"
    assert bool((got[:, N:P] == SENTINEL).all()), "row tail sentinel"


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128, 256])
def test_mixed_units_both_masks(npu, lanes):
    """Divisible unit (full mask) and non-divisible unit (dynamic mask)
    in one VF: both produce correct values, and the non-divisible tail
    keeps the sentinel."""
    E_div = lanes * 2
    E_non = 150
    Pn = _ceil_to_lanes(E_non, lanes)
    a1 = (torch.randn(E_div) * 2).to(torch.float32)
    b1_init = torch.full((E_div,), SENTINEL, dtype=torch.float32)
    a2 = (torch.randn(Pn) * 2).to(torch.float32)
    b2_init = torch.full((Pn,), SENTINEL, dtype=torch.float32)
    kernel = tilelang.compile(_k_mixed(E_div, E_non, lanes, "float32"), target="pto", out_idx=[2, 5])
    out1, out2 = kernel(a1.to(npu), b1_init.to(npu), a2.to(npu), b2_init.to(npu))
    torch.npu.synchronize()
    g1, g2 = out1.cpu(), out2.cpu()
    assert torch.equal(g1, a1 + a1), "divisible unit values"
    assert torch.equal(g2[0:E_non], a2[0:E_non] + a2[0:E_non]), "nondiv unit values"
    assert bool((g2[E_non:Pn] == SENTINEL).all()), "nondiv unit sentinel"


@pytest.mark.pto
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_2d_half_precision(npu, dtype):
    """2D row reset under half precision."""
    M, N, lanes = 4, 100, 128
    P = _ceil_to_lanes(N, lanes)
    a_rows = (torch.randn(M, N) * 2).to(TORCH_DT[dtype])
    a = torch.zeros(M, P, dtype=TORCH_DT[dtype])
    a[:, 0:N] = a_rows
    b_init = torch.full((M * P,), SENTINEL, dtype=TORCH_DT[dtype])
    kernel = tilelang.compile(_k2d(M, N, lanes, dtype), target="pto", out_idx=-1)
    out = kernel(a.to(npu).flatten(), b_init.to(npu).flatten())
    torch.npu.synchronize()
    got = out.cpu().reshape(M, P)
    expect_row = a_rows + a_rows
    assert torch.equal(got[:, 0:N], expect_row), f"{dtype} row values"
    assert bool((got[:, N:P] == SENTINEL).all()), f"{dtype} row tail sentinel"


# ---------------------------------------------------------------------------
# Design 9.2.4 length matrix
# ---------------------------------------------------------------------------


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128, 256])
@pytest.mark.parametrize(
    "e_kind",
    ["lt_l_minus_1", "l_plus_1", "two_l_minus_1", "two_l_plus_17", "three_l"],
    ids=["L-1", "L+1", "2L-1", "2L+17", "3L"],
)
def test_1d_length_matrix(npu, lanes, e_kind):
    """Design 9.2.4 matrix: 1<E<L (L-1, a single lane past the tail),
    E>L non-divisible (L+1, 2L-1, 2L+17) and E>L divisible (3L); padded
    output UB with sentinel past E. E=L and 2L are covered by
    test_vecadd_1d_singlechunk and the mixed units test."""
    E = {
        "lt_l_minus_1": lanes - 1,
        "l_plus_1": lanes + 1,
        "two_l_minus_1": 2 * lanes - 1,
        "two_l_plus_17": 2 * lanes + 17,
        "three_l": 3 * lanes,
    }[e_kind]
    P = _ceil_to_lanes(E, lanes)
    a = (torch.randn(P) * 2).to(torch.float32)
    b_init = torch.full((P,), SENTINEL, dtype=torch.float32)
    kernel = tilelang.compile(_k1d(E, P, lanes, "float32"), target="pto", out_idx=-1)
    out = kernel(a.to(npu), b_init.to(npu))
    torch.npu.synchronize()
    got = out.cpu()
    assert torch.equal(got[0:E], a[0:E] + a[0:E]), f"E={E} L={lanes}: values"
    assert bool((got[E:P] == SENTINEL).all()), f"E={E} L={lanes}: sentinel"


def _k2d_col(M, N, lanes, dtype):
    """i-continuous 2D: row-major (N, P_M) storage accessed c[j * P_M + i]
    — i (extent M) is the selected non-divisible dim, j (extent N) the
    kept coordinate; each row resets remaining and keeps its own tail
    sentinel."""

    P = _ceil_to_lanes(M, lanes)

    @T.prim_func
    def main(
        A: T.Tensor((N * P,), dtype),
        B_init: T.Tensor((N * P,), dtype),
        B: T.Tensor((N * P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((N * P,), dtype)
            b = T.alloc_shared((N * P,), dtype)
            T.copy(A, a)
            T.copy(B_init, b)
            with T.SimdVF(lanes=lanes), T.Parallel(M, N) as (i, j):
                b[j * P + i] = a[j * P + i] + a[j * P + i]
            T.copy(b, B)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128, 256])
def test_2d_i_continuous_nondivisible(npu, lanes):
    """Design 9.2.4: the i-continuous direction with a non-divisible M,
    four kept coordinates with distinct row data — per-row remaining
    reset and per-row sentinel past M."""
    M, N = 100, 4
    P = _ceil_to_lanes(M, lanes)
    a_rows = torch.randn(N, M) * 2
    a = torch.zeros(N, P)
    a[:, 0:M] = a_rows  # padding lanes read as zeros; never stored (mask)
    b_init = torch.full((N * P,), SENTINEL, dtype=torch.float32)
    kernel = tilelang.compile(_k2d_col(M, N, lanes, "float32"), target="pto", out_idx=-1)
    out = kernel(a.to(npu).flatten(), b_init.to(npu))
    torch.npu.synchronize()
    got = out.cpu().reshape(N, P)
    expect_row = a_rows + a_rows
    assert torch.equal(got[:, 0:M], expect_row), "column values"
    assert bool((got[:, M:P] == SENTINEL).all()), "column tail sentinel"


# ---------------------------------------------------------------------------
# Non-divisible mixed-width / index-as-value acceptance
# ---------------------------------------------------------------------------


def _k1d_cast_widen(E, P, lanes, src_dtype):
    """c[i] = f32(a[i]) + f32(a[i]) — widening with a non-divisible E."""

    @T.prim_func
    def main(
        A: T.Tensor((P,), src_dtype),
        B_init: T.Tensor((P,), "float32"),
        B: T.Tensor((P,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), src_dtype)
            b = T.alloc_shared((P,), "float32")
            T.copy(A, a)
            T.copy(B_init, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    b[i] = T.Cast("float32", a[i]) + T.Cast("float32", a[i])
            T.copy(b, B)

    return main


def _k1d_fp8_narrow(E, P, lanes):
    """c[i] = fp8(a[i]) — FP32 -> E4M3FN narrowing with a non-divisible E."""

    @T.prim_func
    def main(
        A: T.Tensor((P,), "float32"),
        C_init: T.Tensor((P,), T.float8_e4m3fn),
        C: T.Tensor((P,), T.float8_e4m3fn),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), "float32")
            c = T.alloc_shared((P,), T.float8_e4m3fn)
            T.copy(A, a)
            T.copy(C_init, c)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    c[i] = T.Cast(T.float8_e4m3fn, a[i])
            T.copy(c, C)

    return main


def _k1d_fp8_widen(E, P, lanes):
    """c[i] = f32(a[i]) — E4M3FN -> FP32 widening with a non-divisible E."""

    @T.prim_func
    def main(
        A: T.Tensor((P,), T.float8_e4m3fn),
        B_init: T.Tensor((P,), "float32"),
        B: T.Tensor((P,), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), T.float8_e4m3fn)
            b = T.alloc_shared((P,), "float32")
            T.copy(A, a)
            T.copy(B_init, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    b[i] = T.Cast("float32", a[i])
            T.copy(b, B)

    return main


def _k1d_index_value(E, P, lanes):
    """c[i] = a[i] + i — the int32 loop index feeds the value stream
    (int->float casts are outside the supported conversion set)."""

    @T.prim_func
    def main(
        A: T.Tensor((P,), "int32"),
        B_init: T.Tensor((P,), "int32"),
        B: T.Tensor((P,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), "int32")
            b = T.alloc_shared((P,), "int32")
            T.copy(A, a)
            T.copy(B_init, b)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E):
                    b[i] = a[i] + i
            T.copy(b, B)

    return main


def _decode_e4m3(code):
    """E4M3FN: 1 sign, 4 exponent (bias 7), 3 mantissa; exp=15/m=7 is NaN."""
    sign = -1.0 if code & 0x80 else 1.0
    exp = (code >> 3) & 0xF
    mant = code & 0x7
    if exp == 0:
        return sign * mant * 2.0**-9
    if exp == 0xF and mant == 0x7:
        return None  # NaN
    return sign * (1.0 + mant / 8.0) * 2.0 ** (exp - 7)


def _e4m3_table():
    vals = []
    for code in range(256):
        value = _decode_e4m3(code)
        if value is not None and (value > 0 or code == 0):
            vals.append((code, value))
    vals.sort(key=lambda pair: pair[1])
    return vals


_E4M3_TABLE = _e4m3_table()


def _encode_rne(value):
    """Nearest E4M3FN code with ties-to-even, saturating (host reference)."""
    limit = _E4M3_TABLE[-1][1]
    sign_bit = 0x80 if value < 0 else 0x00
    mag = min(abs(value), limit)
    code = _E4M3_TABLE[0][0]
    best = abs(_E4M3_TABLE[0][1] - mag)
    for cand, magnitude in _E4M3_TABLE[1:]:
        delta = abs(magnitude - mag)
        if delta < best or (delta == best and (cand & 1) == 0 and (code & 1) != 0):
            code, best = cand, delta
    return code | sign_bit


@pytest.mark.pto
@pytest.mark.parametrize("src_dtype", ["float16", "bfloat16"])
def test_1d_nondiv_cast_widen(npu, src_dtype):
    """E=150, L=128 (Q=2, tail 22): FP16/BF16 -> FP32 widening under the
    per-chunk dynamic mask — valid lanes exact, tail sentinel kept."""
    E, lanes = 150, 128
    P = _ceil_to_lanes(E, lanes)
    a = (torch.randn(P) * 2).to(TORCH_DT[src_dtype])
    b_init = torch.full((P,), SENTINEL, dtype=torch.float32)
    kernel = tilelang.compile(_k1d_cast_widen(E, P, lanes, src_dtype), target="pto", out_idx=-1)
    out = kernel(a.to(npu), b_init.to(npu))
    torch.npu.synchronize()
    got = out.cpu()
    expect = a[0:E].float() + a[0:E].float()
    assert torch.equal(got[0:E], expect), f"{src_dtype}: values"
    assert bool((got[E:P] == SENTINEL).all()), f"{src_dtype}: tail sentinel"


@pytest.mark.pto
def test_1d_nondiv_fp8_narrow(npu):
    """E=150, L=128: FP32 -> E4M3FN narrowing under the dynamic mask —
    exact RNE+SAT codes on the valid lanes, tail sentinel byte kept."""
    E, lanes = 150, 128
    P = _ceil_to_lanes(E, lanes)
    values = [-16.0 + i * 0.125 for i in range(P)]
    a = torch.tensor(values, dtype=torch.float32)
    c_init = torch.full((P,), 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn)
    kernel = tilelang.compile(_k1d_fp8_narrow(E, P, lanes), target="pto", out_idx=-1)
    out = kernel(a.to(npu), c_init.to(npu))
    torch.npu.synchronize()
    raw = out.cpu().view(torch.uint8)
    expect = torch.tensor([_encode_rne(v) for v in values[:E]], dtype=torch.uint8)
    assert torch.equal(raw[0:E], expect), "narrowing codes"
    assert bool((raw[E:P] == 0x7F).all()), "narrowing tail sentinel"


@pytest.mark.pto
def test_1d_nondiv_fp8_widen(npu):
    """E=150, L=128: E4M3FN -> FP32 widening under the dynamic mask —
    every finite code decodes exactly, tail sentinel kept."""
    E, lanes = 150, 128
    P = _ceil_to_lanes(E, lanes)
    all_codes = [c for c in range(256) if c not in (0x7F, 0xFF)]
    codes = all_codes[:75] + all_codes[-75:] + [0] * (P - E)
    a = torch.tensor(codes, dtype=torch.uint8).view(torch.float8_e4m3fn)
    b_init = torch.full((P,), SENTINEL, dtype=torch.float32)
    kernel = tilelang.compile(_k1d_fp8_widen(E, P, lanes), target="pto", out_idx=-1)
    out = kernel(a.to(npu), b_init.to(npu))
    torch.npu.synchronize()
    got = out.cpu()
    expect = torch.tensor([_decode_e4m3(c) for c in codes[:E]], dtype=torch.float32)
    assert torch.equal(got[0:E], expect), "widening values"
    assert bool((got[E:P] == SENTINEL).all()), "widening tail sentinel"


@pytest.mark.pto
def test_1d_nondiv_index_value(npu):
    """E=150, L=128: the loop index feeds the value stream (a[i] + i)
    under the dynamic mask — per-lane index values exact, tail sentinel
    kept."""
    E, lanes = 150, 128
    P = _ceil_to_lanes(E, lanes)
    a = (torch.randn(P) * 1000).to(torch.int32)
    b_init = torch.full((P,), int(SENTINEL), dtype=torch.int32)
    kernel = tilelang.compile(_k1d_index_value(E, P, lanes), target="pto", out_idx=-1)
    out = kernel(a.to(npu), b_init.to(npu))
    torch.npu.synchronize()
    got = out.cpu()
    expect = a[0:E] + torch.arange(E, dtype=torch.int32)
    assert torch.equal(got[0:E], expect), "index values"
    assert bool((got[E:P] == int(SENTINEL)).all()), "index tail sentinel"


# ---------------------------------------------------------------------------
# Real Buffer stride S = P + L (distinct from the padded row stride P)
# ---------------------------------------------------------------------------


def _k2d_stride_sp(M, N, lanes, S, dtype):
    """b[i * S + j] with the real row stride S > P: distinct row data and a
    fully initialized store catch an implementation that rewrote the
    address stride to P (reads/writes would land on wrong elements)."""

    @T.prim_func
    def main(
        A: T.Tensor((M * S,), dtype),
        B_init: T.Tensor((M * S,), dtype),
        B: T.Tensor((M * S,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((M * S,), dtype)
            b = T.alloc_shared((M * S,), dtype)
            T.copy(A, a)
            T.copy(B_init, b)
            with T.SimdVF(lanes=lanes), T.Parallel(M, N) as (i, j):
                b[i * S + j] = a[i * S + j] + a[i * S + j]
            T.copy(b, B)

    return main


def _k2d_col_stride_sp(M, N, lanes, S, dtype):
    """b[j * S + i] with the real row stride S > P_M (i-continuous)."""

    @T.prim_func
    def main(
        A: T.Tensor((N * S,), dtype),
        B_init: T.Tensor((N * S,), dtype),
        B: T.Tensor((N * S,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((N * S,), dtype)
            b = T.alloc_shared((N * S,), dtype)
            T.copy(A, a)
            T.copy(B_init, b)
            with T.SimdVF(lanes=lanes), T.Parallel(M, N) as (i, j):
                b[j * S + i] = a[j * S + i] + a[j * S + i]
            T.copy(b, B)

    return main


@pytest.mark.pto
def test_2d_j_continuous_stride_sp(npu):
    """(4,100) j-continuous with real row stride S = P + L = 256: reads and
    writes must use S (not the fragment's P), rows carry distinct data and
    each row's [N, S) keeps its sentinel."""
    M, N, lanes = 4, 100, 128
    P = _ceil_to_lanes(N, lanes)
    S = P + lanes
    a_rows = torch.randn(M, N) * 2
    a = torch.zeros(M, S)
    a[:, 0:N] = a_rows  # padding readable as zeros
    b_init = torch.full((M * S,), SENTINEL, dtype=torch.float32)
    kernel = tilelang.compile(_k2d_stride_sp(M, N, lanes, S, "float32"), target="pto", out_idx=-1)
    out = kernel(a.to(npu).flatten(), b_init.to(npu))
    torch.npu.synchronize()
    got = out.cpu().reshape(M, S)
    expect_row = a_rows + a_rows
    assert torch.equal(got[:, 0:N], expect_row), "row values"
    assert bool((got[:, N:S] == SENTINEL).all()), "row sentinel through S"


@pytest.mark.pto
def test_2d_i_continuous_stride_sp(npu):
    """(100,4) i-continuous with real row stride S = P_M + L = 256: reads
    and writes must use S, rows carry distinct data and each row's (M, S)
    tail keeps its sentinel."""
    M, N, lanes = 100, 4, 128
    P = _ceil_to_lanes(M, lanes)
    S = P + lanes
    a_rows = torch.randn(N, M) * 2
    a = torch.zeros(N, S)
    a[:, 0:M] = a_rows
    b_init = torch.full((N * S,), SENTINEL, dtype=torch.float32)
    kernel = tilelang.compile(_k2d_col_stride_sp(M, N, lanes, S, "float32"), target="pto", out_idx=-1)
    out = kernel(a.to(npu).flatten(), b_init.to(npu))
    torch.npu.synchronize()
    got = out.cpu().reshape(N, S)
    expect_row = a_rows + a_rows
    assert torch.equal(got[:, 0:M], expect_row), "column values"
    assert bool((got[:, M:S] == SENTINEL).all()), "column sentinel through S"


# ---------------------------------------------------------------------------
# 2D non-divisible mixed-width with index-as-value
# ---------------------------------------------------------------------------


def _k2d_nondiv_mixed_j(M, N, lanes):
    """c[i, j] = i32(a[i, j]) + i + j — j-vectorized, int16->int32
    widening, both loop indices as values on a non-divisible j."""

    P = _ceil_to_lanes(N, lanes)

    @T.prim_func
    def main(
        A: T.Tensor((M * P,), "int16"),
        C_init: T.Tensor((M * P,), "int32"),
        C: T.Tensor((M * P,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((M * P,), "int16")
            c = T.alloc_shared((M * P,), "int32")
            T.copy(A, a)
            T.copy(C_init, c)
            with T.SimdVF(lanes=lanes), T.Parallel(M, N) as (i, j):
                c[i * P + j] = T.Cast("int32", a[i * P + j]) + i + j
            T.copy(c, C)

    return main


def _k2d_nondiv_mixed_i(M, N, lanes):
    """c[j, i] = i32(a[j, i]) + i + j — i-continuous, non-divisible M,
    same widening + index-value combination."""

    P = _ceil_to_lanes(M, lanes)

    @T.prim_func
    def main(
        A: T.Tensor((N * P,), "int16"),
        C_init: T.Tensor((N * P,), "int32"),
        C: T.Tensor((N * P,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((N * P,), "int16")
            c = T.alloc_shared((N * P,), "int32")
            T.copy(A, a)
            T.copy(C_init, c)
            with T.SimdVF(lanes=lanes), T.Parallel(M, N) as (i, j):
                c[j * P + i] = T.Cast("int32", a[j * P + i]) + i + j
            T.copy(c, C)

    return main


def _k2d_nondiv_fp8_narrow_ij(f8, M, N, lanes):
    """c[j, i] = fp8(a[j, i] * s[i] + t[j]) — FP32 -> FP8 narrowing on a
    non-divisible vectorized (i) extent, with lane-varying s[i] and the
    lane-uniform t[j] broadcast."""

    P = _ceil_to_lanes(M, lanes)

    @T.prim_func
    def main(
        A: T.Tensor((N, P), "float32"),
        S: T.Tensor((P,), "float32"),
        Tt: T.Tensor((N,), "float32"),
        C_init: T.Tensor((N, P), f8),
        C: T.Tensor((N, P), f8),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((N, P), "float32")
            s = T.alloc_shared((P,), "float32")
            t = T.alloc_shared((N,), "float32")
            c = T.alloc_shared((N, P), f8)
            T.copy(A, a)
            T.copy(S, s)
            T.copy(Tt, t)
            T.copy(C_init, c)
            with T.SimdVF(lanes=lanes):
                for i, j in T.Parallel(M, N):
                    c[j, i] = T.Cast(f8, a[j, i] * s[i] + t[j])
            T.copy(c, C)

    return main


@pytest.mark.pto
def test_2d_nondiv_mixed_width_index_j(npu):
    """(4,150) j-vectorized, L=128 (Q=2, tail 22): int16->int32 widening
    with both indices as values under the per-chunk dynamic mask — valid
    lanes exact, per-row tail sentinel kept."""
    M, N, lanes = 4, 150, 128
    P = _ceil_to_lanes(N, lanes)
    a_rows = torch.randint(-1000, 1000, (M, N), dtype=torch.int16)
    a = torch.zeros(M, P, dtype=torch.int16)
    a[:, 0:N] = a_rows  # padding lanes read as zeros; never stored (mask)
    c_init = torch.full((M * P,), int(SENTINEL), dtype=torch.int32)
    kernel = tilelang.compile(_k2d_nondiv_mixed_j(M, N, lanes), target="pto", out_idx=-1)
    out = kernel(a.to(npu).flatten(), c_init.to(npu))
    torch.npu.synchronize()
    got = out.cpu().reshape(M, P)
    expect = a_rows.to(torch.int32) + torch.arange(M, dtype=torch.int32)[:, None] + torch.arange(N, dtype=torch.int32)[None, :]
    assert torch.equal(got[:, 0:N], expect), "row values"
    assert bool((got[:, N:P] == int(SENTINEL)).all()), "row tail sentinel"


@pytest.mark.pto
def test_2d_nondiv_mixed_width_index_i(npu):
    """(150,4) i-continuous, L=128 (Q=2, tail 22): the symmetric direction
    — valid lanes exact, per-row tail sentinel kept."""
    M, N, lanes = 150, 4, 128
    P = _ceil_to_lanes(M, lanes)
    a_rows = torch.randint(-1000, 1000, (N, M), dtype=torch.int16)
    a = torch.zeros(N, P, dtype=torch.int16)
    a[:, 0:M] = a_rows
    c_init = torch.full((N * P,), int(SENTINEL), dtype=torch.int32)
    kernel = tilelang.compile(_k2d_nondiv_mixed_i(M, N, lanes), target="pto", out_idx=-1)
    out = kernel(a.to(npu).flatten(), c_init.to(npu))
    torch.npu.synchronize()
    got = out.cpu().reshape(N, P)
    expect = a_rows.to(torch.int32) + torch.arange(M, dtype=torch.int32)[None, :] + torch.arange(N, dtype=torch.int32)[:, None]
    assert torch.equal(got[:, 0:M], expect), "column values"
    assert bool((got[:, M:P] == int(SENTINEL)).all()), "column tail sentinel"


@pytest.mark.pto
def test_2d_nondiv_fp8_narrow(npu):
    """M=150, L=128 (Q=2, tail 22) on the vectorized i dim, N=4 rows:
    FP32 -> E4M3FN narrowing with lane-varying s[i] and a lane-uniform t[j]
    broadcast — exact RNE codes on in-range valid lanes, overflow lanes
    accept the saturated or NaN encoding (environment-defined under the
    global CTRL saturation mode), per-row tail sentinel byte kept."""
    M, N, lanes = 150, 4, 128
    P = _ceil_to_lanes(M, lanes)

    a_vals = [-6.0 + 0.25 * k for k in range(N * M)]
    s_vals = [0.5 + 0.25 * k for k in range(M)]
    t_vals = [-2.0 + 0.5 * k for k in range(N)]

    a = torch.zeros(N, P, dtype=torch.float32)
    a[:, :M] = torch.tensor(a_vals, dtype=torch.float32).reshape(N, M)
    s_t = torch.zeros(P, dtype=torch.float32)
    s_t[:M] = torch.tensor(s_vals, dtype=torch.float32)
    t_t = torch.tensor(t_vals, dtype=torch.float32)
    c_init = torch.full((N, P), 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn)

    kernel = tilelang.compile(_k2d_nondiv_fp8_narrow_ij(T.float8_e4m3fn, M, N, lanes), target="pto", out_idx=-1)
    src = kernel.get_kernel_source()
    vcvt = [ln.strip() for ln in src.split("\n") if "vmi.vcvt" in ln and "=" in ln]
    assert len(vcvt) == 1, f"expected exactly one conversion:\n{src}"
    assert "to_dtype=pto.f8e4m3" in vcvt[0], vcvt[0]
    assert "vmi.vbrc" in src, src
    assert "create_mask" in src and "remaining" in src, src

    out = kernel(a.to(npu), s_t.to(npu), t_t.to(npu), c_init.to(npu))
    torch.npu.synchronize()
    raw = out.cpu().view(torch.uint8)

    ref_t = a[:, :M] * s_t[:M].reshape(1, M) + t_t.reshape(N, 1)
    expect = torch.tensor([_encode_rne(v) for v in ref_t.reshape(-1).tolist()], dtype=torch.uint8).reshape(N, M)
    got = raw[:, :M]
    # Overflow lanes (|v| > 448, the e4m3fn max finite) are environment-
    # defined under the global CTRL saturation mode forced by
    # pto.init_core(): accept the saturated (0x7E/0xFE) or NaN (0x7F/0xFF)
    # encoding (NaN does not necessarily preserve the sign); in-range lanes
    # keep exact RNE+SAT codes. No existing fp8 test in the repo covers SAT
    # overflow, so neither does this one.
    overflow = ref_t.abs() > 448.0
    assert torch.equal(got[~overflow], expect[~overflow]), "in-range narrowing codes"
    if bool(overflow.any()):
        ov = got[overflow]
        ok = (ov == 0x7E) | (ov == 0x7F) | (ov == 0xFE) | (ov == 0xFF)
        assert bool(ok.all()), "overflow lanes must be saturated or NaN encodings"
    assert bool((raw[:, M:P] == 0x7F).all()), "per-row tail sentinel"
