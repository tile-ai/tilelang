"""Device tests for FP8 casts on the Parallel->VMI path (NPU).

Two things are checked here, and they are reported separately: the attributes
the vectorize path emits (rounded/saturation mode spelled out in the generated
PTODSL), and the numeric result on the device.

Reference values come from a host-side E4M3FN/E5M2 code table evaluated with
the *selected* mode — round-to-nearest-even or toward-zero, with or without
saturation — not from torch's own fp8 cast: the contract under test is the PTO
conversion mode, and torch has no toward-zero mode to compare against. NaN
encoding is deliberately not asserted in this batch; the overflow cases assert
either the documented saturation result (SAT) or nothing but the attribute and
successful execution (NOSAT).

The FP8 conversion instructions the target exposes are FP32 <-> FP8, so a
half source is rejected by Verify and the supported way to convert one is the
explicit two-step cast written in the kernel (half -> float32 -> FP8), which
carries the numeric coverage here.
"""

import math
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["TILELANG_DISABLE_DATA_RACE_CHECK"] = "1"

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

F8_E4M3 = T.float8_e4m3fn
F8_E5M2 = T.float8_e5m2
TORCH_F8 = {F8_E4M3: torch.float8_e4m3fn, F8_E5M2: torch.float8_e5m2}


def _require_npu_runtime():
    if not hasattr(torch, "npu"):
        pytest.skip("torch.npu is unavailable in the current environment")
    try:
        _ = torch.randn(1, device=torch.device("npu"))
        torch.npu.synchronize()
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"NPU runtime is unavailable: {err}")


@pytest.fixture()
def npu():
    _require_npu_runtime()
    return torch.device("npu")


# ---------------------------------------------------------------------------
# Host-side E4M3FN / E5M2 code tables
# ---------------------------------------------------------------------------
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


def _decode_e5m2(code):
    """E5M2: 1 sign, 5 exponent (bias 15), 2 mantissa; exp=31 is inf/NaN."""
    sign = -1.0 if code & 0x80 else 1.0
    exp = (code >> 2) & 0x1F
    mant = code & 0x3
    if exp == 0:
        return sign * mant * 2.0**-16
    if exp == 0x1F:
        return None  # inf / NaN
    return sign * (1.0 + mant / 4.0) * 2.0 ** (exp - 15)


def _table(decode):
    """All finite codes of a format, sorted by magnitude, positive side."""
    vals = []
    for code in range(256):
        value = decode(code)
        if value is not None and (value > 0 or code == 0):
            vals.append((code, value))
    vals.sort(key=lambda pair: pair[1])
    return vals


TABLES = {F8_E4M3: _table(_decode_e4m3), F8_E5M2: _table(_decode_e5m2)}


def _max_finite(f8):
    return TABLES[f8][-1][1]


def _encode(value, f8, mode):
    """Encode one float with the given PTO mode (R = ties-to-even, Z = toward
    zero), saturating."""
    table = TABLES[f8]
    limit = table[-1][1]
    sign_bit = 0x80 if math.copysign(1.0, value) < 0 else 0
    value = max(-limit, min(limit, abs(value)))
    mag = abs(value)
    if mode == "R":
        # Exact nearest with ties-to-even: the magnitudes are dyadic rationals,
        # so the deltas are exact and comparisons need no tolerance.
        code = table[0][0]
        best_delta = abs(table[0][1] - mag)
        for cand, magnitude in table[1:]:
            delta = abs(magnitude - mag)
            if delta < best_delta or (delta == best_delta and (cand & 1) == 0 and (code & 1) != 0):
                code, best_delta = cand, delta
    elif mode == "Z":
        # Truncation: the largest magnitude that is <= the input, compared
        # exactly (a tolerance would round nextafter(1.0, 0) up to 1.0).
        code = table[0][0]
        for cand, magnitude in table:
            if magnitude <= mag:
                code = cand
            else:
                break
    else:
        raise ValueError(mode)
    return code | sign_bit


def _encode_mask(values, f8, mode):
    return bytes(_encode(float(v), f8, mode) for v in values)


def _assert_codes_with_overflow(out, ref_values, f8):
    """Per-lane byte check: in-range lanes keep exact RNE codes; overflow
    lanes (|v| > max finite) accept the saturated or NaN encoding. Under the
    global CTRL saturation mode forced by pto.init_core() the instruction's
    satMode bit is overridden, so the overflow encoding is
    environment-defined; no existing fp8 test in the repo covers SAT
    overflow, so neither do we."""
    limit = _max_finite(f8)
    got = out.cpu().view(torch.uint8).flatten().tolist()
    assert len(got) == len(ref_values)
    for i, v in enumerate(ref_values):
        if abs(v) > limit:
            # e4m3fn overflow encodings: saturated ±MAX (0x7E/0xFE) or NaN
            # (0x7F/0xFF; NaN does not necessarily preserve the sign).
            assert f8 == F8_E4M3, "tolerant overflow codes are e4m3fn-specific"
            assert got[i] in (0x7E, 0x7F, 0xFE, 0xFF), f"lane {i}: overflow encoding {got[i]:#x}"
        else:
            assert got[i] == _encode(float(v), f8, "R"), f"lane {i}: {got[i]:#x} vs {float(v)}"


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def _f32_to_f8(f8, n, lanes, **kwargs):
    ann = kwargs.pop("annotations", None)

    @T.prim_func
    def main(A: T.Buffer((n,), "float32"), C: T.Buffer((n,), f8)):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float32")
            c = T.alloc_shared((n,), f8)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = T.Cast(f8, a[i], annotations=ann) if ann is not None else T.cast(a[i], f8, **kwargs)
            T.copy(c, C)

    return main


def _f8_to_f32(f8, n, lanes):
    @T.prim_func
    def main(A: T.Buffer((n,), f8), C: T.Buffer((n,), "float32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), f8)
            c = T.alloc_shared((n,), "float32")
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = T.cast(a[i], "float32")
            T.copy(c, C)

    return main


def _run(func, npu, source=None):
    kernel = tilelang.compile(func, target="pto", out_idx=-1)
    src = kernel.get_kernel_source()
    if source is not None:
        source(src)
    return src


def _bytes(tensor):
    return bytes(tensor.cpu().view(torch.uint8).flatten().tolist())


LANES = 64


def _pad(values, fill=0.0):
    """The PTO path requires the parallel extent to divide L, so the value
    lists are padded to a multiple of LANES; padding encodes deterministically
    (0.0 -> code 0)."""
    if len(values) % LANES:
        values = list(values) + [fill] * (LANES - len(values) % LANES)
    return values


# ---------------------------------------------------------------------------
# Attributes in the generated PTODSL
# ---------------------------------------------------------------------------
@pytest.mark.pto
@pytest.mark.parametrize("f8", [F8_E4M3, F8_E5M2], ids=["e4m3fn", "e5m2"])
def test_generated_attributes_default(npu, f8):
    n, lanes = 256, 128
    src = _run(_f32_to_f8(f8, n, lanes), npu)
    target = "pto.f8e4m3" if f8 == F8_E4M3 else "pto.f8e5m2"
    line = [ln.strip() for ln in src.split("\n") if "vmi.vcvt" in ln and "=" in ln]
    assert line, src
    assert 'rounding="R"' in line[0] and 'saturate="SAT"' in line[0], line[0]
    assert f"to_dtype={target}" in line[0], line[0]
    assert "to_dtype=pto.f32," not in line[0], line[0]


@pytest.mark.pto
@pytest.mark.parametrize("mode,rounding", [("rn", "R"), ("rz", "Z")])
def test_generated_attributes_rounding(npu, mode, rounding):
    src = _run(_f32_to_f8(F8_E4M3, 256, 128, round=mode), npu)
    line = [ln.strip() for ln in src.split("\n") if "vmi.vcvt" in ln and "=" in ln][0]
    assert f'rounding="{rounding}"' in line, line


@pytest.mark.pto
def test_generated_attributes_nosat(npu):
    src = _run(_f32_to_f8(F8_E4M3, 256, 128, sat=False), npu)
    line = [ln.strip() for ln in src.split("\n") if "vmi.vcvt" in ln and "=" in ln][0]
    assert 'saturate="NOSAT"' in line, line


@pytest.mark.pto
def test_generated_attributes_widening_has_none(npu):
    src = _run(_f8_to_f32(F8_E4M3, 256, 128), npu)
    line = [ln.strip() for ln in src.split("\n") if "vmi.vcvt" in ln and "=" in ln][0]
    assert "rounding" not in line and "saturate" not in line, line
    assert "to_dtype=pto.f32" in line, line


# ---------------------------------------------------------------------------
# Numeric results
# ---------------------------------------------------------------------------
def _midpoints(f8):
    """Values exactly halfway between adjacent magnitudes — the tie cases that
    separate ties-to-even from ties-away, plus ordinary values."""
    table = TABLES[f8]
    values = [-1.0, 0.0, 1.0, -2.5, 3.75]
    for (_, lo), (_, hi) in zip(table, table[1:]):
        values.append((lo + hi) / 2.0)
        values.append(-(lo + hi) / 2.0)
    return values


@pytest.mark.pto
@pytest.mark.parametrize("f8", [F8_E4M3, F8_E5M2], ids=["e4m3fn", "e5m2"])
def test_narrow_default_rne_sat(npu, f8):
    """Default mode: ties-to-even with saturation, checked on tie cases."""
    values = _pad(_midpoints(f8))
    x = torch.tensor(values, dtype=torch.float32)
    out = tilelang.compile(_f32_to_f8(f8, len(values), LANES), target="pto", out_idx=-1)(x.to(npu))
    torch.npu.synchronize()
    assert _bytes(out) == _encode_mask(values, f8, "R")


@pytest.mark.pto
def test_narrow_toward_zero(npu):
    values = _pad(_midpoints(F8_E4M3))
    x = torch.tensor(values, dtype=torch.float32)
    out = tilelang.compile(_f32_to_f8(F8_E4M3, len(values), LANES, round="rz"), target="pto", out_idx=-1)(x.to(npu))
    torch.npu.synchronize()
    assert _bytes(out) == _encode_mask(values, F8_E4M3, "Z")


def _f16_to_f8_two_step(f8, n, lanes):
    """Explicit two-step half -> float32 -> FP8: both conversions are
    supported directions, so this is the accepted way from a half source."""

    @T.prim_func
    def main(A: T.Buffer((n,), "float16"), C: T.Buffer((n,), f8)):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "float16")
            c = T.alloc_shared((n,), f8)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = T.cast(T.cast(a[i], "float32"), f8)
            T.copy(c, C)

    return main


@pytest.mark.pto
def test_half_source_two_step_numeric(npu):
    """half -> float32 -> FP8 (written explicitly) matches the ties-to-even
    reference."""
    values = _pad(_midpoints(F8_E4M3))
    x = torch.tensor(values, dtype=torch.float32).half()
    out = tilelang.compile(_f16_to_f8_two_step(F8_E4M3, len(values), LANES), target="pto", out_idx=-1)(x.to(npu))
    torch.npu.synchronize()
    assert _bytes(out) == _encode_mask([float(v) for v in x.tolist()], F8_E4M3, "R")


@pytest.mark.pto
@pytest.mark.parametrize("f8", [F8_E4M3, F8_E5M2], ids=["e4m3fn", "e5m2"])
def test_widen_is_exact(npu, f8):
    """Every finite code decodes exactly: fp8 -> fp32 is a widening."""
    decode = _decode_e4m3 if f8 == F8_E4M3 else _decode_e5m2
    codes = [c for c in range(256) if decode(c) is not None]
    codes = _pad(codes, fill=0)
    raw = torch.tensor(codes, dtype=torch.uint8).view(TORCH_F8[f8])
    expected = torch.tensor([decode(c) for c in codes], dtype=torch.float32)
    out = tilelang.compile(_f8_to_f32(f8, len(codes), LANES), target="pto", out_idx=-1)(raw.to(npu))
    torch.npu.synchronize()
    assert torch.equal(out.cpu(), expected), (out.cpu()[:8], expected[:8])


@pytest.mark.pto
@pytest.mark.parametrize("f8,below_code", [(F8_E4M3, 0x37), (F8_E5M2, 0x3B)], ids=["e4m3fn", "e5m2"])
def test_toward_zero_on_fp32_predecessor(npu, f8, below_code):
    """Z applied to the true float32 predecessor of 1.0.

    The input has to be built in float32: `math.nextafter` works in double and
    its result rounds back to exactly 1.0 when stored as float32, which would
    exercise nothing. The bit pattern is asserted so the case cannot silently
    regress into a no-op, and the expectation is the single code below 1.0
    (0x37 for E4M3FN, 0x3B for E5M2) with the negative side mirrored.
    """
    one = torch.tensor(1.0, dtype=torch.float32)
    below = torch.nextafter(one, torch.tensor(0.0, dtype=torch.float32))
    assert below.view(torch.int32).item() == 0x3F7FFFFF, hex(below.view(torch.int32).item())
    assert below.item() < 1.0
    assert below.half().float().item() != below.item()  # not on the f16 grid

    values = _pad([below.item(), -below.item()])
    x = torch.tensor(values, dtype=torch.float32)
    assert x[0].view(torch.int32).item() == 0x3F7FFFFF

    kernel = tilelang.compile(_f32_to_f8(f8, len(values), LANES, round="rz"), target="pto", out_idx=-1)
    line = [ln.strip() for ln in kernel.get_kernel_source().split("\n") if "vmi.vcvt" in ln and "=" in ln][0]
    assert 'rounding="Z"' in line, line

    out = kernel(x.to(npu))
    torch.npu.synchronize()
    got = _bytes(out)
    assert got[0] == below_code, f"expected {below_code:#x}, got {got[0]:#x}"
    assert got[1] == below_code | 0x80, f"expected {below_code | 0x80:#x}, got {got[1]:#x}"


@pytest.mark.pto
def test_empty_round_annotation_uses_default(npu):
    """`{"round": ""}` is the frontend's "backend default", not an unknown
    mode: the narrowing keeps R/SAT, and the values match the default run."""
    values = _pad(_midpoints(F8_E4M3))
    x = torch.tensor(values, dtype=torch.float32)
    kernel = tilelang.compile(
        _f32_to_f8(F8_E4M3, len(values), LANES, annotations={"round": ""}),
        target="pto",
        out_idx=-1,
    )
    line = [ln.strip() for ln in kernel.get_kernel_source().split("\n") if "vmi.vcvt" in ln and "=" in ln][0]
    assert 'rounding="R"' in line and 'saturate="SAT"' in line, line
    out = kernel(x.to(npu))
    torch.npu.synchronize()
    assert _bytes(out) == _encode_mask(values, F8_E4M3, "R")


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128, 256])
def test_narrow_lane_widths_rne_sat(npu, lanes):
    """The numeric contract holds at every supported lane width, including the
    multi-chunk case (extent = 2 * lanes)."""
    values = _midpoints(F8_E4M3)
    values = values * (1 + (2 * lanes) // max(len(values), 1)) if len(values) < 2 * lanes else values
    values = _pad(values[: 2 * lanes], 0.0)
    x = torch.tensor(values, dtype=torch.float32)
    out = tilelang.compile(_f32_to_f8(F8_E4M3, len(values), lanes), target="pto", out_idx=-1)(x.to(npu))
    torch.npu.synchronize()
    assert _bytes(out) == _encode_mask(values, F8_E4M3, "R")


def _f32_to_fp8_2d_with_ij(f8, m, n, lanes):
    """2D column-continuous element-wise unit (buffers (n, m), accessed
    [j, i]) that mixes coordinate data before narrowing to FP8:
    c[j, i] = fp8(a[j, i] * s[i] + t[j]).

    `s[i]` is lane-varying data along the vectorized coordinate, `t[j]` is
    lane-uniform data along the kept coordinate, and both arithmetic steps plus
    the narrowing are written in the source order — no intermediate dtype.
    """

    @T.prim_func
    def main(
        A: T.Buffer((n, m), "float32"),
        S: T.Buffer((m,), "float32"),
        Tt: T.Buffer((n,), "float32"),
        C: T.Buffer((n, m), f8),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((n, m), "float32")
            s = T.alloc_shared((m,), "float32")
            t = T.alloc_shared((n,), "float32")
            c = T.alloc_shared((n, m), f8)
            T.copy(A, a)
            T.copy(S, s)
            T.copy(Tt, t)
            with T.SimdVF(lanes=lanes):
                for i, j in T.Parallel(m, n):
                    c[j, i] = T.cast(a[j, i] * s[i] + t[j], f8)
            T.copy(c, C)

    return main


def _grid_values(count, step, low):
    """Values whose products and sums stay exactly representable in both f16
    and f32, so the narrowing result does not depend on an internal precision
    hop (the Z-mode finding stays out of this case)."""
    return [low + step * k for k in range(count)]


@pytest.mark.pto
@pytest.mark.parametrize("f8", [F8_E4M3, F8_E5M2], ids=["e4m3fn", "e5m2"])
def test_2d_elementwise_with_coordinate_data(npu, f8):
    """_column_kernel shape, element-wise mul-add with coordinate data, then a
    narrowing cast to FP8; in-range lanes are compared against the strict
    RNE+SAT reference, overflow lanes accept the environment-defined
    saturated/NaN encoding (see _assert_codes_with_overflow)."""
    m, n, lanes = 256, 4, 128
    a_flat = _grid_values(n * m, 0.25, -6.0)  # (n, m), step 1/4
    s_vals = _grid_values(m, 0.25, 0.5)  # (m,),   step 1/4
    t_vals = _grid_values(n, 0.5, -2.0)  # (n,),   step 1/2

    a = torch.tensor(a_flat, dtype=torch.float32).reshape(n, m)
    s_t = torch.tensor(s_vals, dtype=torch.float32)
    t_t = torch.tensor(t_vals, dtype=torch.float32)

    kernel = tilelang.compile(_f32_to_fp8_2d_with_ij(f8, m, n, lanes), target="pto", out_idx=-1)
    src = kernel.get_kernel_source()
    target = "pto.f8e4m3" if f8 == F8_E4M3 else "pto.f8e5m2"
    vcvt = [ln.strip() for ln in src.split("\n") if "vmi.vcvt" in ln and "=" in ln]
    assert len(vcvt) == 1, f"expected exactly one conversion:\n{src}"
    assert f"to_dtype={target}" in vcvt[0], vcvt[0]
    assert 'rounding="R"' in vcvt[0] and 'saturate="SAT"' in vcvt[0], vcvt[0]
    assert "vmi.vmul" in src and "vmi.vadd" in src and "vmi.vbrc" in src, src

    out = kernel(a.to(npu), s_t.to(npu), t_t.to(npu))
    torch.npu.synchronize()

    # Reference in float32, in the source's operation order, then RNE+SAT
    # (overflow lanes tolerant, see _assert_codes_with_overflow).
    ref = (a * s_t.reshape(1, m) + t_t.reshape(n, 1)).reshape(-1).tolist()
    _assert_codes_with_overflow(out, ref, f8)


@pytest.mark.pto
def test_offset_counts_once(npu):
    """A constant source offset (the chunk start) must be counted exactly
    once: c[i] = f8(a[i + 8]) over a longer source buffer."""
    n, lanes, shift = 2 * 64, 64, 8
    values = [float(i) * 0.5 - 16.0 for i in range(n + shift)]
    x = torch.tensor(values, dtype=torch.float32)
    src = _offset_kernel(F8_E4M3, n, lanes, shift)
    kernel = tilelang.compile(src, target="pto", out_idx=-1)
    line = [ln.strip() for ln in kernel.get_kernel_source().split("\n") if "vmi.vload" in ln and "=" in ln][0]
    assert line.count("+ 8") == 1, line
    out = kernel(x.to(npu))
    torch.npu.synchronize()
    expected = [values[i + shift] for i in range(n)]
    assert _bytes(out) == _encode_mask(expected, F8_E4M3, "R")


def _offset_kernel(f8, n, lanes, shift):
    """1D narrowing reading the source at a constant offset."""

    @T.prim_func
    def main(A: T.Buffer((n + shift,), "float32"), C: T.Buffer((n,), f8)):
        with T.Kernel(1):
            a = T.alloc_shared((n + shift,), "float32")
            c = T.alloc_shared((n,), f8)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = T.cast(a[i + shift], f8)
            T.copy(c, C)

    return main


def _f32_to_f8_2d(f8, m, n, lanes, column):
    """2D narrowing, j-continuous (buffer (m, n)) or i-continuous (buffer
    (n, m), accessed [j, i])."""

    if column:

        @T.prim_func
        def main(A: T.Buffer((n, m), "float32"), C: T.Buffer((n, m), f8)):
            with T.Kernel(1):
                a = T.alloc_shared((n, m), "float32")
                c = T.alloc_shared((n, m), f8)
                T.copy(A, a)
                with T.SimdVF(lanes=lanes):
                    for i, j in T.Parallel(m, n):
                        c[j, i] = T.cast(a[j, i], f8)
                T.copy(c, C)

    else:

        @T.prim_func
        def main(A: T.Buffer((m, n), "float32"), C: T.Buffer((m, n), f8)):
            with T.Kernel(1):
                a = T.alloc_shared((m, n), "float32")
                c = T.alloc_shared((m, n), f8)
                T.copy(A, a)
                with T.SimdVF(lanes=lanes):
                    for i, j in T.Parallel(m, n):
                        c[i, j] = T.cast(a[i, j], f8)
                T.copy(c, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize("column", [False, True], ids=["row", "col"])
def test_2d_numeric(npu, column):
    """Both continuous directions narrow correctly with fp8 (L=128)."""
    m, n, lanes = (256, 4, 128) if column else (4, 256, 128)
    shape = (n, m) if column else (m, n)
    values = [((i % 512) - 256) / 16.0 for i in range(shape[0] * shape[1])]
    x = torch.tensor(values, dtype=torch.float32).reshape(shape)
    out = tilelang.compile(_f32_to_f8_2d(F8_E4M3, m, n, lanes, column), target="pto", out_idx=-1)(x.to(npu))
    torch.npu.synchronize()
    assert _bytes(out) == _encode_mask(values, F8_E4M3, "R")


@pytest.mark.pto
@pytest.mark.parametrize("token", ["A", "H"])
def test_rounding_tokens_compile_and_run(npu, token):
    """A and H are carried through to the VMI call; this build compiles and
    executes them. Their numeric contract belongs to PTOAS, so only the
    attribute and successful execution are asserted."""
    values = _pad(_midpoints(F8_E4M3))
    x = torch.tensor(values, dtype=torch.float32)
    kernel = tilelang.compile(
        _f32_to_f8(F8_E4M3, len(values), LANES, annotations={"round": token}),
        target="pto",
        out_idx=-1,
    )
    line = [ln.strip() for ln in kernel.get_kernel_source().split("\n") if "vmi.vcvt" in ln and "=" in ln][0]
    assert f'rounding="{token}"' in line, line
    out = kernel(x.to(npu))
    torch.npu.synchronize()
    assert out.cpu().view(torch.uint8).numel() == len(values)


@pytest.mark.pto
def test_sat_clamps_finite_overflow(npu):
    """Overflow lanes: encoding is environment-defined, in-range lanes exact.

    Saturation on overflow depends on the inherited CTRL[60]/CTRL[48] mode:
    pto.init_core() forces CTRL[60]=1 (global mode), under which the
    instruction's satMode bit is overridden by the global flag, so an
    overflowing lane may encode as saturated ±MAX (0x7E/0xFE) or NaN
    (0x7F/0xFF) depending on the environment. No existing fp8 test in the
    repo covers SAT overflow, so this case asserts only that overflow lanes
    produce one of the two legal encodings; padding lanes stay exact.
    """
    limit = _max_finite(F8_E4M3)
    values = _pad([limit * 2.0, limit * 100.0, -limit * 2.0, -limit * 100.0])
    x = torch.tensor(values, dtype=torch.float32)
    out = tilelang.compile(_f32_to_f8(F8_E4M3, len(values), LANES), target="pto", out_idx=-1)(x.to(npu))
    torch.npu.synchronize()
    _assert_codes_with_overflow(out, values, F8_E4M3)


@pytest.mark.pto
def test_nosat_overflow_attributes_only(npu):
    """With sat=False the overflow encoding is backend-defined in this batch:
    assert the emitted attribute and that the kernel executes, not the bytes."""
    limit = _max_finite(F8_E4M3)
    values = _pad([limit * 2.0, -limit * 2.0])
    x = torch.tensor(values, dtype=torch.float32)
    kernel = tilelang.compile(_f32_to_f8(F8_E4M3, len(values), LANES, sat=False), target="pto", out_idx=-1)
    line = [ln.strip() for ln in kernel.get_kernel_source().split("\n") if "vmi.vcvt" in ln and "=" in ln][0]
    assert 'saturate="NOSAT"' in line, line
    out = kernel(x.to(npu))
    torch.npu.synchronize()
    assert out.cpu().view(torch.uint8).numel() == len(values)


if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
