"""Device acceptance: parallel value-Cast matrix.

Each case compiles through the real TileLang -> Verify/Vectorize -> PTODSL ->
PTOAS VPTO -> LLVM pipeline and executes on the NPU.  References are
computed independently on the CPU with explicit low-bit selection and
target-signedness adjustment (never from the generated kernel or backend).

Coverage:
* 22 representative integer pair cells at VL=128, direct store: all 12
  narrowing pairs, all 6 cross-signed widening pairs, 2 same-signed widening
  pairs, and 2 same-width cross-sign reinterpret pairs;
* exact named boundary values through the full pipeline;
* target-signed continued consumption (same-sign widening of the converted
  target) for representative 8/16-bit sources;
* one tail extent (2VL+17) with padded physical source and output sentinels;
* scalar (lane-uniform load) integer runtime casts and a scalar literal cast;
* the scalar-float expected backend rejection.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["TILELANG_DISABLE_DATA_RACE_CHECK"] = "1"

import torch_npu  # noqa: E402,F401  (import before tilelang/ptodsl)
import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

TORCH_DT = {
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "uint16": torch.uint16,
    "int32": torch.int32,
    "uint32": torch.uint32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
SENTINEL = {
    "int8": -99,
    "uint8": 199,
    "int16": -9999,
    "uint16": 55555,
    "int32": -999999,
    "uint32": 4000000000,
}


def _require_npu_runtime():
    assert hasattr(torch, "npu"), "torch.npu is unavailable in this environment"
    assert torch.npu.is_available(), "torch.npu is unavailable in the current environment"
    source = torch.arange(8, dtype=torch.float32)
    copied = source.to(torch.device("npu"))
    torch.npu.synchronize()
    assert torch.equal(copied.cpu(), source), "NPU host->device copy probe mismatch"


@pytest.fixture(scope="module")
def npu():
    _require_npu_runtime()
    return torch.device("npu")


def _bits(dtype):
    return int("".join(ch for ch in dtype if ch.isdigit()))


def _signed(dtype):
    return dtype.startswith("int")


def _wrap(value, bits, signed):
    value &= (1 << bits) - 1
    if signed and value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


def _convert(value, src, dst):
    """Independent reference: select the source value, truncate to dst bits,
    then interpret the result with the target signedness."""
    src_value = _wrap(int(value), _bits(src), _signed(src))
    result = src_value & ((1 << _bits(dst)) - 1)
    if _signed(dst) and result >= (1 << (_bits(dst) - 1)):
        result -= 1 << _bits(dst)
    return result


def _source_values(src, count):
    bits = _bits(src)
    signed = _signed(src)
    raw = [
        -(1 << (bits - 1)) if signed else 0,
        (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1,
        0,
        1,
        2,
        -1 if signed else (1 << (bits - 1)),
        -2 if signed else (1 << (bits - 1)) + 1,
        127,
        128,
        255,
        256,
        257,
        300,
        65535,
        65536,
        4294967295,
        0x80000000,
    ]
    values = [_wrap(value, bits, signed) for value in raw]
    return [values[index % len(values)] for index in range(count)]


def _tensor(values, dtype):
    return torch.tensor(values, dtype=TORCH_DT[dtype])


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def _cast_kernel(src, dst, lanes, extent=None):
    n = extent if extent is not None else 2 * lanes
    storage = _storage_len(src, lanes, n)

    @T.prim_func
    def main(A: T.Buffer((storage,), src), C: T.Buffer((storage,), dst)):
        with T.Kernel(1):
            a = T.alloc_shared((storage,), src)
            c = T.alloc_shared((storage,), dst)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = T.cast(a[i], dst)
            T.copy(c, C)

    return main, storage


def _consumer_kernel(src, mid, out, lanes):
    n = 2 * lanes
    storage = _storage_len(src, lanes, n)

    @T.prim_func
    def main(A: T.Buffer((storage,), src), C: T.Buffer((storage,), out)):
        with T.Kernel(1):
            a = T.alloc_shared((storage,), src)
            c = T.alloc_shared((storage,), out)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = T.cast(T.cast(a[i], mid), out)
            T.copy(c, C)

    return main


def _scalar_cast_kernel(src, dst, lanes):
    n = lanes

    @T.prim_func
    def main(S: T.Buffer((32,), src), C: T.Buffer((n,), dst)):
        with T.Kernel(1):
            s = T.alloc_shared((32,), src)
            c = T.alloc_shared((n,), dst)
            T.copy(S, s)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    c[i] = T.cast(s[0], dst)
            T.copy(c, C)

    return main


def _physical_lanes(dtype):
    """Physical VMI vector width in elements (256-byte vectors)."""
    return 256 * 8 // _bits(dtype)


def _storage_len(src, lanes, extent):
    """Readable source storage for the real continuous-vload footprint.

    A logical vload of ``size`` lanes is declared with the full physical
    vector extent on the last chunk, so the last chunk base must be followed
    by a full physical vector even when the logical lane count is smaller.
    """
    padded = ((extent + lanes - 1) // lanes) * lanes
    return max(padded, padded - lanes + _physical_lanes(src))


def _tail_kernel(src, dst, lanes, extent):
    storage = _storage_len(src, lanes, extent)

    @T.prim_func
    def main(
        A: T.Buffer((storage,), src),
        S: T.Buffer((storage,), dst),
        C: T.Buffer((storage,), dst),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((storage,), src)
            c = T.alloc_shared((storage,), dst)
            T.copy(A, a)
            T.copy(S, c)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(extent):
                    c[i] = T.cast(a[i], dst)
            T.copy(c, C)

    return main, storage


def _run_pair(npu, src, dst, lanes, extent=None):
    func, storage = _cast_kernel(src, dst, lanes, extent=extent)
    kernel = tilelang.compile(func, target="pto", out_idx=-1)
    n = extent if extent is not None else 2 * lanes
    values = _source_values(src, storage)
    expected = torch.tensor([_convert(value, src, dst) for value in values], dtype=TORCH_DT[dst])
    out = kernel(_tensor(values, src).to(npu))
    torch.npu.synchronize()
    got = out.cpu()
    assert got.dtype == expected.dtype, (src, dst, lanes, got.dtype)
    assert torch.equal(got[0:n], expected[0:n]), (
        f"{src}->{dst} VL{lanes}: value mismatch\n got={got[:16].tolist()}\n exp={expected[:16].tolist()}"
    )
    assert got.numel() == expected.numel() == storage
    return got, expected, storage


# ---------------------------------------------------------------------------
# A. Representative pair cells at VL=128: full compile + device execution
# ---------------------------------------------------------------------------
DEVICE_PAIRS = [
    # Narrowing: one vcvt with NOSAT.
    ("int16", "int8"),
    ("int16", "uint8"),
    ("uint16", "int8"),
    ("uint16", "uint8"),
    ("int32", "int8"),
    ("int32", "uint8"),
    ("int32", "int16"),
    ("int32", "uint16"),
    ("uint32", "int8"),
    ("uint32", "uint8"),
    ("uint32", "int16"),
    ("uint32", "uint16"),
    # Cross-signed widening: source-signed vcvt, then vinterpret_cast.
    ("int8", "uint16"),
    ("int8", "uint32"),
    ("uint8", "int16"),
    ("uint8", "int32"),
    ("int16", "uint32"),
    ("uint16", "int32"),
    # Same-signed widening: one vcvt.
    ("int8", "int16"),
    ("uint16", "uint32"),
    # Same-width cross-sign: one vinterpret_cast.
    ("int8", "uint8"),
    ("uint32", "int32"),
]


@pytest.mark.parametrize(("src", "dst"), DEVICE_PAIRS)
def test_integer_pair_vl_device(npu, src, dst):
    got, _, storage = _run_pair(npu, src, dst, 128)
    # The returned allocation is the readable-storage contract (physical
    # load padding included); only the logical [0, 2*VL) region is written
    # and compared.  The padding beyond it is uninitialized by design.
    assert got.numel() == storage
    assert storage >= 256


def test_reference_examples_are_exact(npu):
    cases = [
        ("int8", "uint32", -1, 4294967295),
        ("uint8", "int32", 255, 255),
        ("int32", "int8", 300, 44),
        ("int32", "uint8", -1, 255),
        ("int32", "uint8", 256, 0),
    ]
    for src, dst, value, expected in cases:
        func, storage = _cast_kernel(src, dst, 128, extent=128)
        kernel = tilelang.compile(func, target="pto", out_idx=-1)
        values = [value] * storage
        source = torch.tensor(values, dtype=TORCH_DT[src])
        out = kernel(source.to(npu))
        torch.npu.synchronize()
        got = out.cpu()
        assert got.numel() == storage
        # Only the logical extent (128) was written; the physical-padding
        # region is not part of the value contract.
        assert int(got[0]) == expected, (src, dst, value, int(got[0]))
        assert bool((got[:128] == expected).all()), (src, dst, value, got.tolist()[:8])


# ---------------------------------------------------------------------------
# B. Target-signed continued consumption
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("src", "mid", "out"),
    [
        ("int8", "uint16", "uint32"),
        ("int8", "uint8", "uint32"),
        ("uint8", "int8", "int32"),
        ("int16", "uint16", "uint32"),
    ],
)
def test_target_signed_continued_widening(npu, src, mid, out):
    """The converted target is consumed by its own same-sign widening: a
    '-' source value must not sign-extend through the unsigned target."""
    func = _consumer_kernel(src, mid, out, 128)
    kernel = tilelang.compile(func, target="pto", out_idx=-1)
    storage = _storage_len(src, 128, 256)
    values = [-1, 0, 1, 127, -128, 255, 256, 300] * 64
    values = [_wrap(value, _bits(src), _signed(src)) for value in values[:storage]]
    expected = torch.tensor(
        [_convert(_convert(v, src, mid), mid, out) for v in values[:256]],
        dtype=TORCH_DT[out],
    )
    got = kernel(torch.tensor(values, dtype=TORCH_DT[src]).to(npu)).cpu()
    torch.npu.synchronize()
    assert torch.equal(got[:256], expected), f"{src}->{mid}->{out}\n got={got[:16].tolist()}\n exp={expected[:16].tolist()}"


# ---------------------------------------------------------------------------
# C. Tail extent with padded source and output sentinels
# ---------------------------------------------------------------------------
def test_tail_extent_with_output_sentinel(npu):
    src, dst, lanes, extent = "int8", "uint32", 128, 2 * 128 + 17
    func, storage = _tail_kernel(src, dst, lanes, extent)
    kernel = tilelang.compile(func, target="pto", out_idx=-1)
    values = _source_values(src, storage)
    sentinel = SENTINEL[dst]
    c_init = torch.full((storage,), sentinel, dtype=TORCH_DT[dst])
    got = kernel(_tensor(values, src).to(npu), c_init.to(npu)).cpu()
    # `kernel` returns the third argument (C); `c_init` seeds it via S.
    torch.npu.synchronize()
    expected = torch.tensor([_convert(values[i], src, dst) for i in range(extent)], dtype=TORCH_DT[dst])
    assert torch.equal(got[0:extent], expected), f"{src}->{dst} VL{lanes} E={extent}: values"
    assert bool((got[extent:storage] == sentinel).all()), f"{src}->{dst} VL{lanes} E={extent}: tail sentinel overwritten"


# ---------------------------------------------------------------------------
# D. Scalar (lane-uniform) integer and float controls
# ---------------------------------------------------------------------------
SCALAR_INT_CASES = [
    ("int8", "uint32"),
    ("int32", "uint8"),
]


@pytest.mark.parametrize(("src", "dst"), SCALAR_INT_CASES)
def test_scalar_integer_runtime_cast(npu, src, dst):
    func = _scalar_cast_kernel(src, dst, 128)
    kernel = tilelang.compile(func, target="pto", out_idx=-1)
    for value in (-1, 0, 1, 255, 256, 300):
        source_values = [_wrap(value, _bits(src), _signed(src))] * 32
        source = torch.tensor(source_values, dtype=TORCH_DT[src])
        out = kernel(source.to(npu)).cpu()
        torch.npu.synchronize()
        expected = _convert(source_values[0], src, dst)
        assert int(out[0]) == expected, (src, dst, value, int(out[0]))
        assert bool((out == expected).all()), (src, dst, value)


def test_scalar_integer_literal_cast(npu):
    @T.prim_func
    def main(C: T.Buffer((128,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((128,), "int32")
            with T.SimdVF(lanes=128):
                for i in T.Parallel(128):
                    c[i] = T.cast(7, "int32")
            T.copy(c, C)

    kernel = tilelang.compile(main, target="pto", out_idx=-1)
    out = kernel().cpu()
    torch.npu.synchronize()
    assert bool((out == 7).all()), out.tolist()[:8]


def test_scalar_float_cast_rejected_by_device_compiler(npu):
    """The BiSheng device compiler rejects scalar float casts in an AIV loop.

    PTODSL, VPTO and LLVM emission support these conversions. The device
    compiler reports "Unsupported scalar instruction in AIV loop" with or
    without the value-Cast planner. This test checks that rejection; the
    scalar planner preserves the existing cast semantics.
    """
    with pytest.raises(Exception) as excinfo:
        tilelang.compile(_scalar_cast_kernel("float32", "float16", 128), target="pto", out_idx=-1)
    assert "Unsupported scalar instruction in AIV loop" in str(excinfo.value)
