"""Cross-feature interaction controls: combined end-to-end device kernels.

The dedicated per-feature suites cover each contract in isolation.  This file
exercises three combinations of features in a single device run:

* a handwritten VMI kernel loads from a nonzero UB buffer start, performs an
  integer reduction, broadcasts the lane-1 result with ``vbrc(group=1)``,
  stores it through an aligned masked store and copies the result back with
  DMA.  Automatic scheduling stays enabled (the generated MTE2->V / V->MTE3
  synchronization is asserted).  Signed-negative and unsigned high-bit inputs
  are compared exactly, with physical source padding and an initialized
  output.

* an auto-vectorized ``T.Parallel`` kernel reads from a nonzero source
  offset, applies a cross-sign widening Cast (source-sign extend then target
  reinterpret), stores through a masked aligned store and copies back with
  DMA.  Both a tail (E = VL-1) and a multi-chunk extent (E = 2*VL+17) are
  covered for int8->uint16 and uint8->int32.  The source storage is
  physically padded, the destination is initialized with sentinels, and every
  logical element plus the untouched tail is checked.  The final alignment
  checker must accept the store (compilation would fail otherwise).

* SIMT, VMI and Parallel kernels are compiled in alternating orders in one
  process.  Each compilation uses fresh inputs and an independent numeric
  reference, and the generated source must retain only the expected
  per-kernel features, so storage or provenance state cannot leak between
  compilations.  Handwritten VMI is never mixed into a Parallel VF.

Every test executes on the device and is skipped when no NPU runtime is
available.
"""

import re
from contextlib import contextmanager

import pytest
import torch
import torch_npu  # noqa: F401,E402  (must import before tilelang/ptodsl)

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower

tirx = tvm.tirx
START = 131
VL = 64
REDUCE_LANES = 128
_ELEMENT_BYTES = {
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
}


def _require_npu_runtime():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("torch.npu is unavailable in the current environment")
    try:
        source = torch.arange(8, dtype=torch.float32)
        copied = source.to(torch.device("npu"))
        torch.npu.synchronize()
        assert torch.equal(copied.cpu(), source)
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"NPU runtime is unavailable in the current environment: {err}")


def _compile(func):
    return tilelang.compile(func, target="pto", out_idx=-1)


def _run(kernel, *inputs):
    device_inputs = [value.to("npu") for value in inputs]
    output = kernel(*device_inputs)
    torch.npu.synchronize()
    return output.cpu()


def _physical_tail_lanes(lanes, element_bytes):
    """Whole-vector load footprint: 256 bytes or one logical vector, whichever is larger."""
    return max(lanes, 256 // element_bytes)


def _padded_source_len(start, extent, lanes, element_bytes):
    last_chunk = ((extent - 1) // lanes) * lanes
    return start + last_chunk + _physical_tail_lanes(lanes, element_bytes)


@contextmanager
def _capture_final_module():
    captured = {}
    original = ascend_transform.VerifyPTOCodegenLegality

    def factory():
        real = original()

        def run(mod):
            captured["module"] = mod
            return real(mod)

        return run

    ascend_transform.VerifyPTOCodegenLegality = factory
    try:
        yield captured
    finally:
        ascend_transform.VerifyPTOCodegenLegality = original


def _final_vload_descriptor_offsets(func):
    """Lower to the pass boundary and return every VMI load descriptor offset."""
    target = normalize_pto_target("pto")
    with _capture_final_module() as captured, target:
        _lower(func, target=target)
    module = captured["module"]
    offsets = []

    def visit(node):
        if isinstance(node, tirx.Call) and hasattr(node.op, "name") and node.op.name == "tl.vmi.vload":
            offsets.append(node.args[0].args[2])

    for func in module.functions.values():
        if isinstance(func, tirx.PrimFunc):
            tirx.stmt_functor.post_order_visit(func.body, visit)
    assert offsets, "expected a lowered VMI load"
    return offsets


def _constant_term(expr):
    variables = []

    def visit(node):
        if isinstance(node, tirx.Var):
            variables.append(node)

    tirx.stmt_functor.post_order_visit(expr, visit)
    substitution = {var: tirx.IntImm(str(var.dtype), 0) for var in variables}
    simplified = tvm.arith.Analyzer().simplify(tirx.stmt_functor.substitute(expr, substitution))
    assert isinstance(simplified, tirx.IntImm), simplified
    return int(simplified.value)


# ---------------------------------------------------------------------------
# Group 1: nonzero-start VMI load, integer reduce, group-1 broadcast,
# aligned masked store, DMA; automatic scheduling enabled.
# ---------------------------------------------------------------------------
def _vmi_reduce_kernel(dtype, lanes=REDUCE_LANES, start=START):
    storage = start + lanes + 64

    # The whole padded source is filled by one aligned DMA; the VMI load then
    # reads at the nonzero start, which the backend handles with
    # vldas/vldus.  A DMA directly into an unaligned UB region is not a legal
    # MTE non-atomic access and is not what this control is about.
    @T.prim_func
    def main(
        A: T.Tensor((storage,), dtype),
        S: T.Tensor((lanes,), dtype),
        C: T.Tensor((lanes,), dtype),
    ):
        with T.Kernel(1):
            work = T.alloc_shared((storage,), dtype)
            out = T.alloc_shared((lanes,), dtype)
            T.copy(A, work)
            T.copy(S, out)
            with T.SimdVF():
                mask = T.vmi.create_mask(lanes, size=lanes)
                x = T.vmi.vload(work[start], size=lanes)
                reduced = T.vmi.vcmax(x, mask)
                broadcast = T.vmi.vbrc(reduced, size=lanes, group=1)
                T.vmi.vstore(broadcast, out[0], mask)
            T.copy(out, C)

    return main


def _padded_reduce_input(values, torch_dtype):
    padded = torch.zeros(START + REDUCE_LANES + 64, dtype=torch_dtype)
    padded[START : START + REDUCE_LANES] = values
    return padded


def _int32_reduce_input(round_index):
    generator = torch.Generator().manual_seed(0xA101 + round_index)
    values = torch.randint(-(2**31), 2**31 - 1, (REDUCE_LANES,), generator=generator).to(torch.int32)
    values[0] = -(2**31)
    values[1] = 2**31 - 1
    values[2] = -1
    values[3] = 0
    values[4] = 1
    return values


def _uint32_reduce_input(round_index):
    generator = torch.Generator().manual_seed(0xA102 + round_index)
    values = torch.randint(0, 2**31 - 1, (REDUCE_LANES,), generator=generator).to(torch.uint32)
    values[0] = 2**32 - 1
    values[1] = 2**31  # INT32_MAX + 1, high bit set
    values[2] = 2**31 - 1
    values[3] = 0
    return values


@pytest.mark.parametrize(
    ("dtype", "torch_dtype", "sentinel", "make_input"),
    [
        ("int32", torch.int32, -7, _int32_reduce_input),
        ("uint32", torch.uint32, 7, _uint32_reduce_input),
    ],
    ids=["int32-signed-negative", "uint32-high-bit"],
)
def test_nonzero_start_integer_reduce_broadcast_store(dtype, torch_dtype, sentinel, make_input):
    _require_npu_runtime()
    kernel = _compile(_vmi_reduce_kernel(dtype))
    source = kernel.get_kernel_source()
    load_line = next(line for line in source.splitlines() if "vmi.vload" in line)
    assert "addptr" in load_line and str(START) in load_line, load_line
    broadcast_line = next(line for line in source.splitlines() if "vmi.vbrc" in line)
    assert "pto.cast(" not in broadcast_line, broadcast_line
    assert "vmi.vcmax" in source, source
    assert "vmi.create_mask" in source, source
    assert 'set_flag("MTE2", "V"' in source, source
    assert 'wait_flag("MTE2", "V"' in source, source
    assert 'set_flag("V", "MTE3"' in source, source
    assert 'wait_flag("V", "MTE3"' in source, source

    for round_index in range(2):
        values = make_input(round_index)
        sentinels = torch.full((REDUCE_LANES,), sentinel, dtype=torch_dtype)
        output = _run(kernel, _padded_reduce_input(values, torch_dtype), sentinels)
        expected = torch.full(
            (REDUCE_LANES,),
            max(int(value) for value in values.tolist()),
            dtype=torch_dtype,
        )
        torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Group 2: automatic Parallel nonzero source access, cross-sign widening
# Cast (extend then reinterpret), aligned masked store, DMA; tail and
# multi-chunk extents with padded source and tail sentinels.
# ---------------------------------------------------------------------------
_CAST_CASES = [
    ("int8", "uint16", torch.int8, torch.uint16, "pto.si16", "pto.ui16", 30000),
    ("uint8", "int32", torch.uint8, torch.int32, "pto.ui32", "pto.si32", -999999),
]
_EXTENTS = [VL - 1, 2 * VL + 17]


def _parallel_cast_kernel(src, dst, extent, lanes=VL, start=START, read_start=None):
    # Continuous vloads cover whole physical vectors: the source storage
    # covers 256 bytes or one logical VL, whichever is larger, after the
    # last chunk base.  ``read_start`` is used only by the lower-only
    # descriptor control, so both kernels keep identical shapes.
    source_len = _padded_source_len(start, extent, lanes, _ELEMENT_BYTES[src])
    out_len = extent + 2 * lanes
    read = start if read_start is None else read_start

    @T.prim_func
    def main(
        A: T.Tensor((source_len,), src),
        S: T.Tensor((out_len,), dst),
        C: T.Tensor((out_len,), dst),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((source_len,), src)
            c = T.alloc_shared((out_len,), dst)
            T.copy(A, a)
            T.copy(S, c)
            with T.SimdVF(lanes=lanes):
                for j in T.Parallel(extent):
                    c[j] = T.cast(a[read + j], dst)
            T.copy(c, C)

    return main


def _source_values(src, torch_src, source_len):
    if src == "int8":
        values = torch.arange(source_len) * 37 % 251 - 125
        return values.to(torch_src)
    values = torch.arange(source_len) * 53 % 256
    return values.to(torch_src)


@pytest.mark.parametrize(
    ("src", "dst", "torch_src", "torch_dst", "intermediate", "target", "sentinel"),
    _CAST_CASES,
    ids=["int8_to_uint16", "uint8_to_int32"],
)
@pytest.mark.parametrize("extent", _EXTENTS, ids=["tail", "multi_chunk"])
def test_parallel_nonzero_source_cross_sign_cast_store(src, dst, torch_src, torch_dst, intermediate, target, sentinel, extent):
    _require_npu_runtime()
    kernel = _compile(_parallel_cast_kernel(src, dst, extent))
    source = kernel.get_kernel_source()
    load_line = next(line for line in source.splitlines() if "vmi.vload" in line)
    assert "addptr" in load_line, load_line
    # Exact final-descriptor check: lower the same-shape kernel once with the
    # source read at 0 and once at START, then verify that the constant part of
    # the final merged descriptor offset differs by exactly START elements.
    actual_offsets = _final_vload_descriptor_offsets(_parallel_cast_kernel(src, dst, extent))
    control_offsets = _final_vload_descriptor_offsets(_parallel_cast_kernel(src, dst, extent, read_start=0))
    assert len(actual_offsets) == len(control_offsets)
    for actual, control in zip(actual_offsets, control_offsets):
        assert _constant_term(actual) - _constant_term(control) == START
    cast_lines = [
        line.strip()
        for line in source.splitlines()
        if ("vmi.vcvt" in line or "vinterpret_cast" in line) and "vmi.vbrc(" not in line and "vmi.vci(" not in line
    ]
    assert len(cast_lines) == 2, cast_lines
    assert "vmi.vcvt" in cast_lines[0], cast_lines
    assert f"to_dtype={intermediate}" in cast_lines[0], cast_lines[0]
    assert "vinterpret_cast" in cast_lines[1], cast_lines
    assert f"to_dtype={target}" in cast_lines[1], cast_lines[1]
    assert "vmi.create_mask" in source, source
    store_lines = [line for line in source.splitlines() if "vmi.vstore" in line]
    assert store_lines, source
    for line in store_lines:
        assert "mask" in line, line

    source_len = _padded_source_len(START, extent, VL, _ELEMENT_BYTES[src])
    out_len = extent + 2 * VL
    values = _source_values(src, torch_src, source_len)
    sentinels = torch.full((out_len,), sentinel, dtype=torch_dst)
    output = _run(kernel, values, sentinels)
    logical = values[START : START + extent].to(torch.int64)
    if dst == "uint16":
        expected = (logical & 0xFFFF).to(torch.uint16)
    else:
        expected = logical.to(torch.int32)
    torch.testing.assert_close(output[:extent], expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(output[extent:], sentinels[extent:], rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Group 3: SIMT and VMI/Parallel compilation succession and interleaving;
# independent numeric checks for every compilation.
# ---------------------------------------------------------------------------
def _simt_copy_kernel(dtype="int32", lanes=8):
    # The destination is seeded with a distinct sentinel, so a missing SIMT
    # local store cannot leave the expected output in place.
    @T.prim_func
    def main(
        A: T.Tensor((64,), dtype),
        S: T.Tensor((64,), dtype),
        C: T.Tensor((64,), dtype),
    ):
        with T.Kernel(1):
            src = T.alloc_shared((64,), dtype)
            dst = T.alloc_shared((64,), dtype)
            T.copy(A, src)
            T.copy(S, dst)
            with T.SimtVF(threads=1):
                local = T.alloc_local((64,), dtype)
                local[T.Ramp(0, 1, lanes)] = src[T.Ramp(0, 1, lanes)]
                dst[T.Ramp(0, 1, lanes)] = local[T.Ramp(0, 1, lanes)]
            T.copy(dst, C)

    return main


def _compile_simt(round_index):
    kernel = _compile(_simt_copy_kernel())
    source = kernel.get_kernel_source()
    assert "vmi.vload" not in source, source
    assert "vmi.vstore" not in source, source
    # Same-width signless local backing plus explicit signed logical
    # adaptation on the load from and store back to the shared buffer.
    assert "pto.alloc_buffer((64,), pto.i32)" in source, source
    assert "pto.alloc_buffer((64,), pto.si32)" not in source, source
    assert re.search(
        r"pto\.cast\(pto\.load\(local, 0, contiguous=8\), pto\.si32\)",
        source,
    ), source
    assert re.search(r"pto\.cast\(pto\.load\(.*contiguous=8\), pto\.i32\)", source), source
    generator = torch.Generator().manual_seed(0x51A1 + round_index)
    values = torch.randint(-1000, 1000, (64,), generator=generator).to(torch.int32)
    sentinels = torch.full((64,), -777777, dtype=torch.int32)
    output = _run(kernel, values, sentinels)
    torch.testing.assert_close(output[:8], values[:8], rtol=0.0, atol=0.0)
    torch.testing.assert_close(output[8:], sentinels[8:], rtol=0.0, atol=0.0)


def _compile_vmi(round_index):
    kernel = _compile(_vmi_reduce_kernel("int32"))
    source = kernel.get_kernel_source()
    assert "vmi.vload" in source and "vmi.vstore" in source, source
    assert "vmi.vbrc" in source, source
    broadcast_line = next(line for line in source.splitlines() if "vmi.vbrc" in line)
    assert "pto.cast(" not in broadcast_line, broadcast_line
    values = _int32_reduce_input(round_index)
    sentinels = torch.full((REDUCE_LANES,), -7, dtype=torch.int32)
    output = _run(kernel, _padded_reduce_input(values, torch.int32), sentinels)
    expected = torch.full(
        (REDUCE_LANES,),
        max(int(value) for value in values.tolist()),
        dtype=torch.int32,
    )
    torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)


def _compile_parallel(round_index):
    extent = VL - 1
    kernel = _compile(_parallel_cast_kernel("int8", "uint16", extent))
    source = kernel.get_kernel_source()
    assert "vinterpret_cast" in source and "vmi.vload" in source, source
    assert "vmi.vstore" in source, source
    source_len = _padded_source_len(START, extent, VL, _ELEMENT_BYTES["int8"])
    out_len = extent + 2 * VL
    values = _source_values("int8", torch.int8, source_len)
    sentinels = torch.full((out_len,), 30000, dtype=torch.uint16)
    output = _run(kernel, values, sentinels)
    expected = (values[START : START + extent].to(torch.int64) & 0xFFFF).to(torch.uint16)
    torch.testing.assert_close(output[:extent], expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(output[extent:], sentinels[extent:], rtol=0.0, atol=0.0)


def test_simt_vmi_parallel_compile_sequences_do_not_leak_state():
    _require_npu_runtime()
    sequences = [
        ("simt", "vmi", "parallel"),
        ("parallel", "vmi", "simt"),
        ("vmi", "simt", "parallel"),
    ]
    builders = {
        "simt": _compile_simt,
        "vmi": _compile_vmi,
        "parallel": _compile_parallel,
    }
    for round_index, sequence in enumerate(sequences):
        for name in sequence:
            builders[name](round_index)
