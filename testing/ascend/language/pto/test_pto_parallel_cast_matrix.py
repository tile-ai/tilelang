"""Parallel value-Cast planning: action structure and attribute rules.

Hardware independent. Each case lowers the real TileLang PTO pipeline with
``tilelang.engine.lower.lower`` (VerifyParallelToPTO + VectorizeParallelToPTO
+ PTODSL codegen; no PTOAS/device) and inspects the generated source and the
module at the Verify boundary.

Covered contract:

* ``PlanValueCast(cast, CastContext{value_kind, logical_lanes}, reason)``
  dispatches scalar vs VMI first, then integer vs non-integer; Verify and
  Vectorize classify the current Cast input independently and consistently;
* integer VMI actions: identity 0, same-width cross-sign 1 vinterpret_cast,
  same-sign widening 1 vcvt, cross-sign widening source-sign extend + same-width
  vinterpret_cast (2), supported narrowing 1 vcvt NOSAT; all 36 integer pairs
  at VL=128 follow the table;
* the converted input is produced once and steps are emitted in order with
  continuous dtype/VL and an exact final type; vinterpret_cast participates in
  the local DCE;
* scalar plans keep ordinary scalar lowering (KeepInput / validated rebuild),
  no new scalar rounding/saturation mode, and no parallel planner is wired
  into generic codegen;
* attribute rules: accepted conversions pin the emitted rounding/saturation
  mode, and unsupported combinations are rejected with a reason (unknown keys,
  illegal sat policies, float <-> integer, f16 <-> bf16, out-of-scope FP8,
  lane-uniform FP8);
* classification scenarios: external scalars, uniform loads, Bind aliases, 2D
  kept/selected coordinates, nested casts, and scalar cast then broadcast.

Full backend compilation and the device numerical checks live in
``test_pto_parallel_cast_matrix_run.py``.
"""

from __future__ import annotations

import os
import re
import subprocess
from contextlib import contextmanager

import pytest
import torch  # noqa: F401  (torch/torch_npu must import before tilelang/ptodsl)
import torch_npu  # noqa: F401

import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower

tirx = tvm.tirx
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

INT_TYPES = ["int8", "uint8", "int16", "uint16", "int32", "uint32"]
CROSS_SIGN_WIDEN_INTERMEDIATE = {
    ("int8", "uint16"): "int16",
    ("int8", "uint32"): "int32",
    ("uint8", "int16"): "uint16",
    ("uint8", "int32"): "uint32",
    ("int16", "uint32"): "int32",
    ("uint16", "int32"): "uint32",
}
PTO_INT_NAME = {
    "int8": "pto.si8",
    "uint8": "pto.ui8",
    "int16": "pto.si16",
    "uint16": "pto.ui16",
    "int32": "pto.si32",
    "uint32": "pto.ui32",
}


def _bits(dtype):
    return int("".join(ch for ch in dtype if ch.isdigit()))


def _signed(dtype):
    return dtype.startswith("int")


def _pair_category(src, dst):
    if src == dst:
        return "identity"
    if _bits(src) == _bits(dst):
        return "reinterpret"
    if _bits(dst) > _bits(src):
        return "widen" if _signed(src) == _signed(dst) else "widen_and_reinterpret"
    return "narrow"


# ---------------------------------------------------------------------------
# Lowering / capture helpers
# ---------------------------------------------------------------------------
def _target():
    return normalize_pto_target("pto")


def _lower_source(func):
    return _lower(func, target=_target()).kernel_source


@contextmanager
def _verify_capture():
    captured = {}
    original = ascend_transform.VerifyParallelToPTO

    def factory():
        verify = original()

        def run(mod):
            captured["module"] = mod
            return verify(mod)

        return run

    ascend_transform.VerifyParallelToPTO = factory
    try:
        yield captured
    finally:
        ascend_transform.VerifyParallelToPTO = original


def _lower_capture(func):
    error = None
    with _verify_capture() as captured:
        try:
            _lower(func, target=_target())
        except Exception as exc:  # noqa: BLE001 - expected diagnostics
            error = exc
    return captured.get("module"), error


def _expect_lower_error(func, pattern):
    _, error = _lower_capture(func)
    assert error is not None, f"expected a Verify diagnostic matching {pattern!r}"
    assert re.search(pattern, str(error)), f"unexpected diagnostic: {error}"


def _cast_lines(source):
    """Lines emitted by a value-Cast plan.

    Excludes the scalar-bridge `vinterpret_cast` that wraps `vbrc` (the scalar
    signedness recovery), which is broadcasting, not a cast action.
    """
    lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if "vmi.vcvt" in stripped or ("vinterpret_cast" in stripped and "vmi.vbrc(" not in stripped and "vmi.vci(" not in stripped):
            lines.append(stripped)
    return lines


def _one(source, needle):
    hits = [line.strip() for line in source.splitlines() if needle in line]
    assert len(hits) == 1, f"expected one line containing {needle!r}, got {hits!r}"
    return hits[0]


def _count(source, needle):
    return source.count(needle)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
def _cast_kernel(src, dst, lanes, annotations=None):
    n = 2 * lanes

    @T.prim_func
    def main(A: T.Buffer((n,), src), C: T.Buffer((n,), dst)):
        with T.Kernel(1):
            a = T.alloc_shared((n,), src)
            c = T.alloc_shared((n,), dst)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    if annotations is None:
                        c[i] = T.cast(a[i], dst)
                    else:
                        c[i] = T.Cast(dst, a[i], annotations=annotations)
            T.copy(c, C)

    return main


def _ann_kernel(src, dst, lanes, annotations, *, uniform=False):
    n = lanes

    @T.prim_func
    def main(A: T.Buffer((n,), src), C: T.Buffer((n,), dst)):
        with T.Kernel(1):
            a = T.alloc_shared((n,), src)
            c = T.alloc_shared((n,), dst)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    value = a[0] if uniform else a[i]
                    c[i] = T.Cast(dst, value, annotations=annotations)
            T.copy(c, C)

    return main


def _alias_kernel(src, dst, lanes, uniform):
    n = lanes

    @T.prim_func
    def main(A: T.Buffer((n,), src), C: T.Buffer((n,), dst)):
        with T.Kernel(1):
            a = T.alloc_shared((n,), src)
            c = T.alloc_shared((n,), dst)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    x = a[0] if uniform else a[i]
                    y = x
                    c[i] = T.cast(y, dst)
            T.copy(c, C)

    return main


def _external_scalar_kernel(vl):
    n = vl

    @T.prim_func
    def main(S: T.int8, C: T.Buffer((n,), "int32")):
        with T.Kernel(1):
            c = T.alloc_shared((n,), "int32")
            with T.SimdVF(lanes=vl):
                for i in T.Parallel(n):
                    c[i] = T.cast(S, "int32")
            T.copy(c, C)

    return main


def _binary_scalar_cast_kernel(vl):
    n = vl

    @T.prim_func
    def main(A: T.Buffer((n,), "int16"), B: T.Buffer((n,), "int8"), C: T.Buffer((n,), "int16")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "int16")
            b = T.alloc_shared((n,), "int8")
            c = T.alloc_shared((n,), "int16")
            T.copy(A, a)
            T.copy(B, b)
            with T.SimdVF(lanes=vl):
                for i in T.Parallel(n):
                    c[i] = a[i] + T.cast(b[0], "int16")
            T.copy(c, C)

    return main


def _coordinate_kernel(vl, which):
    m, n = 4, vl
    out = "int16"

    @T.prim_func
    def main(A: T.Buffer((m, n), "int32"), C: T.Buffer((m, n), out)):
        with T.Kernel(1):
            a = T.alloc_shared((m, n), "int32")
            c = T.alloc_shared((m, n), out)
            T.copy(A, a)
            with T.SimdVF(lanes=vl):
                for i, j in T.Parallel(m, n):
                    if which == "kept_of_load":
                        c[i, j] = T.cast(a[i, 0], out)
                    else:
                        c[i, j] = T.cast(a[i, j], out)
            T.copy(c, C)

    return main


def _nested_cast_kernel(vl):
    n = vl

    @T.prim_func
    def main(A: T.Buffer((n,), "int16"), C: T.Buffer((n,), "uint32")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "int16")
            c = T.alloc_shared((n,), "uint32")
            T.copy(A, a)
            with T.SimdVF(lanes=vl):
                for i in T.Parallel(n):
                    c[i] = T.cast(T.cast(a[i], "uint16"), "uint32")
            T.copy(c, C)

    return main


def _dead_chain_kernel(vl):
    n = vl

    @T.prim_func
    def main(A: T.Buffer((n,), "int8"), C: T.Buffer((n,), "int8")):
        with T.Kernel(1):
            a = T.alloc_shared((n,), "int8")
            c = T.alloc_shared((n,), "int8")
            T.copy(A, a)
            with T.SimdVF(lanes=vl):
                for i in T.Parallel(n):
                    # The assignment creates the Bind whose pure conversion
                    # chain must be eliminated by the local DCE.
                    unused = T.cast(T.cast(a[i], "uint16"), "uint32")  # noqa: F841
                    c[i] = a[i]
            T.copy(c, C)

    return main


def _fp8_kernel(src, dst, lanes, annotations=None, **kwargs):
    n = lanes

    @T.prim_func
    def main(A: T.Buffer((n,), src), C: T.Buffer((n,), dst)):
        with T.Kernel(1):
            a = T.alloc_shared((n,), src)
            c = T.alloc_shared((n,), dst)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    if annotations is not None:
                        c[i] = T.Cast(dst, a[i], annotations=annotations)
                    else:
                        c[i] = T.cast(a[i], dst, **kwargs)
            T.copy(c, C)

    return main


# ---------------------------------------------------------------------------
# A. 36 integer pairs at VL=128: action structure (lower-only, fast)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("src", INT_TYPES)
@pytest.mark.parametrize("dst", INT_TYPES)
def test_integer_pair_action_structure(src, dst):
    category = _pair_category(src, dst)
    source = _lower_source(_cast_kernel(src, dst, 128))
    lines = _cast_lines(source)
    assert _count(source, "vmi.vload") == 1, source
    if category == "identity":
        assert lines == [], lines
        return
    if category == "reinterpret":
        assert len(lines) == 1 and "vinterpret_cast" in lines[0], lines
        assert f"to_dtype={PTO_INT_NAME[dst]}" in lines[0], lines[0]
        return
    if category == "widen":
        assert len(lines) == 1 and "vmi.vcvt" in lines[0], lines
        assert f"to_dtype={PTO_INT_NAME[dst]}" in lines[0], lines[0]
        assert "saturate" not in lines[0], lines[0]
        return
    if category == "widen_and_reinterpret":
        intermediate = CROSS_SIGN_WIDEN_INTERMEDIATE[(src, dst)]
        assert len(lines) == 2, lines
        assert "vmi.vcvt" in lines[0], lines[0]
        assert f"to_dtype={PTO_INT_NAME[intermediate]}" in lines[0], lines[0]
        assert "vinterpret_cast" in lines[1], lines[1]
        assert f"to_dtype={PTO_INT_NAME[dst]}" in lines[1], lines[1]
        first_result = lines[0].split("=")[0].strip()
        assert lines[1].count(first_result) == 1, lines
        return
    # Narrowing: one direct logical vcvt with NOSAT.
    assert len(lines) == 1 and "vmi.vcvt" in lines[0], lines
    assert 'saturate="NOSAT"' in lines[0], lines[0]
    assert f"to_dtype={PTO_INT_NAME[dst]}" in lines[0], lines[0]


def _count_cast_nodes(module):
    func = next(iter(module.functions.values()))
    found = []

    def visit(node):
        if isinstance(node, tirx.Cast):
            found.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return found


@pytest.mark.parametrize("dtype", INT_TYPES)
def test_identity_cast_reaches_planner_and_emits_zero_actions(dtype):
    # A plain same-dtype T.cast is folded by the frontend, so the annotated
    # raw Cast is used to prove the planner identity path is reachable: the
    # manual sat=True marker must be validated and normalized away, and the
    # emission has zero conversion actions.
    func = _ann_kernel(dtype, dtype, 128, {"sat": True})
    module, error = _lower_capture(func)
    assert error is None, error
    casts = [node for node in _count_cast_nodes(module) if str(node.dtype) == dtype and str(node.value.dtype) == dtype]
    assert casts, "the annotated same-dtype Cast must reach Verify"
    source = _lower_source(func)
    assert _cast_lines(source) == []


# ---------------------------------------------------------------------------
# B. Producer once, ordered binds, scope and DCE
# ---------------------------------------------------------------------------
def test_cross_sign_widening_producer_and_order():
    source = _lower_source(_cast_kernel("int8", "uint32", 256))
    lines = _cast_lines(source)
    assert len(lines) == 2
    vload = _one(source, "vmi.vload")
    assert vload.split("=")[0].strip() in lines[0]
    assert source.index(lines[0]) < source.index(lines[1])
    assert _count(source, "vmi.vload") == 1


def test_conversion_binds_stay_between_load_and_store():
    source = _lower_source(_cast_kernel("int8", "uint32", 128))
    vload_index = source.index("vmi.vload")
    first_cast = source.index(_cast_lines(source)[0])
    store_index = source.index("vmi.vstore")
    assert vload_index < first_cast < store_index


def test_dead_pure_conversion_chain_is_dropped():
    source = _lower_source(_dead_chain_kernel(128))
    assert "vmi.vcvt" not in source, source
    assert "vinterpret_cast" not in source, source
    # The dead chain's vload is gone too; only the live a[i] load remains.
    assert _count(source, "vmi.vload") == 1, source


def test_nested_cast_chains_inner_then_outer():
    source = _lower_source(_nested_cast_kernel(128))
    lines = _cast_lines(source)
    assert len(lines) == 2, lines
    assert "vinterpret_cast" in lines[0] and "to_dtype=pto.ui16" in lines[0]
    assert "vmi.vcvt" in lines[1] and "to_dtype=pto.ui32" in lines[1]
    assert _count(source, "vmi.vload") == 1


# ---------------------------------------------------------------------------
# C. Verify/Vectorize classification consistency
# ---------------------------------------------------------------------------
def test_external_scalar_cast_stays_scalar():
    source = _lower_source(_external_scalar_kernel(128))
    assert _cast_lines(source) == []
    assert "pto.cast(" in source
    assert "vbrc" in source


def test_uniform_load_cast_stays_scalar():
    source = _lower_source(_ann_kernel("int8", "int32", 128, {}, uniform=True))
    assert _cast_lines(source) == []
    assert "pto.cast(" in source


def test_bind_alias_keeps_vmi_classification():
    source = _lower_source(_alias_kernel("int8", "int32", 128, uniform=False))
    assert len(_cast_lines(source)) == 1
    assert _count(source, "vmi.vload") == 1


def test_scalar_cast_then_parent_broadcast():
    source = _lower_source(_binary_scalar_cast_kernel(128))
    assert "vbrc" in source
    assert "pto.cast(" in source
    assert _cast_lines(source) == []


@pytest.mark.parametrize(
    ("which", "expect_vector"),
    [
        ("kept_of_load", False),
        ("selected_of_load", True),
    ],
)
def test_2d_coordinate_classification(which, expect_vector):
    source = _lower_source(_coordinate_kernel(128, which))
    lines = _cast_lines(source)
    if expect_vector:
        assert len(lines) == 1, lines
    else:
        assert lines == [], lines
        assert "pto.cast(" in source


def test_lane_one_vector_context_is_not_constructible():
    # {kVectorVmi, 1} would be a VMI value with one logical lane; the current
    # Parallel VL contract cannot produce it (lanes are 64/128/256), and the
    # planner rejects it explicitly.  The observable frontend boundary:
    with pytest.raises(ValueError, match="lanes must be one of"):
        T.SimdVF(lanes=1)


# ---------------------------------------------------------------------------
# D. Attribute table
# ---------------------------------------------------------------------------
def _lower_ann(src, dst, annotations):
    return _lower_source(_ann_kernel(src, dst, 128, annotations))


def test_integer_narrowing_accepts_sat_false():
    source = _lower_ann("int16", "int8", {"sat": False})
    line = _cast_lines(source)[0]
    assert 'saturate="NOSAT"' in line, line


def test_float_narrowing_accepts_sat_false_and_nosat():
    source = _lower_ann("float32", "float16", {"sat": False})
    line = _cast_lines(source)[0]
    assert 'rounding="R"' in line and 'saturate="NOSAT"' in line, line


def test_fp8_narrowing_accepts_round_and_sat_false():
    source = _lower_source(_fp8_kernel("float32", "float8_e5m2", 128, {"round": "Z", "sat": False}))
    line = _cast_lines(source)[0]
    assert 'rounding="Z"' in line and 'saturate="NOSAT"' in line, line


def test_integer_narrowing_rejects_sat_true():
    _expect_lower_error(
        _ann_kernel("int16", "int8", 128, {"sat": True}),
        "integer narrowing accepts only sat=False",
    )


def test_unknown_annotation_is_rejected():
    _expect_lower_error(
        _ann_kernel("int8", "int32", 128, {"mystery": 1}),
        "unknown cast annotation",
    )


def test_float_int_conversion_is_rejected():
    _expect_lower_error(
        _ann_kernel("float32", "int32", 128, {}),
        "float <-> integer conversions need per-combination",
    )


def test_f16_bf16_cross_family_is_rejected():
    _expect_lower_error(
        _ann_kernel("float16", "bfloat16", 128, {}),
        "cross-family f16 <-> bf16",
    )


def test_fp8_to_half_is_rejected():
    _expect_lower_error(
        _fp8_kernel("float8_e4m3fn", "float16", 128),
        "FP8 -> FP16/BF16 and FP8 <-> integer are out of scope",
    )


def test_lane_uniform_fp8_conversion_is_rejected():
    _expect_lower_error(
        _ann_kernel("float8_e4m3fn", "float32", 128, {}, uniform=True),
        "non-identity FP8 conversions have no scalar lowering",
    )


# ---------------------------------------------------------------------------
# E. Scalar paths and literal boundaries (existing semantics)
# ---------------------------------------------------------------------------
def test_scalar_float_widening_stays_scalar():
    source = _lower_source(_ann_kernel("float16", "float32", 128, {}, uniform=True))
    assert _cast_lines(source) == []
    assert "vbrc" in source


def test_scalar_float_narrowing_stays_scalar():
    source = _lower_source(_ann_kernel("float32", "float16", 128, {}, uniform=True))
    assert _cast_lines(source) == []
    assert "vbrc" in source


def test_scalar_integer_narrowing_keeps_plain_cast():
    source = _lower_source(_ann_kernel("int32", "int8", 128, {}, uniform=True))
    assert _cast_lines(source) == []
    assert "pto.cast(" in source


def test_actual_t_cast_literal_out_of_range_is_rejected_by_frontend():
    with pytest.raises(Exception, match="exceeds maximum of int8"):
        T.cast(300, "int8")


def test_actual_t_cast_negative_to_unsigned_is_rejected_by_frontend():
    with pytest.raises(Exception, match="cannot make uint from negative value"):
        T.cast(-1, "uint8")


# ---------------------------------------------------------------------------
# F. Direct planner contexts and scalar rebuilt span/annotations
# ---------------------------------------------------------------------------
def test_plan_value_cast_context_matrix_directly(tmp_path):
    """The two passes can only produce scalar lanes=1 and VMI lanes=64/128/256,
    so {kVectorVmi, 1} and {kScalar, 128} are unreachable end-to-end.  A local
    C++ probe calls PlanValueCast directly for both and for positive
    vector/scalar plans, without adding a product FFI."""
    binary = tmp_path / "cast_plan_probe"
    source = os.path.join(os.path.dirname(__file__), "cast_plan_context_probe.cc")
    command = [
        "c++",
        "-std=c++17",
        "-O0",
        source,
        "-I",
        f"{REPO_ROOT}/src",
        "-I",
        f"{REPO_ROOT}/3rdparty/tvm/include",
        "-I",
        f"{REPO_ROOT}/3rdparty/tvm/src",
        "-I",
        f"{REPO_ROOT}/3rdparty/tvm/3rdparty/dlpack/include",
        "-I",
        f"{REPO_ROOT}/3rdparty/tvm/3rdparty/tvm-ffi/include",
        "-I",
        f"{REPO_ROOT}/3rdparty/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include",
        "-L",
        f"{REPO_ROOT}/build/lib",
        "-ltilelang",
        "-ltvm_compiler",
        "-ltvm_ffi",
        "-ltvm_runtime",
        f"-Wl,-rpath,{REPO_ROOT}/build/lib",
        "-o",
        str(binary),
    ]
    build = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert build.returncode == 0, build.stdout + build.stderr
    env = dict(os.environ, LD_LIBRARY_PATH=f"{REPO_ROOT}/build/lib")
    run = subprocess.run([str(binary)], env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "FAIL" not in run.stdout, run.stdout
    assert "kVectorVmi lanes=1 is rejected" in run.stdout
    assert "kScalar lanes=128 is rejected" in run.stdout


@contextmanager
def _vectorize_capture():
    captured = {}
    original = ascend_transform.VectorizeParallelToPTO

    def factory():
        vectorize = original()

        def run(mod):
            out = vectorize(mod)
            captured["module"] = out
            return out

        return run

    ascend_transform.VectorizeParallelToPTO = factory
    try:
        yield captured
    finally:
        ascend_transform.VectorizeParallelToPTO = original


def _span_scalar_kernel(vl, span):
    @T.prim_func
    def main(S: T.Buffer((32,), "int32"), C: T.Buffer((vl,), "int8")):
        with T.Kernel(1):
            s = T.alloc_shared((32,), "int32")
            c = T.alloc_shared((vl,), "int8")
            T.copy(S, s)
            with T.SimdVF(lanes=vl):
                for i in T.Parallel(vl):
                    c[i] = T.cast(s[0], "int8", sat=False, span=span)
            T.copy(c, C)

    return main


def test_scalar_rebuild_preserves_span_and_drop_provenance():
    # Span(source_name, line, end_line, column, end_column).
    span = tvm.ir.Span(tvm.ir.SourceName("cast_span_probe"), 3, 3, 7, 11)
    with _vectorize_capture() as captured:
        _lower(_span_scalar_kernel(128, span), target=_target())
    module = captured["module"]
    func = next(iter(module.functions.values()))
    rebuilt = []

    def visit(node):
        if isinstance(node, tirx.Cast) and str(node.dtype) == "int8":
            rebuilt.append(node)

    tirx.stmt_functor.post_order_visit(func.body, visit)
    assert rebuilt, "the rebuilt scalar Cast is missing"
    matches = [
        node
        for node in rebuilt
        if node.span is not None and node.span.source_name is not None and node.span.source_name.name == "cast_span_probe"
    ]
    assert matches, "the rebuilt scalar Cast lost the original span"
    node = matches[0]
    assert node.span.line == 3 and node.span.column == 7
    assert node.span.end_line == 3 and node.span.end_column == 11
    # The validated sat=False marker is proven equivalent to the ordinary
    # truncating scalar lowering and is normalized away; no new attribute
    # format is introduced.
    assert "sat" not in node.annotations
    assert "round" not in node.annotations
