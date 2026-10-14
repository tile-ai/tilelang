"""PTO Parallel mapping-contract tests.

Hardware independent: each case lowers TileLang IR for the PTO target with
:func:`tilelang.engine.lower.lower` (TileLang IR only, no PTOAS/Bisheng device
compilation and no device execution) and inspects the module exactly as
``VerifyParallelToPTO`` receives it: the module is captured at the Verify
boundary instead of the final lowered IR.

Covered contract:

* the logical ``T.Parallel`` extent ``E`` stays on the For and the accesses
  keep using the original loop variable (no For/Buffer rewrite);
* the fragment executing dimension is padded to ``P = ceil(E / L) * L`` and
  is written under the PTO-specific ``pto_parallel_loop_layout`` annotation
  (never the SIMT ``parallel_loop_layout`` key);
* ``Q = P / L`` and ``tail_lanes = E % L`` follow from ``E`` and ``L``;
* 2D units pad only the selected dimension and keep the other extent;
* a real (padded) row stride is accepted and buffer shapes are untouched;
* lanes ``{64, 128, 256}`` are accepted and agree with the annotation;
* the rejection diagnostics fire: gather access, invalid lanes, a missing or
  invalid lane annotation, and hand-written VMI mixed with ``T.Parallel``.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

import tilelang.ascend.language as T
from tilelang.ascend import transform as ascend_transform
from tilelang.ascend.target import normalize_pto_target
from tilelang.engine.lower import lower as _lower
from tilelang.layout.fragment import Fragment
import tvm

_PTO_LAYOUT_ATTR = "pto_parallel_loop_layout"
_SIMT_LAYOUT_ATTR = "parallel_loop_layout"
_LANES_ATTR = "tl.simdvf_lanes"


def _ceil_to_lanes(extent: int, lanes: int) -> int:
    return (extent + lanes - 1) // lanes * lanes


def _as_int(expr) -> int:
    if isinstance(expr, int):
        return expr
    value = getattr(expr, "value", None)
    assert value is not None, f"expected an integer constant, got {expr!r}"
    return int(value)


@contextmanager
def _verify_boundary_capture():
    """Capture the module handed to VerifyParallelToPTO, then delegate."""
    captured = {}
    original = ascend_transform.VerifyParallelToPTO

    def capturing_factory():
        verify_pass = original()

        def run(mod):
            captured["module"] = mod
            return verify_pass(mod)

        return run

    ascend_transform.VerifyParallelToPTO = capturing_factory
    try:
        yield captured
    finally:
        ascend_transform.VerifyParallelToPTO = original


def _lower_capture(func, *, expect_error=False):
    """Lower *func* for PTO and return (module_at_verify, error)."""
    target = normalize_pto_target("pto")
    error = None
    with _verify_boundary_capture() as captured:
        try:
            with target:
                _lower(func, target=target)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            error = exc
    if error is not None and not expect_error:
        raise error
    return captured.get("module"), error


def _body_of(mod):
    assert len(mod.functions) == 1, f"expected a single function, got {list(mod.functions)}"
    return next(iter(mod.functions.values())).body


def _pto_parallel_loops(mod):
    loops = []

    def visit(node):
        if isinstance(node, tvm.tirx.For) and _PTO_LAYOUT_ATTR in node.annotations:
            loops.append(node)

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    return loops


def _single_pto_parallel_loop(mod):
    loops = _pto_parallel_loops(mod)
    assert len(loops) == 1, f"expected exactly one pto_parallel_loop_layout unit, got {len(loops)}"
    return loops[0]


def _fragment_of(loop) -> Fragment:
    fragment = loop.annotations[_PTO_LAYOUT_ATTR]
    assert isinstance(fragment, Fragment), f"annotation is not a Fragment: {type(fragment)}"
    return fragment


def _simd_vf_block(mod):
    blocks = []

    def visit(node):
        if isinstance(node, tvm.tirx.SBlock) and _LANES_ATTR in node.annotations:
            blocks.append(node)

    tvm.tirx.stmt_functor.post_order_visit(_body_of(mod), visit)
    assert len(blocks) == 1, f"expected exactly one lane-annotated SIMD_VF block, got {len(blocks)}"
    return blocks[0]


def _input_shape(fragment):
    return [_as_int(dim) for dim in fragment.get_input_shape()]


def _first_store(body):
    stores = []

    def visit(node):
        if isinstance(node, tvm.tirx.BufferStore) and not stores:
            stores.append(node)

    tvm.tirx.stmt_functor.post_order_visit(body, visit)
    assert stores, "expected a BufferStore in the parallel unit"
    return stores[0]


def _rewrite_pto_layout_attr(mod, mutate):
    """Return *mod* with each pto_parallel_loop_layout annotation rewritten.

    ``mutate(value)`` returns the replacement value; returning ``None`` drops
    the annotation. Used to drive the annotation-level rejection diagnostics.
    """

    def postorder(node):
        if isinstance(node, tvm.tirx.For) and _PTO_LAYOUT_ATTR in node.annotations:
            annotations = dict(node.annotations)
            replacement = mutate(annotations[_PTO_LAYOUT_ATTR])
            if replacement is None:
                del annotations[_PTO_LAYOUT_ATTR]
            else:
                annotations[_PTO_LAYOUT_ATTR] = replacement
            return tvm.tirx.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                node.body,
                thread_binding=node.thread_binding,
                annotations=annotations,
            )
        return None

    func = next(iter(mod.functions.values()))
    new_body = tvm.tirx.stmt_functor.ir_transform(func.body, None, postorder, ["tirx.For"])
    new_mod = tvm.IRModule({func.attrs["global_symbol"]: func.with_body(new_body)})
    return new_mod


def _run_verify_directly(mod):
    target = normalize_pto_target("pto")
    with target:
        return ascend_transform.VerifyParallelToPTO()(mod)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def _k1d(E, out_extent, lanes, dtype="float32"):
    """1D scale-by-2 with a padded source and a padded-or-compact output UB."""

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


def _k2d_row(M, N, lanes, dtype="float32"):
    """2D row scale-by-2: j (inner, extent N) is selected and non-divisible;
    both UBs use the padded row stride P, a real stride != N."""

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


def _k2d_col(M, N, lanes, dtype="float32"):
    """2D column scale-by-2: i (extent M) is selected and non-divisible, j is
    kept; row-major (N, P_M) storage accessed c[j * P + i]."""

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


def _k_gather(E, lanes, dtype="float32"):
    """Gather access: a[2 * i] is neither continuous nor lane-uniform."""

    P = _ceil_to_lanes(E, lanes)

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        B: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(E // 2):
                    b[i] = a[i * 2] + a[i * 2]
            T.copy(b, B)

    return main


def _k_mixed_vmi(E, lanes, dtype="float32"):
    """Hand-written VMI next to a T.Parallel unit in the same SIMD_VF block."""

    P = _ceil_to_lanes(E, lanes)

    @T.prim_func
    def main(
        A: T.Tensor((P,), dtype),
        B: T.Tensor((P,), dtype),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((P,), dtype)
            b = T.alloc_shared((P,), dtype)
            T.copy(A, a)
            with T.SimdVF(lanes=lanes):
                mask = T.vmi.create_mask(lanes, size=lanes)
                value = T.vmi.vload(a[0], size=lanes)
                T.vmi.vstore(value, b[0], mask)
                for i in T.Parallel(E):
                    b[i] = a[i] + a[i]
            T.copy(b, B)

    return main


# ---------------------------------------------------------------------------
# Mapping contract
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_1d_nondivisible_keeps_logical_extent_and_pads_fragment():
    E, lanes = 150, 128
    P = _ceil_to_lanes(E, lanes)
    mod, _ = _lower_capture(_k1d(E, P, lanes))
    loop = _single_pto_parallel_loop(mod)

    assert _as_int(loop.extent) == E, "the For must keep the logical extent E"
    assert _SIMT_LAYOUT_ATTR not in loop.annotations, "the PTO lane layout must not use the SIMT key"

    fragment = _fragment_of(loop)
    assert _input_shape(fragment) == [P], "the selected dimension must be padded to P"
    assert _as_int(fragment.get_thread_size()) == lanes, "the fragment thread extent must be L"
    assert P // lanes == 2, "Q = P / L"
    assert E % lanes == 22, "tail_lanes = E % L"

    store = _first_store(loop.body)
    assert store.indices[0].same_as(loop.loop_var), "the store must keep the original loop variable and buffer"


@pytest.mark.pto
def test_divisible_extent_keeps_e_and_has_no_tail():
    E, lanes = 256, 128
    P = _ceil_to_lanes(E, lanes)
    mod, _ = _lower_capture(_k1d(E, P, lanes))
    loop = _single_pto_parallel_loop(mod)

    assert _as_int(loop.extent) == E
    fragment = _fragment_of(loop)
    assert _input_shape(fragment) == [P]
    assert P == E and E % lanes == 0, "divisible extents keep P == E and tail_lanes == 0"


@pytest.mark.pto
@pytest.mark.parametrize("lanes", [64, 128, 256])
def test_lane_matrix_agrees_with_the_annotation(lanes):
    E = 150
    P = _ceil_to_lanes(E, lanes)
    mod, _ = _lower_capture(_k1d(E, P, lanes))
    loop = _single_pto_parallel_loop(mod)

    assert _as_int(loop.extent) == E
    block = _simd_vf_block(mod)
    assert _as_int(block.annotations[_LANES_ATTR]) == lanes, "the lanes annotation must match the request"
    fragment = _fragment_of(loop)
    assert _input_shape(fragment) == [P]
    assert _as_int(fragment.get_thread_size()) == lanes


@pytest.mark.pto
def test_2d_selected_dimension_padded_and_kept_dimension_unchanged():
    M, N, lanes = 4, 150, 128
    P = _ceil_to_lanes(N, lanes)
    mod, _ = _lower_capture(_k2d_row(M, N, lanes))
    loop = _single_pto_parallel_loop(mod)

    inner = loop.body
    assert isinstance(inner, tvm.tirx.For), "expected the two-layer Parallel nest"
    assert _as_int(loop.extent) == M, "the outer loop keeps its extent"
    assert _as_int(inner.extent) == N, "the selected inner loop keeps its logical extent"

    fragment = _fragment_of(loop)
    assert _input_shape(fragment) == [M, P], "only the selected dimension is padded"
    assert _as_int(fragment.get_thread_size()) == lanes

    store = _first_store(inner.body)
    expected_index = loop.loop_var * P + inner.loop_var
    assert tvm.ir.structural_equal(store.indices[0], expected_index), "the store keeps the row-major real stride index i * P + j"


@pytest.mark.pto
def test_2d_column_selection_pads_the_selected_dimension_only():
    M, N, lanes = 150, 4, 128
    P = _ceil_to_lanes(M, lanes)
    mod, _ = _lower_capture(_k2d_col(M, N, lanes))
    loop = _single_pto_parallel_loop(mod)

    inner = loop.body
    assert _as_int(loop.extent) == M
    assert _as_int(inner.extent) == N

    fragment = _fragment_of(loop)
    assert _input_shape(fragment) == [P, N], "the selected i dimension is padded, j keeps its extent"
    assert _as_int(fragment.get_thread_size()) == lanes


# ---------------------------------------------------------------------------
# Rejection diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.pto
def test_gather_access_is_rejected():
    E, lanes = 150, 128
    _, error = _lower_capture(_k_gather(E, lanes), expect_error=True)
    assert error is not None, "a gather access must be rejected"
    message = str(error)
    assert "continuous" in message or "pto_parallel_loop_layout" in message, f"expected a continuity/layout diagnostic, got: {message}"


@pytest.mark.pto
def test_invalid_lanes_is_rejected_by_the_frontend():
    with pytest.raises(ValueError, match="lanes must be one of"):
        T.SimdVF(lanes=96)


@pytest.mark.pto
def test_missing_pto_lane_layout_is_rejected():
    E, lanes = 150, 128
    P = _ceil_to_lanes(E, lanes)
    mod, _ = _lower_capture(_k1d(E, P, lanes))
    stripped_mod = _rewrite_pto_layout_attr(mod, lambda value: None)
    with pytest.raises(Exception, match="pto_parallel_loop_layout is missing"):
        _run_verify_directly(stripped_mod)


@pytest.mark.pto
def test_invalid_lane_annotation_is_rejected_by_verify():
    E, lanes = 150, 128
    P = _ceil_to_lanes(E, lanes)
    mod, _ = _lower_capture(_k1d(E, P, lanes))
    assert _as_int(_simd_vf_block(mod).annotations[_LANES_ATTR]) == lanes

    func = next(iter(mod.functions.values()))

    def postorder(node):
        if isinstance(node, tvm.tirx.SBlock) and _LANES_ATTR in node.annotations:
            annotations = dict(node.annotations)
            annotations[_LANES_ATTR] = tvm.tirx.IntImm("int64", 96)
            return tvm.tirx.SBlock(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                node.body,
                init=node.init,
                alloc_buffers=node.alloc_buffers,
                match_buffers=node.match_buffers,
                annotations=annotations,
            )
        return None

    body = tvm.tirx.stmt_functor.ir_transform(func.body, None, postorder, ["tirx.SBlock"])
    broken_mod = tvm.IRModule({func.attrs["global_symbol"]: func.with_body(body)})
    with pytest.raises(Exception, match="lanes must be one of"):
        _run_verify_directly(broken_mod)


@pytest.mark.pto
def test_handwritten_vmi_mixed_with_parallel_is_rejected():
    E, lanes = 150, 128
    _, error = _lower_capture(_k_mixed_vmi(E, lanes), expect_error=True)
    assert error is not None, "mixing hand-written VMI with T.Parallel must be rejected"
    message = str(error)
    assert "mixing" in message or "cannot be mixed" in message, f"expected the hand-written VMI mixing diagnostic, got: {message}"


# ---------------------------------------------------------------------------
# Op-level dtype matrix for sub/div/unary
# ---------------------------------------------------------------------------


def _binary_kernel(n, lanes, dtype, op):
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), dtype)
            b_ub = T.alloc_shared((n,), dtype)
            c_ub = T.alloc_shared((n,), dtype)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF(lanes=lanes):
                for i in T.Parallel(n):
                    if op == "sub":
                        c_ub[i] = a_ub[i] - b_ub[i]
                    else:
                        c_ub[i] = a_ub[i] / b_ub[i]
            T.copy(c_ub, C)

    return main


@pytest.mark.pto
def test_verify_bf16_div_rejected():
    with pytest.raises(Exception, match=r"vdiv on .*bfloat16.*operation matrix"):
        _lower(_binary_kernel(128, 128, "bfloat16", "div"), target="pto")


@pytest.mark.pto
def test_verify_uint16_div_rejected():
    with pytest.raises(Exception, match=r"vdiv on uint16.*operation matrix"):
        _lower(_binary_kernel(128, 128, "uint16", "div"), target="pto")


@pytest.mark.pto
def test_verify_fp8_sub_rejected():
    with pytest.raises(Exception, match=r"FP8 arithmetic is not supported"):
        _lower(_binary_kernel(128, 128, "float8_e4m3fn", "sub"), target="pto")


@pytest.mark.pto
def test_verify_fp8_div_rejected():
    with pytest.raises(Exception, match=r"FP8 arithmetic is not supported"):
        _lower(_binary_kernel(128, 128, "float8_e5m2", "div"), target="pto")
