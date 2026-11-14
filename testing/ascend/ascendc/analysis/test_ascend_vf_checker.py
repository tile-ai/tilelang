"""VF legality is checked on source TIR, before scheduling or device lowering."""

import pytest
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend.analysis import VFChecker
from tvm import tirx
from testing.ascend._ir import copy, kernel, seq


def _check(before, error=None):
    if error:
        with pytest.raises(ValueError, match=error):
            VFChecker()(before)
    else:
        tvm.ir.assert_structural_equal(VFChecker()(before), before)


@pytest.mark.parametrize(
    "vf, cast, error", [(None, False, None), (None, True, "DMA copies cannot perform type casting"), ("SIMT_VF", True, None)]
)
def test_copy_dtype_contract(vf, cast, error):
    a = tirx.decl_buffer((64,), "float16", name="A")
    ub = tirx.decl_buffer((64,), "float32" if cast else "float16", name="ub", scope="shared.dyn")
    body = copy(a, ub)
    if vf:
        body = tirx.SBlock([], [], [], vf, body)
    _check(kernel(body, buffers=[ub], params=[a]), error)


@pytest.mark.parametrize("vf", [None, "SIMT_VF", "SIMD_VF"])
def test_parallel_loop_requires_a_vf(vf):
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    body = tirx.For(i, 0, 64, tirx.ForKind.PARALLEL, tirx.BufferStore(ub, tirx.const(1, "float32"), [i]))
    if vf:
        body = tirx.SBlock([], [], [], vf, body)
    _check(kernel(body, buffers=[ub]), None if vf else "Parallel loops outside VF blocks")


@pytest.mark.parametrize("vf", ["SIMT_VF", "SIMD_VF"])
@pytest.mark.parametrize("access", ["load", "store", "copy"])
def test_global_memory_access_depends_on_vf_kind(vf, access):
    a = tirx.decl_buffer((64,), "float32", name="A")
    ub = tirx.decl_buffer((64,), "float32", name="ub", scope="shared.dyn")
    if access == "copy":
        body = copy(a, ub)
    elif access == "load":
        body = tirx.BufferStore(ub, a[0], [0])
    else:
        body = tirx.BufferStore(a, ub[0], [0])
    _check(
        kernel(tirx.SBlock([], [], [], vf, body), buffers=[ub], params=[a]),
        "SIMD_VF blocks cannot access global memory" if vf == "SIMD_VF" else None,
    )


@pytest.mark.parametrize("vf", ["SIMT_VF", "SIMD_VF"])
@pytest.mark.parametrize("outside", [False, True])
def test_local_scalar_writes_cannot_require_pointer_capture(vf, outside):
    value = tirx.decl_buffer((1,), "int32", name="value", scope="local.var")
    body = tirx.SBlock([], [], [], vf, tirx.Evaluate(T.fill(value, 1)), alloc_buffers=[] if outside else [value])
    _check(kernel(body, buffers=[value] if outside else []), "pointer-type capture" if outside else None)


@pytest.mark.parametrize("vf, scope", [("SIMD_VF", "shared.dyn"), ("SIMT_VF", "shared.dyn"), ("SIMT_VF", "local")])
@pytest.mark.parametrize("outside", [False, True])
def test_vf_allocation_lifetime(vf, scope, outside):
    # Construct IR directly so this exercises VFChecker's scope rule rather
    # than the parser's generic immutable-variable visibility check.
    temp = tirx.decl_buffer((1,), "int32", name="temp", scope=scope)
    out = tirx.decl_buffer((1,), "int32", name="out")
    block = tirx.SBlock([], [], [], vf, tirx.BufferStore(temp, 1, [0]), alloc_buffers=[] if outside else [temp])
    before = kernel(seq(block, tirx.BufferStore(out, temp[0], [0])), buffers=[temp] if outside else [], params=[out])
    _check(before, None if outside else "accessed outside its allocation scope")
