"""VF legality is checked on source TIRx, before scheduling or device lowering."""

import pytest
from tvm_tirx import ir
import tilelang.ascend.language as T
from tilelang.ascend.analysis import VFChecker
from tilelang._compat.tensor import TensorView
from testing.ascend._ir import copy, kernel, seq


def _tensor(name, shape=(64,), dtype="float32", scope="global"):
    return TensorView.create(name, shape, dtype, scope=scope).tensor


def _vf(name, body, buffers=()):
    allocations = [ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty])) for tensor in buffers]
    return ir.RegionStmt("tl.scope", [], [], {"name_hint": name}, [*allocations, body])


def _check(before, error=None):
    if error:
        with pytest.raises(ValueError, match=error):
            VFChecker()(before)
    else:
        ir.assert_structural_equal(VFChecker()(before), before)


@pytest.mark.parametrize(
    "vf, cast, error", [(None, False, None), (None, True, "DMA copies cannot perform type casting"), ("SIMT_VF", True, None)]
)
def test_copy_dtype_contract(vf, cast, error):
    source = _tensor("A", dtype="float16")
    shared = _tensor("ub", dtype="float32" if cast else "float16", scope="shared.dyn")
    body = copy(source, shared)
    _check(kernel(_vf(vf, body) if vf else body, buffers=[shared], params=[source]), error)


@pytest.mark.parametrize("vf", [None, "SIMT_VF", "SIMD_VF"])
def test_parallel_loop_requires_a_vf(vf):
    shared = _tensor("ub", scope="shared.dyn")
    index = ir.Var("index", "int32")
    body = ir.For(index, 0, 64, 1, ir.ForKind.PARALLEL, ir.TensorStore(shared, [index], ir.const(1, "float32")))
    _check(kernel(_vf(vf, body) if vf else body, buffers=[shared]), None if vf else "Parallel loops outside VF blocks")


@pytest.mark.parametrize("vf", ["SIMT_VF", "SIMD_VF"])
@pytest.mark.parametrize("access", ["load", "store", "copy"])
def test_global_memory_access_depends_on_vf_kind(vf, access):
    source = _tensor("A")
    shared = _tensor("ub", scope="shared.dyn")
    if access == "copy":
        body = copy(source, shared)
    elif access == "load":
        body = ir.TensorStore(shared, [0], source[0])
    else:
        body = ir.TensorStore(source, [0], shared[0])
    _check(
        kernel(_vf(vf, body), buffers=[shared], params=[source]), "SIMD_VF blocks cannot access global memory" if vf == "SIMD_VF" else None
    )


@pytest.mark.parametrize("vf", ["SIMT_VF", "SIMD_VF"])
@pytest.mark.parametrize("outside", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_local_scalar_writes_cannot_require_pointer_capture(vf, outside, alias):
    value = _tensor("value", (1,), "int32", "local.var")
    view = _tensor("view", (1,), "int32", "local.var")
    body = ir.Evaluate(T.fill(view if alias else value, 1))
    if alias:
        body = seq(ir.Bind(view, ir.Call("tirx.decl_tensor", [value.data], ty_args=[view.ty])), body)
    _check(
        kernel(_vf(vf, body, [] if outside else [value]), buffers=[value] if outside else []), "pointer-type capture" if outside else None
    )


@pytest.mark.parametrize("vf, scope", [("SIMD_VF", "shared.dyn"), ("SIMT_VF", "shared.dyn"), ("SIMT_VF", "local")])
@pytest.mark.parametrize("outside", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_vf_allocation_lifetime(vf, scope, outside, alias):
    temporary = _tensor("temp", (1,), "int32", scope)
    view = _tensor("view", (1,), "int32", scope)
    output = _tensor("out", (1,), "int32")
    body = ir.TensorStore(temporary, [0], 1)
    if alias:
        body = seq(body, ir.Bind(view, ir.Call("tirx.decl_tensor", [temporary.data], ty_args=[view.ty])))
    block = _vf(vf, body, [] if outside else [temporary])
    before = kernel(
        seq(block, ir.TensorStore(output, [0], (view if alias else temporary)[0])), buffers=[temporary] if outside else [], params=[output]
    )
    _check(before, None if outside else "accessed outside its allocation scope")
