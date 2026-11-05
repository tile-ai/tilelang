"""The public WMMA intrinsic must keep its fragment write through cleanup."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.tirx import op

import tilelang
import tilelang.language as T


@pytest.mark.parametrize("cleanup", ["native", "tilelang", "combined"])
def test_public_wmma_load_preserves_fragment_write(cleanup):
    source = ir.Var("source", tirx.TensorType([256], "float16", scope="global"))
    fragment = ir.Var("fragment", tirx.TensorType([256], "float16", scope="wmma.matrix_a"))
    output = ir.Var("output", tirx.TensorType([1], "float16", scope="global"))
    writer = T.tvm_load_matrix_sync(fragment, 16, 16, 16, 0, op.tensor_data_ptr(source), 16, "row_major")
    assert writer.op.same_as(ir.Op.get("tl.wmma_load_matrix_sync"))
    writer.validate()
    before = tirx.Function(
        [source, output],
        [
            ir.Bind(fragment, ir.Call("tirx.alloc_tensor", [], ty_args=[fragment.ty])),
            ir.Evaluate(writer),
            ir.TensorStore(output, [0], ir.TensorLoad(fragment, [0])),
        ],
        ir.NoneType(),
    )
    assert tirx.analysis.check_well_formed(before)
    assert tvm_ffi.get_global_func("tl.SideEffect")(writer) == ir.CallEffectKind.Opaque.value
    mod = ir.IRModule({"main": before})
    if cleanup in ("tilelang", "combined"):
        mod = tilelang.transform.Simplify()(mod)
        mod = tilelang.transform.RemoveNoOp()(mod)
    if cleanup in ("native", "combined"):
        mod = tirx.transform.RemoveNoOp()(mod)
    after = mod["main"]
    assert tirx.analysis.check_well_formed(after)
    calls, stores = [], []
    tvm_ffi.structural_walk(after, (ir.Call, calls.append), order="pre")
    tvm_ffi.structural_walk(after, (ir.TensorStore, stores.append), order="pre")
    assert sum(call.op.same_as(writer.op) for call in calls) == 1
    assert len(stores) == 1
    assert stores[0].dest.same_as(output)


def test_public_wmma_load_validates_integer_shape():
    source = ir.Var("source", tirx.TensorType([256], "float16", scope="global"))
    fragment = ir.Var("fragment", tirx.TensorType([256], "float16", scope="wmma.matrix_a"))
    writer = T.tvm_load_matrix_sync(fragment, ir.const(16.0, "float32"), 16, 16, 0, op.tensor_data_ptr(source), 16, "row_major")
    with pytest.raises(TypeError, match="scalar integers"):
        writer.validate()
