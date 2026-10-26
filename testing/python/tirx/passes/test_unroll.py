from __future__ import annotations

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang
from tilelang.transform import _tirx_passes

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


@pytest.mark.parametrize("begin,end,step", [(0, 6, 2), (0, 5, 2), (3, 14, 4), (8, 3, 2)])
def test_tilelang_unroll_preserves_non_unit_step(begin, end, step):
    index = ir.Var("index", "int32")
    output = ir.Var("output", tirx.TensorType([16], "int32"))
    loop = ir.For(index, begin, end, step, ir.ForKind.UNROLLED, ir.TensorStore(output, [index], index))
    module = ir.IRModule({"main": tirx.Function([output], loop)})
    with ir.transform.PassContext(config={"tl.UnrollLoop": {"explicit_unroll": True}}):
        module = tilelang.transform.UnrollLoop()(module)
    module = tilelang.transform.LetInline()(module)
    module = _tirx_passes.StmtSimplify()(module)
    stores = []
    tvm_ffi.structural_walk(module, (ir.TensorStore, lambda node: stores.append(node)))
    assert [int(store.indices[0]) for store in stores] == list(range(begin, end, step))
    assert [int(store.value) for store in stores] == list(range(begin, end, step))
    assert tirx.analysis.check_well_formed(module)
