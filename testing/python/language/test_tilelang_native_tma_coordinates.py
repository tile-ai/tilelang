"""TMA coordinates are native expressions, including literal annotation operands."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

from tilelang.cuda import language as T
from tilelang.testing.ir import collect_nodes


@pytest.mark.parametrize("scatter", [False, True])
@pytest.mark.parametrize("column", [0, 1 << 31, "dynamic"])
def test_tma_coordinate_payload_preserves_literals_and_variable_identity(scatter, column):
    global_tensor = ir.Var("global_tensor", tirx.TensorType((64, 128), "float16"))
    shared_tensor = ir.Var("shared_tensor", tirx.TensorType((4, 64), "float16", scope="shared"))
    barrier = ir.Var("barrier", tirx.TensorType((1,), "uint64", scope="shared"))
    row = ir.Var("row", "int32")
    col = ir.Var("col", "int32") if column == "dynamic" else column
    rows = [0, row, ir.IntImm("int64", 2), 3]
    if scatter:
        call = T.tma_scatter4(shared_tensor, global_tensor, col, rows, annotations={"note": "caller"})
    else:
        call = T.tma_gather4(global_tensor, shared_tensor, col, rows, barrier=barrier, annotations={"note": "caller"})
    fields = tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[int(call.attrs["operand_count"])], call.attrs["annotation_schema"])
    assert fields["note"] == "caller"
    assert fields["gather4_rows"][1].same_as(row)
    assert fields["gather4_rows"][2].same_as(rows[2])
    for index in (0, 2, 3):
        value = fields["gather4_rows"][index]
        assert isinstance(value, ir.IntImm) and int(value) == int(rows[index])
    if column == "dynamic":
        assert fields["gather4_col"].same_as(col)
    else:
        value = fields["gather4_col"]
        assert isinstance(value, ir.IntImm) and int(value) == column
        assert value.ty == ir.PrimType("int64" if column >= 1 << 31 else "int32")

    params = [global_tensor, shared_tensor, barrier, row]
    if column == "dynamic":
        params.append(col)
    function = tirx.Function(params, [ir.Evaluate(call)])
    assert tirx.analysis.check_well_formed(function)
    assert not tirx.analysis.undefined_vars(function)
    recovered = ir.load_json(ir.save_json(function))
    ir.assert_structural_equal(function, recovered)
    (recovered_call,) = [node for node in collect_nodes(recovered, ir.Call) if node.op.name == "tl.tileop.copy"]
    recovered_fields = tvm_ffi.get_global_func("tl.DecodeAnnotations")(
        recovered_call.args[int(recovered_call.attrs["operand_count"])], recovered_call.attrs["annotation_schema"]
    )
    assert recovered_fields["gather4_rows"][1].same_as(recovered.params[3])
