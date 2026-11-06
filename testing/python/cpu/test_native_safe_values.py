"""The public safe-value annotation keeps numeric values and lexical overrides."""

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tilelang import language as T
from tvm_tirx import ir, tirx

from tilelang.testing.packed import compile_packed_function


@pytest.mark.parametrize("outer,inner", [(-2, -5), (-2.0, -5.0)])
def test_native_safe_value_numbers_merge_and_restore_after_nested_scope(outer, inner):
    @T.prim_func
    def copy_with_safe_values(A: T.Tensor((3,), "float32"), B: T.Tensor((18,), "float32")):
        with T.sblock("root"):
            T.annotate_safe_value({A: outer})
            T.copy(A[0:6], B[0:6])
            with T.sblock("nested"):
                T.annotate_safe_value({A: inner})
                T.copy(A[0:6], B[6:12])
            T.copy(A[0:6], B[12:18])

    target = tvm_tirx.target.Target("c")
    function = copy_with_safe_values.with_attr("target", target)
    module = ir.IRModule({"main": function})
    with target:
        module = tvm_ffi.get_global_func("tl.transform.LayoutInference")()(module)
        module = tvm_ffi.get_global_func("tl.transform.LowerTileOp")()(module)
    lowered = module["main"]
    assert tirx.analysis.check_well_formed(lowered)
    assert not tirx.analysis.undefined_vars(lowered)
    fallback_values = []

    def visit(call):
        if isinstance(call.op, ir.Op) and call.op.name == "prim.if_then_else":
            value = call.args[2]
            while isinstance(value, ir.prim.Broadcast):
                value = value.value
            assert isinstance(value, ir.FloatImm)
            fallback_values.append(value.value)

    tvm_ffi.structural_walk(lowered.body, (ir.Call, visit), order="post")
    assert fallback_values == [float(outer), float(inner), float(outer)]

    module = tvm_ffi.get_global_func("tl.transform.LegalizeSafeMemoryAccess")()(module)
    module = tvm_ffi.get_global_func("tl.transform.LowerSourceAnnotations")()(module)
    module = ir.IRModule({"main": module["main"].with_attr("global_symbol", "main")})
    executable = compile_packed_function(module, c_codegen="compute")
    source = tvm_tirx.runtime.tensor(np.arange(3, dtype="float32"))
    output = tvm_tirx.runtime.tensor(np.full(18, 999, dtype="float32"))
    executable(source, output)
    expected = np.concatenate([np.pad(np.arange(3), (0, 3), constant_values=value) for value in (outer, inner, outer)])
    np.testing.assert_array_equal(output.numpy(), expected)
