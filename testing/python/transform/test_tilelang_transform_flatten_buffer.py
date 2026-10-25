import pytest

import tilelang
import tilelang.testing
from tilelang import tvm
import tilelang.language as T


def _collect_address_indices(function):
    indices = []

    def visit(node):
        if isinstance(node, tvm.ir.Call) and node.op.name == "tirx.address_of":
            indices.append(node.args[0].indices[0])

    tvm.tirx.stmt_functor.post_order_visit(function.body, visit)
    return indices


@pytest.mark.parametrize("extent,stride,expected_type", [(1 << 30, 4, "int64"), (1 << 20, 2, "int32")])
def test_flattened_address_indices_use_overflow_safe_arithmetic(extent, stride, expected_type):
    @T.prim_func
    def before(source: T.Tensor((1,), T.float16)):
        for index in T.serial(extent):
            T.evaluate(T.address_of(source[index * stride]))

    module = tvm.IRModule({"main": before})
    original = _collect_address_indices(module["main"])
    assert len(original) == 1
    assert original[0].ty == "int32"
    with tvm.target.Target("c"):
        module = tilelang.transform.FlattenBuffer()(module)
    indices = _collect_address_indices(module["main"])
    assert len(indices) == 1
    assert indices[0].ty == expected_type
    assert tvm.tirx.analysis.check_well_formed(module)
    tvm.ir.assert_structural_equal(module, tvm.ir.load_json(tvm.ir.save_json(module)))
    variables = tvm.tirx.analysis.undefined_vars(indices[0])
    value = tvm.tirx.stmt_functor.substitute(indices[0], {variables[0]: tvm.ir.const(extent - 1)})
    assert int(tvm.sym.Analyzer().simplify(value)) == (extent - 1) * stride


if __name__ == "__main__":
    tilelang.testing.main()
