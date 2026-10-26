import numpy as np
import pytest

from tvm_tirx import runtime, tirx
from tilelang._compat import ir, sym
from tvm_tirx.ir import prim


@pytest.fixture(autouse=True)
def register_types():
    ir.register_custom_dtype("test_codegen_custom_float", 242, "float32")
    ir.register_custom_dtype("test_codegen_custom_byte", 243, "uint8")


@pytest.mark.parametrize("backend", ["c", "llvm"])
@pytest.mark.parametrize("code,bits,storage", [(242, 32, "float32"), (243, 8, "uint8")])
def test_custom_tensor_storage_copy_without_numeric_conversion(backend, code, bits, storage, compile_tilelang):
    dtype = ir.prim_type_from_dlpack((code, bits, 1))
    source = ir.Var("source", tirx.TensorType([8], dtype))
    output = ir.Var("output", tirx.TensorType([8], dtype))
    index = ir.Var("index", "int32")
    body = ir.For(
        index,
        0,
        8,
        1,
        ir.ForKind.DEFAULT,
        ir.TensorStore(output, [index], ir.TensorLoad(source, [index])),
    )
    function = tirx.Function([source, output], body).with_attr("global_symbol", "copy_custom")
    executable = compile_tilelang(function, backend)
    values = np.arange(8).astype(storage)
    result = runtime.empty((8,), storage)
    executable(runtime.tensor(values), result)
    np.testing.assert_array_equal(result.numpy(), values)


@pytest.mark.parametrize("backend", ["c", "llvm"])
@pytest.mark.parametrize("constant", [False, True])
def test_custom_conversion_requires_backend_semantics(backend, constant, compile_tilelang):
    custom = ir.PrimType("custom[test_codegen_custom_float]")
    source = ir.Var("source", tirx.TensorType([1], "float32"))
    output = ir.Var("output", tirx.TensorType([1], "float32"))
    value = ir.const(1.00075, "float32") if constant else ir.TensorLoad(source, [0])
    converted = ir.cast(ir.cast(value, custom), "float32")
    simplified = sym.Analyzer().simplify(converted)
    assert isinstance(simplified, prim.Cast)
    assert isinstance(simplified.value, prim.Cast)
    function = tirx.Function([source, output], ir.TensorStore(output, [0], simplified)).with_attr("global_symbol", "custom_cast")
    with pytest.raises(ValueError, match="Custom dtype casts require a backend numeric lowering"):
        compile_tilelang(function, backend)
