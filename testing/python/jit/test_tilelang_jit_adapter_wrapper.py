import pytest
import torch

from tilelang import tvm
from tilelang.jit.adapter.shape import OutputShapeResolver
from tilelang.jit.adapter.wrapper import TLCPUSourceWrapper


def test_adapter_source_wrapper_requires_lowered_modules():
    with pytest.raises(ValueError, match="missing: device_mod, host_mod"):
        TLCPUSourceWrapper(
            scheduled_ir_module=tvm.IRModule(),
            source="",
            target=tvm.target.Target("c"),
        )


def test_output_shape_expressions():
    tir = tvm.tirx
    n, stride, k = (tir.Var(name, "int32") for name in ("n", "stride", "k"))
    source = tir.decl_buffer((n,), "float32", name="A", strides=(stride,))
    output = tir.decl_buffer((n * 2, (n + k + 3) // 4, stride + 1), "float32", name="C")
    program = tir.PrimFunc([output.data, source.data, k], tir.Evaluate(0), buffer_map={output.data: output, source.data: source})
    for func in (program, tvm.ir.load_json(tvm.ir.save_json(program))):
        resolver = OutputShapeResolver(func, [0])
        assert resolver(0, [None, torch.empty_strided((7,), (5,)), 3]) == [14, 3, 6]
        with pytest.raises(ValueError, match="output C dimension 0.*no live input"):
            resolver(0, [None, None, 3])


@pytest.mark.parametrize("case", ["negative", "unresolved", "zero_divisor"])
def test_output_shape_reports_invalid_expression(case):
    tir = tvm.tirx
    n = tir.Var("n", "int32")
    expressions = {
        "negative": (n - 8, "negative extent -1"),
        "unresolved": (tir.call_extern("int32", "unresolved_dim", n), "did not resolve to an integer"),
        "zero_divisor": (n // (n - 7), "Divide by zero"),
    }
    expr, reason = expressions[case]
    output = tir.decl_buffer((expr,), "float32", name="C")
    func = tir.PrimFunc([n, output.data], tir.Evaluate(0), buffer_map={output.data: output})
    with pytest.raises(ValueError, match="output C dimension 0") as error:
        OutputShapeResolver(func, [1])(1, [7, None])
    assert reason in str(error.value)


@pytest.mark.parametrize("value", [None, 1.5, torch.tensor(7)])
def test_output_shape_rejects_noninteger_scalar(value):
    tir = tvm.tirx
    n = tir.Var("n", "int32")
    output = tir.decl_buffer((n * 2,), "float32", name="C")
    func = tir.PrimFunc([n, output.data], tir.Evaluate(0), buffer_map={output.data: output})
    with pytest.raises(ValueError, match="output C dimension 0"):
        OutputShapeResolver(func, [1])(1, [value, None])


if __name__ == "__main__":
    pytest.main([__file__])
