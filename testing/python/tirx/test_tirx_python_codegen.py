import ast
from types import SimpleNamespace

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def _source(function, backend="python"):
    function = function.with_attr("global_symbol", "kernel")
    build = tvm_ffi.get_global_func(f"target.build.tilelang_{backend}_without_compile")
    module = build(ir.IRModule({"kernel": function}), Target("cuda" if backend == "cutedsl" else "c"))
    source = module.inspect_source("py")
    ast.parse(source)
    return source


def _run(source, *arguments):
    namespace = {
        "cute": SimpleNamespace(kernel=lambda function: function),
        "cutlass": SimpleNamespace(range=range, range_constexpr=range, Int32=int),
        "tl": SimpleNamespace(ThreadIdx=lambda: None, BlockIdx=lambda: None),
    }
    exec(source, namespace)
    return namespace["kernel"](*arguments)


@pytest.mark.parametrize("step", [1, 2, -1, -3])
def test_python_codegen_native_tensor_loops(step):
    source_tensor = TensorView.create("source", [20], "int32")
    output = TensorView.create("output", [20], "int32")
    index = ir.Var("index", "int32")
    begin, end = (2, 18) if step > 0 else (18, 2)
    loop = ir.For(
        index,
        begin,
        end,
        step,
        ir.ForKind.DEFAULT,
        ir.SeqStmt([output.store([index], source_tensor.load([index]) + 7)]),
    )
    function = tirx.Function([source_tensor.tensor, output.tensor], [loop])
    values = np.arange(20, dtype="int32")
    actual = np.full_like(values, -1)
    expected = actual.copy()
    expected[begin:end:step] = values[begin:end:step] + 7
    _run(_source(function), values, actual)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("backend", ["python", "cutedsl"])
def test_codegen_native_scalar_let_and_tuple(backend):
    value = ir.Var("value", "int32")
    temporary = ir.Var("temporary", "int32")
    result = ir.prim.Let(temporary, value + 3, temporary * 2)
    function = tirx.Function([value], [ir.Return(result)], ret_type=ir.PrimType("int32"))
    if backend == "cutedsl":
        with pytest.raises(ValueError, match="cannot return a value"):
            _source(function, backend)
    else:
        assert _run(_source(function), 9) == 24


@pytest.mark.parametrize("kind", [ir.ForKind.DEFAULT, ir.ForKind.UNROLLED])
@pytest.mark.parametrize("step", [2, -2])
def test_cutedsl_preserves_loop_step(kind, step):
    index = ir.Var("index", "int32")
    loop = ir.For(index, 8 if step < 0 else 0, 0 if step < 0 else 8, step, kind, ir.SeqStmt([]))
    source = _source(tirx.Function([], [loop]), "cutedsl")
    assert f", {step})" in source
    _run(source)


def test_cutedsl_native_break_uses_loop_guard():
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 10, 2, ir.ForKind.DEFAULT, ir.SeqStmt([ir.Break()]))
    source = _source(tirx.Function([], [loop]), "cutedsl")
    assert "_loop_break_0 = 1" in source
    assert "break" not in {node.__class__.__name__.lower() for node in ast.walk(ast.parse(source))}
    _run(source)


def test_cutedsl_preserves_postprocessor():
    name = "tilelang_callback_cutedsl_postproc"
    previous = tvm_ffi.get_global_func(name, allow_missing=True)
    tvm_ffi.register_global_func(name, lambda source, target: source + "\npostprocessed = True\n", override=True)
    try:
        assert "postprocessed = True" in _source(tirx.Function([], []), "cutedsl")
    finally:
        if previous is None:
            tvm_ffi.remove_global_func(name)
        else:
            tvm_ffi.register_global_func(name, previous, override=True)


def test_cutedsl_vector_expression_preserves_ssa_indentation():
    source = TensorView.create("source", [2], "float32")
    output = TensorView.create("output", [2], "float32")
    indices = [ir.prim.Ramp(0, 1, 2)]
    value = source.load(indices) + ir.prim.Broadcast(ir.const(1.0, "float32"), 2)
    function = tirx.Function([source.tensor, output.tensor], [output.store(indices, value)])
    generated = _source(function, "cutedsl")
    assert "make_filled_tensor" in generated


@pytest.mark.parametrize(
    "expression,expected",
    [
        (lambda value: value << 2, 52),
        (lambda value: value >> 2, 3),
        (lambda value: value & 7, 5),
        (lambda value: value | 16, 29),
        (lambda value: value ^ 7, 10),
        (lambda value: ~value, -14),
    ],
)
def test_python_codegen_native_bitwise_nodes(expression, expected):
    value = ir.Var("value", "int32")
    function = tirx.Function([value], [ir.Return(expression(value))], ret_type=ir.PrimType("int32"))
    assert _run(_source(function), 13) == expected
