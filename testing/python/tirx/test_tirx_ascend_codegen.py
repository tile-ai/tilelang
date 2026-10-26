import ast

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_operator_helpers", "ascend_codegen")


def _function(parameters, body):
    function = tirx.Function(parameters, body, ret_type=ir.TupleType([]))
    assert tirx.analysis.check_well_formed(function)
    return function.with_attr("global_symbol", "main").with_attr("calling_conv", 2)


def _source(function, backend="pto"):
    target = Target({"kind": "ascend", "keys": ["pto"] if backend == "pto" else ["ascend"]})
    function = function.with_attr("target", target)
    build = tvm_ffi.get_global_func(f"target.build.tilelang_{backend}_without_compile")
    source = build(ir.IRModule({"main": function}), target).inspect_source()
    if backend == "pto":
        ast.parse(source)
    return source


def _scope(name, body, **attributes):
    return ir.RegionStmt("tl.scope", [], [], {"name_hint": name, **attributes}, body)


def _set_direction(direction):
    return ir.Evaluate(ir.Call("tl.ascend_set_mmad_direction", [direction]))


def _mad_program(body_builder):
    flag = ir.Var("flag", "int32")
    left = TensorView.create("left", [256], "bfloat16", scope="shared.l0a")
    right = TensorView.create("right", [256], "bfloat16", scope="shared.l0b")
    output = TensorView.create("output", [256], "float32", scope="shared.l0c")

    def mad():
        arguments = [tirx.address_of(tensor.load(0)) for tensor in (output, left, right)]
        return ir.Evaluate(ir.Call("tl.ascend_mad", [*arguments, 16, 16, 16, 0, 1, 0, True]))

    return _function([flag], [tensor.allocate() for tensor in (left, right, output)] + body_builder(flag, mad))


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_native_kernel_signature_and_global_store(backend):
    output = TensorView.create("output", [1], "float32")
    scale = ir.Var("scale", "float32")
    source = _source(_function([output.tensor, scale], [output.store(0, scale)]), backend)
    assert "output" in source
    assert "scale" in source
    assert "main" in source
    assert "pto.ptr(pto.f32" in source if backend == "pto" else "float" in source


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_native_unit_returns_preserve_early_exit(backend):
    output = TensorView.create("output", [1], "int32")
    flag = ir.Var("flag", "int32")
    function = _function(
        [output.tensor, flag],
        [ir.If(flag > 0, ir.Return(ir.Tuple([])), None), output.store(0, 9), ir.Return(ir.Tuple([]))],
    )
    source = _source(function, backend)
    assert source.count("return") >= 2
    if backend == "pto":
        syntax = ast.parse(source)
        returns = sorted(node.lineno for node in ast.walk(syntax) if isinstance(node, ast.Return))
        stores = [node.lineno for node in ast.walk(syntax) if isinstance(node, ast.Constant) and node.value == 9]
        assert returns[0] < stores[0] < returns[-1]
    else:
        assert source.index("return") < source.index(" = 9") < source.rindex("return")


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_native_simt_scope_allocations_and_captures(backend):
    output = TensorView.create("output", [32], "float32")
    thread = ir.Var("thread", "int32")
    local = TensorView.create("local_value", [1], "float32", scope="local")
    body = [local.allocate(), local.store(0, ir.const(3.0, "float32")), output.store(thread, local.load(0))]
    launch = ir.RegionStmt("tirx.launch_thread", ["threadIdx.x", 32], [thread], {}, body)
    source = _source(_function([output.tensor], [_scope("SIMT_VF", launch)]), backend)
    assert "local_value" in source
    assert "output" in source
    assert "get_tid_x" in source if backend == "pto" else "threadIdx.x" in source


def test_native_simd_scope_retains_vecscope_and_nested_barrier():
    inner = _scope("simdvf", [], **{"tl.simdvf_scope": ir.const(1)})
    body = _scope("simdvf", [inner], **{"tl.simdvf_scope": ir.const(1)})
    source = _source(_function([], [_scope("SIMD_VF", body)]))
    assert source.count("with pto.vecscope():") == 1
    assert 'pto.mem_bar("VST_VLD")' in source


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_native_ascend_dma_copy_lowering(backend):
    if tvm_ffi.get_global_func("tl.testing.LowerOperator", allow_missing=True) is None:
        pytest.fail("Requires the native operator test helpers")
    source = TensorView.create("source", [2, 16], "float32")
    destination = TensorView.create("destination", [2, 16], "float32", scope="shared.dyn")
    operation = tvm_ffi.get_global_func("tl.ParseOperator")(ir.Call("tl.tileop.ascend_copy", [source.tensor, destination.tensor]), {})
    target = Target("ascend")
    body = tvm_ffi.get_global_func("tl.testing.LowerOperator")(operation, target, ir.Range(0, 1), ir.const(0), {}, {})
    generated = _source(_function([source.tensor], [destination.allocate(), body]), backend)
    assert "mte" in generated if backend == "pto" else "asc_copy_gm2ub_align" in generated


@pytest.mark.parametrize("step", [1, 2, -2])
def test_pto_preserves_native_loop_bounds_and_step(step):
    index = ir.Var("index", "int32")
    loop = ir.For(index, 8 if step < 0 else 0, 0 if step < 0 else 8, step, ir.ForKind.DEFAULT, [])
    source = _source(_function([], [loop]))
    assert f", {step}):" in source


def test_pto_loop_local_unroll_factor():
    index = ir.Var("index", "int32")
    loop = ir.For(index, 0, 8, 1, ir.ForKind.UNROLLED, [], annotations={"pragma_unroll_factor": 2})
    source = _source(_function([], [loop]))
    assert "unroll_factor=2" in source


def test_pto_correlates_immutable_direction_guards():
    def body(flag, mad):
        return [
            ir.If(ir.prim.And(flag > 0, flag < 10), [_set_direction("n")], None),
            ir.If(flag > 0, [ir.If(flag < 10, [mad(), _set_direction("m")], None)], None),
            mad(),
        ]

    source = _source(_mad_program(body))
    directions = ["n_dir=True" in line for line in source.splitlines() if line.strip().startswith("pto.mad(")]
    assert directions == [True, False]


def test_pto_rejects_ambiguous_direction():
    with pytest.raises(Exception, match="MAD direction depends on control flow"):
        _source(_mad_program(lambda flag, mad: [ir.If(flag > 0, [_set_direction("n")], None), mad()]))


def test_pto_native_break_preserves_reaching_direction():
    def body(flag, mad):
        return [ir.While(ir.const(True), [_set_direction("n"), ir.Break(), _set_direction("m")]), mad()]

    source = _source(_mad_program(body))
    assert "break" in source
    assert "n_dir=True" in source


@pytest.mark.parametrize("backend", ["ascend", "pto"])
def test_native_mixed_sections_capture_scalar_allocation(backend):
    local = TensorView.create("shared_scalar", [1], "int32", scope="local.var")
    function = _mad_program(
        lambda flag, mad: [
            local.allocate(),
            local.store(0, 1),
            _scope("CUBE", [local.store(0, 2), mad()]),
            _scope("VECTOR", [local.store(0, 3)]),
        ]
    )
    source = _source(function, backend)
    assert "shared_scalar" in source
    if backend == "pto":
        assert "tl.finalize_mixed_kernel(main)" in source
        assert source.count("with tl.mixed_kernel_section(") == 2


@pytest.mark.parametrize("binary", [b"test-binary", bytearray(b"test-binary"), b""])
def test_ascend_compile_callback_accepts_bytes_and_rejects_empty(binary):
    name = "tilelang_callback_ascend_compile"
    previous = tvm_ffi.get_global_func(name, allow_missing=True)
    tvm_ffi.register_global_func(name, lambda *arguments: binary, override=True)
    try:
        build = tvm_ffi.get_global_func("target.build.tilelang_ascend")
        module = ir.IRModule({"main": _function([], [])})
        if not binary:
            with pytest.raises(Exception, match="empty binary"):
                build(module, Target("ascend"))
        else:
            result = build(module, Target("ascend"))
            assert "main" in result.inspect_source("asc")
    finally:
        if previous is None:
            tvm_ffi.remove_global_func(name)
        else:
            tvm_ffi.register_global_func(name, previous, override=True)
