"""Native loop hints reach the CUDA and CuTeDSL source emitters."""

import ast

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang  # noqa: F401  (register the TileLang emitters and metadata codec)


def loop_source(backend, annotation, carrier=None, *, unrolled=True):
    index = ir.Var("i", "int32")
    body = ir.Evaluate(ir.Call("tirx.call_extern", [ir.StringImm("consume"), index], ty="void"))
    annotations = {} if annotation is None else {"pragma_unroll_factor": annotation}
    loop = ir.For(
        index,
        ir.const(0),
        ir.const(8),
        ir.const(1),
        ir.ForKind.UNROLLED if unrolled else ir.ForKind.DEFAULT,
        [body],
        annotations=annotations,
    )
    statement = loop
    if carrier is not None:
        schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(
            {"node": index, "key": "pragma_unroll_factor", "value": ir.const(carrier)}
        )
        statement = ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, [loop])
    function = tirx.Function(
        [],
        [statement],
        ir.PrimType("void"),
        ir.DictAttrs({"global_symbol": "native_unroll", "calling_conv": 2}),
    )
    module = ir.IRModule({"native_unroll": function})
    before = ir.save_json(module)
    builder = tvm_ffi.get_global_func("target.build.tilelang_" + backend + "_without_compile")
    runtime = builder(module, Target({"kind": "cuda", "arch": "sm_100a"}))
    assert ir.save_json(module) == before
    return runtime.inspect_source("cutedsl" if backend == "cutedsl" else "cuda")


@pytest.mark.parametrize("backend", ["cuda", "cutedsl"])
@pytest.mark.parametrize("hint_type", ["host_integer", "native_int32", "native_int64"])
def test_native_unroll_factor_is_not_dropped(backend, hint_type):
    hint = 4 if hint_type == "host_integer" else ir.IntImm("int32" if hint_type == "native_int32" else "int64", 4)
    source = loop_source(backend, hint)
    if backend == "cuda":
        assert "#pragma unroll 4\n" in source
    else:
        tree = ast.parse(source)
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For))
        assert isinstance(loop.iter, ast.Call)
        assert loop.iter.func.attr == "range"
        factor = next(keyword.value for keyword in loop.iter.keywords if keyword.arg == "unroll")
        assert isinstance(factor, ast.Constant) and factor.value == 4


@pytest.mark.parametrize("backend", ["cuda", "cutedsl"])
@pytest.mark.parametrize("annotation,carrier,expected", [(None, 2, 2), (4, 8, 4)])
def test_loop_local_hint_precedes_attribute_fallback(backend, annotation, carrier, expected):
    source = loop_source(backend, annotation, carrier)
    if backend == "cuda":
        assert f"#pragma unroll {expected}\n" in source
    else:
        assert f", unroll={expected})" in source


@pytest.mark.parametrize("backend", ["cuda", "cutedsl"])
def test_default_unroll_policy_and_serial_kind_are_preserved(backend):
    default = loop_source(backend, None)
    serial = loop_source(backend, 4, unrolled=False)
    if backend == "cuda":
        assert "#pragma unroll\n" in default
        assert "#pragma unroll" not in serial
    else:
        assert "cutlass.range_constexpr(" in default
        assert "unroll=" not in serial
        assert "range_constexpr(" not in serial
