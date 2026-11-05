"""A statement-only handle annotation must not cast a C helper's void result."""

import json
import os
from pathlib import Path
import re

import pytest
import tvm_ffi
import tvm_tirx as tvm
from tvm_tirx import ir, tirx

import tilelang.testing
from tilelang.contrib import nvcc
from tilelang.cuda.backend import tilelang_callback_cuda_compile


CODEGENS = [
    ("c", "c"),
    ("c_host", "c"),
    ("cuda_without_compile", {"kind": "cuda", "arch": "sm_100"}),
    ("hip_without_compile", {"kind": "hip", "mcpu": "gfx90a"}),
    ("metal_without_compile", "metal"),
    ("ascend_without_compile", "ascend"),
]


def _compile_cuda(source):
    try:
        compiler = nvcc.get_nvcc_compiler()
    except RuntimeError as error:
        pytest.skip(str(error))
    if not Path(compiler).is_file():
        pytest.skip(f"NVCC is not installed: {compiler}")
    return tilelang_callback_cuda_compile(source, tvm.target.Target({"kind": "cuda", "arch": "sm_100"}))


def _source(codegen, target, statements, *, params=(), imported="", label="kernel"):
    target = tvm.target.Target(target)
    body = ir.SeqStmt(statements)
    if imported:
        schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": None, "key": "pragma_import_c", "value": imported})
        body = ir.SeqStmt(
            [
                ir.RegionStmt(
                    "tl.attr",
                    [payload],
                    [],
                    {
                        "schema_version": 1,
                        "annotation_schema": schema,
                    },
                    body,
                )
            ]
        )
    function = tirx.Function(list(params), body, ir.PrimType("void")).with_attr("global_symbol", label)
    if target.kind.name != "c":
        function = function.with_attr("calling_conv", ir.CallingConv.DEVICE_KERNEL_LAUNCH)
        function = function.with_attr("tirx.kernel_launch_params", [])
    build = tvm_ffi.get_global_func(f"target.build.tilelang_{codegen}", allow_missing=True)
    if build is None:
        pytest.skip(f"{codegen} codegen is not built")
    result = build(tvm.IRModule.from_expr(function), target).inspect_source()
    artifacts = os.environ.get("EXTERN79_ARTIFACTS")
    if artifacts:
        out = Path(artifacts) / (label + "-" + codegen + ".txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result)
    return result


@pytest.mark.parametrize(("codegen", "target"), CODEGENS)
@pytest.mark.parametrize("pure", [False, True])
def test_discarded_pointer_call_has_no_result_cast(codegen, target, pure):
    op = "tirx.call_pure_extern" if pure else "tirx.call_extern"
    call = ir.Call(op, [ir.StringImm("discard_pointer"), ir.IntImm("int32", 7)], ty=ir.PtrType("void"))
    source = _source(codegen, target, [ir.Evaluate(call)], label=f"discard_{int(pure)}")
    assert re.search(r"\n\s*discard_pointer\(7\);", source), source
    assert "((void*)discard_pointer" not in source
    assert call.ty == ir.PtrType("void")
    if codegen in {"c", "c_host"}:
        # Ignoring a result does not authorize changing forward declarations.
        assert re.search(r"void\s*\*\s*discard_pointer\(int(?:32_t)?\);", source), source


@pytest.mark.parametrize(("codegen", "target"), CODEGENS)
def test_discarded_outer_call_keeps_nested_pointer_result_cast(codegen, target):
    inner = tirx.call_extern("make_pointer", ty=ir.PtrType("float32"))
    outer = tirx.call_extern("discard_pointer", inner, ty=ir.PtrType("void"))
    source = _source(codegen, target, [ir.Evaluate(outer)], label="nested_result")
    assert "((void*)discard_pointer" not in source
    assert re.search(r"discard_pointer\(\(\([^\n]*float\*\)make_pointer\(\)\)\);", source), source


@pytest.mark.parametrize(("codegen", "target"), CODEGENS)
def test_value_used_pointer_call_keeps_result_cast(codegen, target):
    result = ir.Var("pointer_value", ir.PtrType("float32"))
    source = _source(
        codegen,
        target,
        [
            ir.Bind(result, tirx.call_extern("make_pointer", ty=result.ty)),
            ir.Evaluate(tirx.call_extern("consume_pointer", result, ty="int32")),
        ],
        label="used_result",
    )
    assert re.search(r"\(\([^\n]*float\*\)make_pointer\(\)\)", source), source
    assert "consume_pointer(pointer_value)" in source


@pytest.mark.parametrize("codegen", ["c", "c_host"])
def test_discarded_internal_call_keeps_existing_void_declaration(codegen):
    helper = tirx.Function([], ir.SeqStmt([]), ir.PrimType("void")).with_attr("global_symbol", "internal_helper")
    caller = tirx.Function(
        [], ir.SeqStmt([ir.Evaluate(tirx.call_extern("internal_helper", ty=ir.PtrType("void")))]), ir.PrimType("void")
    ).with_attr("global_symbol", "caller")
    module = tvm.IRModule({"internal_helper": helper, "caller": caller})
    source = tvm_ffi.get_global_func(f"target.build.tilelang_{codegen}")(module, tvm.target.Target("c")).inspect_source()
    assert re.search(r"\n\s*internal_helper\(\);", source), source
    assert not re.search(r"void\s*\*\s*internal_helper", source), source
    assert "((void*)internal_helper" not in source


@pytest.mark.parametrize("nested", [False, True])
@tilelang.testing.requires_cuda(support_required="compile-only")
def test_cuda_void_helper_discarded_result_compiles(nested):
    pointer = ir.Var("pointer", ir.PtrType("float32"))
    argument = tirx.call_extern("make_pointer", pointer, ty=pointer.ty) if nested else pointer
    call = tirx.call_extern("discard_pointer", argument, ty=ir.PtrType("void"))
    source = _source(
        "cuda_without_compile",
        {"kind": "cuda", "arch": "sm_100"},
        [ir.Evaluate(call)],
        params=[pointer],
        label=f"cuda_discard_{int(nested)}",
        imported="""
__device__ void* make_pointer(float* input) { return input; }
__device__ void discard_pointer(float* input) { input[0] += 1.0f; }
""",
    )
    binary = _compile_cuda(source)
    assert binary
    print(json.dumps({"case": "discarded_void", "nested": nested, "cubin_bytes": len(binary)}))


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_cuda_used_void_helper_remains_a_type_error():
    result = ir.Var("pointer_value", ir.PtrType("float32"))
    source = _source(
        "cuda_without_compile",
        {"kind": "cuda", "arch": "sm_100"},
        [
            ir.Bind(result, tirx.call_extern("returns_void", ty=result.ty)),
            ir.Evaluate(tirx.call_extern("consume_pointer", result, ty="int32")),
        ],
        label="cuda_used_void",
        imported="""
__device__ void returns_void() {}
__device__ int consume_pointer(float* input) { return input != nullptr; }
""",
    )
    assert "((float*)returns_void())" in source
    with pytest.raises(RuntimeError, match="invalid type conversion"):
        _compile_cuda(source)


@tilelang.testing.requires_cuda(support_required="compile-only")
def test_cuda_used_real_pointer_result_compiles():
    pointer = ir.Var("pointer", ir.PtrType("float32"))
    result = ir.Var("pointer_value", ir.PtrType("float32"))
    source = _source(
        "cuda_without_compile",
        {"kind": "cuda", "arch": "sm_100"},
        [
            ir.Bind(result, tirx.call_extern("make_pointer", pointer, ty=result.ty)),
            ir.Evaluate(tirx.call_extern("consume_pointer", result, ty="int32")),
        ],
        params=[pointer],
        label="cuda_used_pointer",
        imported="""
__device__ void* make_pointer(float* input) { return input; }
__device__ int consume_pointer(float* input) { return input != nullptr; }
""",
    )
    assert "((float*)make_pointer(pointer))" in source
    binary = _compile_cuda(source)
    assert binary


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
