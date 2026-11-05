"""Raw import attributes must work in every native C-like codegen.

These tests bypass lowering so that source boundaries cannot be repaired by
frontend or transform passes. They need codegen registrations, not devices or
backend toolchains.
"""

import re

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang.testing

CODEGENS = [
    pytest.param("c", "c", 'extern "C"', id="cpu-c"),
    pytest.param("c_host", "c", 'extern "C"', id="c-host"),
    pytest.param(
        "cuda_without_compile",
        {"kind": "cuda", "arch": "sm_80"},
        "__device__",
        marks=tilelang.testing.requires_cuda.marks(support_required="compile-only"),
        id="cuda",
    ),
    pytest.param(
        "hip_without_compile",
        {"kind": "hip", "mcpu": "gfx90a"},
        "__device__",
        marks=tilelang.testing.requires_rocm.marks(support_required="compile-only"),
        id="hip",
    ),
    # Metal source codegen is available without the runtime; check its FFI
    # registration in _generate_source instead of requiring a Metal runtime.
    pytest.param(
        "metal_without_compile",
        "metal",
        "inline",
        marks=tilelang.testing.requires_metal.marks(support_required="optional"),
        id="metal",
    ),
    pytest.param(
        "ascend_without_compile",
        "ascend",
        "__aicore__ inline",
        marks=tilelang.testing.requires_ascend.marks(support_required="compile-only"),
        id="ascend",
    ),
]
FIRST_HELPER = "int my_helper(int x) { return x + 1; }"
SECOND_HELPER = "int call_helper(int x) { return my_helper(x); }"


def _generate_source(codegen, target, imports):
    build = tvm_ffi.get_global_func(f"target.build.tilelang_{codegen}", allow_missing=True)
    if build is None:
        pytest.skip(f"{codegen} codegen is not built")
    target = tvm_tirx.target.Target(target)
    body = ir.SeqStmt([ir.Evaluate(tirx.call_extern("call_helper", ir.IntImm("int32", 1), ty="int32"))])
    for source in reversed(imports):
        schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": None, "key": "pragma_import_c", "value": source})
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
    func = tirx.Function([], body, ir.PrimType("void")).with_attr("global_symbol", "import_kernel")
    if target.kind.name != "c":
        func = func.with_attr("calling_conv", tvm_tirx.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
        func = func.with_attr("tirx.kernel_launch_params", [])
    return build(tvm_tirx.IRModule.from_expr(func), target).inspect_source()


@pytest.mark.parametrize(("codegen", "target", "qualifier"), CODEGENS)
@pytest.mark.parametrize("suffix", ["", "\n", " // trailing comment", " // trailing comment \\", " \\\n"])
def test_raw_import_boundaries(codegen, target, qualifier, suffix):
    first = f"{qualifier} {FIRST_HELPER}{suffix}"
    second = f"{qualifier} {SECOND_HELPER}"
    source = _generate_source(codegen, target, [first, second])

    assert 0 <= source.rfind("#include") < source.index(first)
    assert source.index(first) < source.index(second) < source.index("import_kernel(")

    # C line splicing happens before comment removal. Neither import may
    # consume the next helper or the generated kernel.
    logical_source = re.sub(r"\\\r?\n", "", source)
    logical_source = re.sub(r"//[^\n]*", "", logical_source)
    assert second in logical_source.splitlines()
    assert "import_kernel(" in logical_source
    assert "call_helper(1)" in logical_source


@pytest.mark.parametrize(("codegen", "target", "qualifier"), CODEGENS)
def test_empty_raw_import(codegen, target, qualifier):
    first = f"{qualifier} {FIRST_HELPER}"
    second = f"{qualifier} {SECOND_HELPER}"
    source = _generate_source(codegen, target, ["", first, second])
    assert source.index(first) < source.index(second) < source.index("import_kernel(")
    assert "call_helper(1)" in source


def test_raw_import_requires_string():
    with pytest.raises(RuntimeError, match="pragma_import_c expects a string-valued source"):
        _generate_source("c_host", "c", [ir.IntImm("int32", 1)])


if __name__ == "__main__":
    tilelang.testing.main()
