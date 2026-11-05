"""Independent volatile source consumers, including typed pointer aliases."""

import re

import pytest
import tvm_ffi
import tvm_tirx as tvm
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

from testing.python.transform import test_tilelang_native_shared_arena as author


def _kernel(*, indirect, merged, scope="shared", qualifier=True):
    scratch = author.tensor("volatile_scratch", 8, scope=scope)
    second = author.tensor("ordinary_scratch", 8, scope=scope)
    output = ir.Var("output_ptr", ir.PtrType("int32", scope="global"))
    result = author.tensor("output", 1, scope="global")
    pointer = ir.Var("offset_pointer", scratch.data.ty)
    view = author.tensor("view", 7, scope=scope)
    statements = [
        ir.Bind(result, ir.Call("tirx.decl_tensor", [output], ty_args=[result.ty])),
        author.allocate(scratch, {"tirx.volatile": qualifier}),
        author.store(scratch, 7, 1),
        author.allocate(second),
        author.store(second, 9),
    ]
    base = ir.Call("tirx.ptr_byte_offset", [scratch.data, ir.const(4, "int64")], ty=pointer.ty)
    if indirect:
        statements.append(ir.Bind(pointer, base))
        base = pointer
    statements.extend(
        [
            ir.Bind(view, ir.Call("tirx.decl_tensor", [base], ty_args=[view.ty])),
            author.store(result, author.load(view)),
        ]
    )
    func = author.function(statements, [output])
    if merged:
        func = author.merge(func, static=True, reuse=False)
    tx = ir.Var("tx", "int32")
    return tirx.Function(
        [output],
        [ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(32, "int32")], [tx], {}, func.body)],
        ret_type=ir.PrimType("void"),
        attrs=ir.DictAttrs({"global_symbol": "main", "calling_conv": 2}),
    )


@pytest.mark.parametrize(
    "backend,target,format", [("cuda", {"kind": "cuda", "arch": "sm_90a"}, "cuda"), ("hip", "rocm", "hip"), ("metal", "metal", "metal")]
)
@pytest.mark.parametrize("indirect", [False, True])
@pytest.mark.parametrize("merged", [False, True])
def test_volatile_access_after_offset_alias(backend, target, format, indirect, merged, tmp_path):
    func = _kernel(indirect=indirect, merged=merged)
    assert tirx.analysis.check_well_formed(func)
    (tmp_path / "input.py").write_text(func.script())
    module = tvm_ffi.get_global_func("target.build.tilelang_" + backend + "_without_compile")(tvm.IRModule({"main": func}), Target(target))
    source = module.inspect_source(format)
    (tmp_path / ("kernel." + format)).write_text(source)
    output_line = next(line for line in source.splitlines() if re.search(r"output_ptr\[0\]\s*=", line))
    assert re.search(r"\bvolatile\b", output_line), output_line


@pytest.mark.parametrize(
    "backend,target,format", [("cuda", {"kind": "cuda", "arch": "sm_90a"}, "cuda"), ("hip", "rocm", "hip"), ("metal", "metal", "metal")]
)
@pytest.mark.parametrize("merged", [False, True])
def test_false_volatile_attribute_is_not_treated_as_true(backend, target, format, merged, tmp_path):
    func = _kernel(indirect=False, merged=merged, qualifier=False)
    module = tvm_ffi.get_global_func("target.build.tilelang_" + backend + "_without_compile")(tvm.IRModule({"main": func}), Target(target))
    source = module.inspect_source(format)
    (tmp_path / ("kernel." + format)).write_text(source)
    output_line = next(line for line in source.splitlines() if re.search(r"output_ptr\[0\]\s*=", line))
    assert not re.search(r"\bvolatile\b", output_line), output_line
