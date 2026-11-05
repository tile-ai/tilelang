"""Non-author checks for new LLVM consumers and integer-carried pointer provenance."""

import re
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target
from testing.python.transform import test_tilelang_native_shared_arena as author
from testing.python.transform.test_tilelang_native_volatile_alias import _kernel


def llvm_source(function, tmp_path):
    function = tirx.Function(
        function.params,
        [*function.body.seq[0].body.seq, ir.Return(ir.const(0, "int32"))],
        ret_type=ir.PrimType("int32"),
        attrs=ir.DictAttrs({"global_symbol": "main"}),
    )
    assert tirx.analysis.check_well_formed(function)
    (tmp_path / "input.py").write_text(function.script())
    module = tvm_ffi.get_global_func("target.build.llvm")(tvm_tirx.IRModule({"main": function}), Target("llvm"))
    source = module.inspect_source("ll")
    (tmp_path / "kernel.ll").write_text(source)
    return source


@pytest.mark.parametrize("indirect", [False, True])
@pytest.mark.parametrize("merged", [False, True])
def test_llvm_preserves_volatile_offset_alias(indirect, merged, tmp_path):
    source = llvm_source(_kernel(indirect=indirect, merged=merged), tmp_path)
    assert re.search(r"load volatile i32", source), source


def test_llvm_false_volatile_is_false(tmp_path):
    source = llvm_source(_kernel(indirect=True, merged=False, qualifier=False), tmp_path)
    assert not re.search(r"\b(?:load|store) volatile\b", source), source


def integer_pointer_kernel(merged):
    scratch = author.tensor("volatile_scratch", 8, scope="shared")
    second = author.tensor("ordinary_scratch", 8, scope="shared")
    output = ir.Var("output_ptr", ir.PtrType("int32", scope="global"))
    result = author.tensor("output", 1, scope="global")
    encoded = ir.Var("encoded_address", "uint64")
    pointer = ir.Var("decoded_address", scratch.data.ty)
    view = author.tensor("view", 8, scope="shared")
    stmts = [
        ir.Bind(result, ir.Call("tirx.decl_tensor", [output], ty_args=[result.ty])),
        author.allocate(scratch, {"tirx.volatile": True}),
        author.store(scratch, 7),
        ir.Bind(encoded, tirx.reinterpret(scratch.data, ty="uint64")),
        author.allocate(second),
        author.store(second, 9),
        ir.Bind(pointer, tirx.reinterpret(encoded, ty=pointer.ty)),
        ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])),
        author.store(result, author.load(view)),
    ]
    func = author.function(stmts, [output])
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
    "backend,target,format",
    [("cuda", {"kind": "cuda", "arch": "sm_90a"}, "cuda"), ("hip", "rocm", "hip"), ("metal", "metal", "metal"), ("llvm", "llvm", "ll")],
)
@pytest.mark.parametrize("merged", [False, True])
def test_integer_pointer_transport_preserves_volatile(backend, target, format, merged, tmp_path):
    func = integer_pointer_kernel(merged)
    assert tirx.analysis.check_well_formed(func)
    if backend == "llvm":
        source = llvm_source(func, tmp_path)
        assert re.search(r"load volatile i32", source), source
    else:
        (tmp_path / "input.py").write_text(func.script())
        module = tvm_ffi.get_global_func("target.build.tilelang_" + backend + "_without_compile")(
            tvm_tirx.IRModule({"main": func}), Target(target)
        )
        source = module.inspect_source(format)
        (tmp_path / ("kernel." + format)).write_text(source)
        output_line = next(line for line in source.splitlines() if re.search(r"output_ptr\[0\]\s*=", line))
        assert re.search(r"\bvolatile\b", output_line), output_line
