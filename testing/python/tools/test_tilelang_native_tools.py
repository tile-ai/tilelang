"""Debugging tools consume native IR and real source modules without legacy TVM."""

import re
import sys

import pytest
import torch
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.tirx.layout import Iter

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.tools.lower_trace import disable, enable
from tilelang.tools.lower_trace.core import _make_patched_source_module
from tilelang.tools.pass_visualizer import capture_structure
from tilelang.tools.pass_visualizer.core import kernel_to_tir


def test_structure_tree_preserves_native_parameter_layout_and_thread_scope():
    n = ir.Var("n", "int32")
    tensor = ir.Var(
        "source",
        tirx.TensorType(
            [n],
            "float32",
            align=16,
            layout=tirx.TileLayout.from_iters([Iter(n, 2, "m")], offset={"m": 3}),
        ),
    )
    thread = ir.Var("worker", "int32")
    i = ir.Var("i", "int32")
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(32)],
        [thread],
        {},
        [ir.For(i, 0, n, 2, ir.ForKind.DEFAULT, [ir.Evaluate(tensor[i])])],
    )
    function = tirx.Function([n, tensor], [body]).with_attr("global_symbol", "native")
    before = ir.save_json(function)
    text = "\n".join(capture_structure(ir.IRModule({"native": function})))
    assert "param_types" in text and "buffer_map" not in text
    assert "source" in text and "float32" in text and "align=16" in text
    assert "layout=" in text and "m" in text and "3" in text
    assert "RegionStmt(tirx.launch_thread)" in text
    assert "body_params" in text and "worker" in text and "result_vars" in text
    assert "step=2" in text and "threadIdx.x" in text
    assert ir.save_json(function) == before
    assert kernel_to_tir(function).same_as(function)
    assert kernel_to_tir(lambda: function).same_as(function)
    with pytest.raises(SystemExit, match="expected a Function"):
        kernel_to_tir(lambda: 1)
    assert "tvm" not in sys.modules


def test_structure_tree_decodes_tensor_annotations_and_native_tileop_operands():
    @T.prim_func
    def kernel(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((32,), "float32")
            T.annotate_layout({shared: T.Layout((32,), lambda i: i)})
            T.copy(A, shared, annotations={"coalesced_width": 2})
            T.copy(shared, B)

    text = "\n".join(capture_structure(ir.IRModule({"main": kernel})))
    assert "RegionStmt(tl.launch_kernel)" in text
    assert "Evaluate: tl.layout_map" in text
    assert "Bind(Tensor(shared" in text
    assert "layout_map" in text and "shared" in text
    assert "coalesced_width" in text
    assert "T.copy" in text and "src_region" in text and "dst_region" in text
    assert "TupleExpr" not in text
    assert "buffer_map" not in text and "SBlock" not in text


def test_patched_c_source_keeps_real_exports_constants_and_imports(tmp_path):
    create = tvm_ffi.get_global_func("runtime.CSourceModuleCreate")
    source = "int alpha(void) { return 3; }\nint beta(void) { return 5; }\n"
    original = create(source, "c", ["alpha", "beta"], ["weight"])
    dependency = create("int helper(void) { return 1; }\n", "c", ["helper"], [])
    original.import_module(dependency)
    edited = source.replace("return 3", "return 7")
    patched = _make_patched_source_module(original, edited)
    assert patched.inspect_source() == edited
    assert original.inspect_source() == source
    assert list(patched.get_write_formats()) == ["c"]
    assert list(patched["get_func_names"]()) == ["alpha", "beta"]
    assert list(patched["get_const_vars"]()) == ["weight"]
    assert patched.implements_function("alpha") and patched.implements_function("beta")
    assert patched.implements_function("helper", query_imports=True)
    assert len(patched.imports) == 1 and patched.imports[0].same_as(dependency)
    assert patched.is_compilation_exportable() and not patched.is_runnable()
    path = tmp_path / "edited.c"
    patched.write_to_file(str(path), "c")
    assert path.read_text() == edited


def test_patched_generic_source_retains_its_write_format():
    original = tvm_ffi.get_global_func("runtime.SourceModuleCreate")("// initial\n", "metal")
    patched = _make_patched_source_module(original, "// edited\n")
    assert patched.inspect_source() == "// edited\n"
    assert list(patched.get_write_formats()) == ["metal"]
    assert original.inspect_source() == "// initial\n"


@tilelang.testing.requires_cuda
def test_codegen_edit_recompiles_actual_native_cuda_source(tmp_path):
    @T.prim_func
    def add(A: T.Tensor((32,), "int32"), B: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                B[i] = A[i] + 3

    from tilelang.jit.adapter.nvrtc import is_nvrtc_available

    if not is_nvrtc_available:
        pytest.skip("NVRTC source-edit execution requires the CUDA Python bindings")
    source = tmp_path / "edited.cu"
    enable(mode="html", trace_dir=str(tmp_path / "traces"), codegen_output=str(source))
    try:
        first = tilelang.compile(add, target="cuda", out_idx=[1], execution_backend="nvrtc")
        data = torch.arange(32, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(first(data), data + 3, rtol=0, atol=0)
        generated = source.read_text()
        edited, count = re.subn(r"\+ 3(?=[;)])", "+ 7", generated)
        assert count == 1, generated
        source.write_text(edited)
        second = tilelang.compile(add, target="cuda", out_idx=[1], execution_backend="nvrtc")
        torch.testing.assert_close(second(data), data + 7, rtol=0, atol=0)
        assert second.get_kernel_source() == edited
        assert source.read_text() == edited
        assert "tvm" not in sys.modules
    finally:
        disable()
