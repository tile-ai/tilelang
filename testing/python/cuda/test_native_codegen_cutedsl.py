"""TileLang CuTeDSL emitter over actual native tensors/regions and CUDA execution."""

import ast
import importlib.util
import os
from pathlib import Path
import site
import sys

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

ROOT = Path(__file__).resolve().parents[3]


def _literal(value, dtype="int32"):
    return ir.const(value, dtype)


def _tensor(name, size, dtype="float32", scope="global"):
    return ir.Var(name, tirx.TensorType([size], dtype, scope=scope, layout="default", data_alignment=4))


def _call(name, *args, dtype="void", attrs=None):
    operands = [arg if isinstance(arg, ir.Expr) else ir.const(arg) for arg in args]
    return ir.Call(name, operands, ty_args=[ir.PrimType(dtype)], attrs=attrs)


def _thread(body, tx):
    return ir.RegionStmt(
        "tirx.launch_thread", args=[ir.StringImm("threadIdx.x"), _literal(32)], body_params=[tx], attrs={}, body=body, result_vars=[]
    )


def _source(parameters, body, *, name="native_kernel", tmp_path=None):
    func = tirx.Function(parameters, body, ir.PrimType("void"))
    func = func.with_attr("global_symbol", name).with_attr("calling_conv", 2)
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"})
    if any(isinstance(parameter.ty, tirx.TensorType) for parameter in parameters):
        host_target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"}, host="c")
        func = func.without_attr("calling_conv").with_attr("target", host_target)
        with host_target:
            native = tirx.transform.LowerTensorLayouts()(ir.IRModule({name: func}))
            native = tirx.transform.FlattenTensor()(native)
            native = tirx.transform.SplitHostDevice()(native)
        devices = {gvar: f.with_attr("global_symbol", name) for gvar, f in native.functions.items() if f.attrs.get("calling_conv", 0) == 2}
        assert len(devices) == 1
        native = ir.IRModule(devices)
    else:
        native = ir.IRModule({name: func})
    module = tvm_ffi.get_global_func("target.build.tilelang_cutedsl_without_compile")(native, target)
    source = module.inspect_source("cuda")
    ast.parse(source)
    assert source == module.inspect_source("cutedsl")
    if tmp_path:
        (tmp_path / "native.json").write_text(ir.save_json(func))
        (tmp_path / "device.py").write_text(source)
    return source


def _load_cute_helpers():
    extra = os.environ.get("TILELANG_TEST_CUTEDSL_SITE")
    if extra:
        site.addsitedir(extra)
    pytest.importorskip("cutlass")
    name = "_tilelang_native_cutedsl_helpers"
    if name not in sys.modules:
        path = ROOT / "tilelang/contrib/cutedsl/__init__.py"
        spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _run(source, data, output, tmp_path, *, dynamic_size=None, shared_bytes=0):
    if not tvm_tirx.cuda(0).exist:
        pytest.skip("CUDA device required for CuTeDSL numerical validation")
    _load_cute_helpers()
    import cutlass.cute as cute
    from cuda.bindings import driver

    # The real emitted kernel remains unchanged. This wrapper supplies the
    # launch boundary normally owned by the CuTeDSL JIT adapter.
    dynamic = dynamic_size is not None
    wrapper = """\n@cute.jit
def launch(a: cute.Tensor, b: cute.Tensor, stream: driver.CUstream):
    native_kernel(a, b%s).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream, smem=%d)
""" % ((", cutlass.Int32(%d)" % dynamic_size) if dynamic else "", shared_bytes)
    path = tmp_path / "executable.py"
    path.write_text(
        "import cutlass\nimport cutlass.cute as cute\nfrom cuda.bindings import driver\n"
        "import _tilelang_native_cutedsl_helpers as tl\n" + source + wrapper
    )
    name = "_native_cutedsl_" + tmp_path.name.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    a, b = tvm_tirx.runtime.tensor(data, tvm_tirx.cuda(0)), tvm_tirx.runtime.tensor(output, tvm_tirx.cuda(0))
    ca, cb = cute.runtime.from_dlpack(a), cute.runtime.from_dlpack(b)
    compiled = cute.compile(module.launch, ca, cb, driver.CUstream(0))
    compiled(ca, cb, driver.CUstream(0))
    tvm_tirx.cuda(0).sync()
    return b.numpy()


@pytest.mark.parametrize("n", [1, 17, 32])
def test_native_cutedsl_dynamic_add(n, tmp_path):
    size = ir.Var("size", "int32")
    a, b = _tensor("a", size), _tensor("b", size)
    tx = ir.Var("tx", "int32")
    value = ir.TensorLoad(a, [tx]) + _literal(1.25, "float32")
    body = _thread([ir.If(tx < size, [ir.TensorStore(b, [tx], value)], None)], tx)
    source = _source([a, b, size], [body], tmp_path=tmp_path)
    data = np.arange(n, dtype="float32") - 13
    actual = _run(source, data, np.full(n, -99, dtype="float32"), tmp_path, dynamic_size=n)
    np.testing.assert_array_equal(actual, data + np.float32(1.25))


def test_native_cutedsl_shared_exchange(tmp_path):
    a, b, shared = _tensor("a", 32), _tensor("b", 32), _tensor("shared", 32, scope="shared")
    tx = ir.Var("tx", "int32")
    allocation = ir.Bind(shared, ir.Call("tirx.alloc_tensor", [], ty_args=[shared.ty]))
    statements = [
        allocation,
        ir.TensorStore(shared, [tx], ir.TensorLoad(a, [tx])),
        ir.Evaluate(ir.Call("tirx.gpu_storage_sync", [ir.StringImm("shared")])),
        ir.TensorStore(b, [tx], ir.TensorLoad(shared, [(tx + 1) % 32])),
    ]
    source = _source([a, b], [_thread(statements, tx)], tmp_path=tmp_path)
    data = np.arange(32, dtype="float32")
    actual = _run(source, data, np.full(32, -1, dtype="float32"), tmp_path, shared_bytes=128)
    np.testing.assert_array_equal(actual, np.roll(data, -1))


def test_native_cutedsl_source_lexical_empty_branch_and_noop(tmp_path):
    cond = ir.Var("cond", "bool")
    body = [ir.If(cond, [ir.Evaluate(_literal(0))], [ir.Evaluate(_literal(1))])]
    source = _source([cond], body, tmp_path=tmp_path)
    assert "if cond:" in source and source.count("pass") == 2


def test_native_cutedsl_vector_and_narrow_shift_source(tmp_path):
    a, b = _tensor("a", 32, "int8"), _tensor("b", 32, "int8")
    tx = ir.Var("tx", "int32")
    value = ir.prim.RShift(ir.TensorLoad(a, [tx]), _literal(2, "int8"))
    source = _source([a, b], [_thread([ir.TensorStore(b, [tx], value)], tx)], tmp_path=tmp_path)
    assert ">>" in source


@pytest.mark.parametrize("mode", ["truncdiv", "truncmod", "floordiv", "floormod"])
@pytest.mark.parametrize("divisor", [-3, 3])
def test_native_cutedsl_signed_integer_contract(mode, divisor, tmp_path):
    a, b = _tensor("a", 32, "int32"), _tensor("b", 32, "int32")
    tx = ir.Var("tx", "int32")
    node = {"truncdiv": ir.prim.Div, "truncmod": ir.prim.Mod, "floordiv": ir.prim.FloorDiv, "floormod": ir.prim.FloorMod}[mode]
    value = node(ir.TensorLoad(a, [tx]), _literal(divisor))
    source = _source([a, b], [_thread([ir.TensorStore(b, [tx], value)], tx)], tmp_path=tmp_path)
    data = np.arange(-16, 16, dtype="int32")
    if mode == "floordiv":
        expected = data // divisor
    elif mode == "floormod":
        expected = data % divisor
    else:
        quotient = (np.abs(data) // abs(divisor)) * np.where((data < 0) != (divisor < 0), -1, 1)
        expected = quotient if mode == "truncdiv" else data - quotient * divisor
    actual = _run(source, data, np.full(32, -99, dtype="int32"), tmp_path)
    np.testing.assert_array_equal(actual, expected.astype("int32"))


def test_native_cutedsl_pointer_view_byte_offset(tmp_path):
    a, b, view = _tensor("a", 33), _tensor("b", 32), _tensor("view", 32)
    tx = ir.Var("tx", "int32")
    offset = tirx.ptr_byte_offset(a.data, 4, ty=ir.PtrType("float32", "global"))
    pointer = ir.Var("view_ptr", offset.ty)
    statements = [
        ir.Bind(pointer, offset),
        ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])),
        ir.TensorStore(b, [tx], ir.TensorLoad(view, [tx])),
    ]
    source = _source([a, b], [_thread(statements, tx)], tmp_path=tmp_path)
    data = np.arange(33, dtype="float32") - 19
    actual = _run(source, data, np.full(32, -99, dtype="float32"), tmp_path)
    np.testing.assert_array_equal(actual, data[1:])


def test_native_cutedsl_narrow_shift_numeric(tmp_path):
    a, b = _tensor("a", 32, "int8"), _tensor("b", 32, "int8")
    tx = ir.Var("tx", "int32")
    value = ir.prim.RShift(ir.TensorLoad(a, [tx]), _literal(2, "int8"))
    source = _source([a, b], [_thread([ir.TensorStore(b, [tx], value)], tx)], tmp_path=tmp_path)
    data = np.arange(-64, 64, 4, dtype="int8")
    actual = _run(source, data, np.full(32, -99, dtype="int8"), tmp_path)
    np.testing.assert_array_equal(actual, data >> 2)
