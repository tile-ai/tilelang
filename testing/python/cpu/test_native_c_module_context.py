"""Linked C units share a context; separately loaded libraries keep their imports."""

import ctypes
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import tvm_ffi
from tvm_tirx import codegen, ir, runtime, tirx
from tvm_tirx.target import Target
from tvm_tirx.tirx.compilation_pipeline import finalize_host_passes

import tilelang
import tilelang.cpu.language as C


def _cpu_kernel():
    @C.prim_func
    def kernel(output: C.Tensor((1,), "int32"), n: C.int32):
        with C.Kernel(1):
            for i in C.serial(4, 5):
                C.assume(n >= i + 3)
                output[0] = n + i

    return kernel


@pytest.mark.parametrize("device_kind,host_kind", [("c", "c"), ("c", "llvm"), ("llvm", "c")])
def test_c_cpu_host_compute_combinations_execute(device_kind, host_kind, tmp_path):
    if "llvm" in (device_kind, host_kind) and not runtime.enabled("llvm"):
        pytest.skip("Optional LLVM code generation is disabled")

    def target(kind):
        return Target({"kind": "llvm", "mcpu": "generic"} if kind == "llvm" else {"kind": kind})

    device = target(device_kind)
    host = device if host_kind == device_kind else target(host_kind)
    artifact = tilelang.engine.lower(_cpu_kernel(), target=device, target_host=host, enable_host_codegen=True, enable_device_compile=True)
    library = _export(artifact.rt_mod, tmp_path / "cpu")
    loaded = runtime.load_module(str(library))
    output = torch.full((1,), -99, dtype=torch.int32)
    loaded["kernel"](output, 7)
    assert output.tolist() == [11]
    output.fill_(-99)
    with pytest.raises(RuntimeError, match="Assume"):
        loaded["kernel"](output, 6)
    assert output.tolist() == [-99]


def _packed_source(name, callback, *, offset=0, kind="c"):
    target = Target({"kind": "llvm", "mcpu": "generic"} if kind == "llvm" else {"kind": "c"})
    value = ir.Var("value", "int32")
    void = ir.TupleType([])
    effect = tirx.call_packed(callback, value + offset, ty=void)
    function = tirx.Function(
        [value],
        [ir.Evaluate(effect), ir.Return(ir.Tuple([]))],
        void,
        ir.DictAttrs({"global_symbol": name, "target": target.with_host(target)}),
    )
    module = tilelang.transform.MakePackedAPI()(ir.IRModule({name: function}))
    module = finalize_host_passes()(module)
    key = "target.build.tilelang_c_host" if kind == "c" else "target.build.llvm"
    return tvm_ffi.get_global_func(key)(module, target)


def _export(module, directory, *, language="c++"):
    directory.mkdir(parents=True)
    output = directory / "module.so"
    objects = directory / "objects"
    objects.mkdir()
    options = ["-std=c++17"] if language == "c++" else ["-std=c11"]
    if "TL_TEMPLATE_PATH" in os.environ:
        options.append("-I" + os.environ["TL_TEMPLATE_PATH"])
    codegen.Executable(module).export_library(str(output), workspace_dir=str(objects), cc=language, options=options)
    return output


def _loaded_c_source(name, callback, directory, *, language="c++"):
    # Import a real C DSO with the same callback contract and owned lifetime.
    library = _export(_packed_source(name, callback), directory, language=language)
    return tvm_ffi.load_module(str(library), keep_module_alive=False)


@pytest.mark.parametrize("language", ["cc", "c++"])
def test_three_c_units_share_loader_context_and_packed_import(language, tmp_path):
    entries = [_packed_source(f"entry_{index}", "context31_sentinel", offset=index) for index in range(3)]
    for child in entries[1:]:
        entries[0].import_module(child)
    library = _export(entries[0], tmp_path / "three_units", language=language)
    sources = list((library.parent / "objects").glob("*.c"))
    core_sources = [path for path in sources if path.name != "devc.c"]
    assert len(core_sources) == 3
    for index in range(3):
        assert sum(f"__tvm_ffi_entry_{index}(" in path.read_text() for path in core_sources) == 1
    # LLVM emits the import table as an object; LLVM-OFF emits one C wrapper.
    import_sources = [path for path in sources if path.name == "devc.c"]
    assert len(import_sources) == (0 if runtime.enabled("llvm") else 1)
    if import_sources:
        assert "__tvm_ffi__library_bin" in import_sources[0].read_text()
    loaded = runtime.load_module(str(library))
    observed = []
    tvm_ffi.register_global_func("tl.testing.context31.three_units", observed.append, override=True)
    sentinel = _loaded_c_source("context31_sentinel", "tl.testing.context31.three_units", tmp_path / "sentinel", language=language)
    loaded.import_module(sentinel)
    for index in range(3):
        assert loaded[f"entry_{index}"](10) is None
    assert observed == [10, 11, 12]
    # The FFI loader must find and fill the exported canonical address.
    handle = ctypes.CDLL(str(library))
    assert ctypes.c_void_p.in_dll(handle, "__tvm_ffi__library_ctx").value is not None


@pytest.mark.parametrize("order", ["left,right", "right,left"])
@pytest.mark.parametrize("visibility", ["local", "global"])
def test_independent_c_libraries_keep_their_imports(order, visibility, tmp_path):
    libraries = {}
    for name in ("left", "right"):
        module = _packed_source("context31_entry_" + name, "context31_sentinel")
        libraries[name] = str(_export(module, tmp_path / name))
    command = [sys.executable, "-B", __file__, "--isolation-probe", order, visibility, json.dumps(libraries)]
    process = subprocess.run(command, env=dict(os.environ), text=True, capture_output=True)
    (tmp_path / "child.log").write_text(process.stdout + process.stderr)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "context31_independent_imports_passed" in process.stdout


def _isolation_probe(order, visibility, libraries):
    observed = []
    handles, modules = [], {}
    for name in order.split(","):
        if visibility == "global":
            handles.append(ctypes.CDLL(libraries[name], mode=ctypes.RTLD_GLOBAL))
        module = tvm_ffi.load_module(libraries[name], keep_module_alive=False)
        callback = "tl.testing.context31." + name
        tvm_ffi.register_global_func(callback, lambda value, name=name: observed.append([name, value]), override=True)
        module.import_module(_loaded_c_source("context31_sentinel", callback, Path(libraries[name]).parent / "sentinel"))
        modules[name] = module
    # Same imported function spelling resolves through each library's own context.
    assert modules["left"]["context31_entry_left"](31) is None
    assert modules["right"]["context31_entry_right"](47) is None
    assert observed == [["left", 31], ["right", 47]], observed
    functions = [modules[name]["context31_entry_" + name] for name in ("left", "right")]
    del module
    modules.clear()
    # Remove the extra dlopen references used only to request RTLD_GLOBAL.
    if handles:
        import _ctypes

        for handle in handles:
            _ctypes.dlclose(handle._handle)
        handles.clear()
    gc.collect()
    assert functions[0](53) is None
    assert functions[1](67) is None
    assert observed == [["left", 31], ["right", 47], ["left", 53], ["right", 67]]
    # Each returned packed Function retains its owning module and imports;
    # after the last Function is released the unpinned DSOs can unload.
    functions.clear()
    gc.collect()
    maps = Path("/proc/self/maps")
    if maps.exists():
        loaded_paths = maps.read_text()
        assert all(path not in loaded_paths for path in libraries.values())
    assert "tvm" not in sys.modules
    print("context31_independent_imports_passed", observed)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--isolation-probe":
    _isolation_probe(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]))
