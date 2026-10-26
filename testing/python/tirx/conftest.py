import ctypes
import importlib
import os
from pathlib import Path
import sys

import pytest
import tvm_ffi


@pytest.fixture
def compile_tilelang(native_tirx_passes):
    from tilelang.engine import lower
    from tvm_tirx.codegen.executable import Executable
    from tvm_tirx.target import Target

    def compile_function(function, target):
        target = target if isinstance(target, Target) else Target(target)
        artifact = lower(
            function,
            target=target,
            target_host=target.host if target.host is not None else target if target.kind.name in ("c", "llvm") else "c",
            enable_host_codegen=True,
            enable_device_compile=True,
        )
        return Executable(artifact.rt_mod)

    return compile_function


@pytest.fixture
def compile_lowered_cpu(native_tirx_passes):
    """Finish CPU pass-unit-test IR without repeating frontend transformations."""
    from tilelang._compat import ir
    from tilelang import transform
    from tilelang.transform import _tirx_passes
    from tvm_tirx import tirx
    from tvm_tirx.codegen.executable import Executable
    from tvm_tirx.target import Target

    def compile_function(function, target="c"):
        target = target if isinstance(target, Target) else Target(target)
        if target.kind.name not in ("c", "llvm"):
            raise ValueError("The lowered CPU fixture only supports C and LLVM")
        target = target.with_host(target.host if target.host is not None else target)
        if isinstance(function, tirx.Function):
            if not function.attrs or "global_symbol" not in function.attrs:
                function = function.with_attr("global_symbol", "main")
            module = ir.IRModule({"main": function})
        else:
            module = function
        passes = [
            _tirx_passes.BindTarget(target),
            _tirx_passes.LowerTensorLayouts(),
            _tirx_passes.FlattenTensor(),
            _tirx_passes.NarrowDataType(32),
            _tirx_passes.VectorizeLoop(False),
            _tirx_passes.StmtSimplify(),
            _tirx_passes.AnnotateEntryFunc(),
            transform.MakePackedAPI(),
            _tirx_passes.LowerTVMBuiltin(),
            transform.LowerIntrin(),
        ]
        for compiler_pass in passes:
            module = compiler_pass(module)
        assert ir.check_well_formed(module)
        builder = "tilelang_c_host" if target.kind.name == "c" else "tilelang_llvm"
        return Executable(tvm_ffi.get_global_func(f"target.build.{builder}")(module, target))

    return compile_function


@pytest.fixture
def tilelang_builders(monkeypatch, native_tirx_passes):
    compilation = importlib.import_module("tvm_tirx.tirx._compile")
    original = compilation._build_for_target

    def build(module, target):
        names = {
            "c": "tilelang_c_host",
            "cuda": "tilelang_cuda",
            "metal": "tilelang_metal",
            "webgpu": "tilelang_webgpu",
            "llvm": "tilelang_llvm",
        }
        if target.kind.name not in names:
            return original(module, target)
        return tvm_ffi.get_global_func(f"target.build.{names[target.kind.name]}")(module, target)

    monkeypatch.setattr(compilation, "_build_for_target", build)


@pytest.fixture(scope="session")
def native_tirx_passes():
    library = None
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    prefix = "" if sys.platform == "win32" else "lib"
    default = Path(__file__).resolve().parents[3] / "build" / "lib" / f"{prefix}tilelang_tirx_test_passes{suffix}"
    path = Path(os.environ.get("TILELANG_TIRX_TEST_LIBRARY", default))
    if tvm_ffi.get_global_func("tl.transform.AnnotateReadOnlyParams", allow_missing=True) is None:
        if not path.is_file():
            if "TILELANG_TIRX_TEST_LIBRARY" in os.environ:
                pytest.fail(f"TILELANG_TIRX_TEST_LIBRARY does not exist: {path}")
            pytest.skip("Build the tilelang_tirx_test_passes CMake target to test native TileLang passes")
        library = ctypes.CDLL(str(path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    import tilelang.transform._ffi_api
    import tilelang._compat.cuda_codegen
    import tilelang._compat.ws

    tvm_ffi.init_ffi_api("tl.transform", tilelang.transform._ffi_api.__name__)
    yield library


@pytest.fixture(scope="session")
def native_operator_helpers(native_tirx_passes):
    helper_library = None
    if tvm_ffi.get_global_func("tl.testing.CloneOperator", allow_missing=True) is None:
        suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
        prefix = "" if sys.platform == "win32" else "lib"
        default = Path(__file__).resolve().parents[3] / "build" / "lib" / f"{prefix}tilelang_tirx_test_passes{suffix}"
        path = Path(os.environ.get("TILELANG_TIRX_TEST_LIBRARY", default))
        helper_path = Path(
            os.environ.get("TILELANG_TIRX_TEST_HELPERS_LIBRARY", path.parent / f"{prefix}tilelang_tirx_test_helpers{suffix}")
        )
        if not helper_path.is_file():
            pytest.fail("Build the tilelang_tirx_test_helpers target alongside the full extension")
        helper_library = ctypes.CDLL(str(helper_path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    yield helper_library


@pytest.fixture(scope="session")
def native_layouts(native_tirx_passes):
    from tilelang import _ffi_api

    tvm_ffi.init_ffi_api("tl", _ffi_api.__name__)
    from tilelang import layout

    return layout


@pytest.fixture(scope="session")
def ascend_codegen(native_tirx_passes):
    library = None
    if tvm_ffi.get_global_func("target.build.tilelang_pto", allow_missing=True) is None:
        path = os.environ.get("TILELANG_TIRX_TEST_ASCEND_LIBRARY")
        if not path:
            pytest.skip("Requires the native Ascend CCE/PTO backend (USE_ASCEND=ON)")
        library = ctypes.CDLL(path, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    return library
