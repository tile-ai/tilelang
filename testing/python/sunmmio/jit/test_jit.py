import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter.sunmmio import (
    SunmmioKernelABI,
    SunmmioKernelAdapter,
    SunmmioKernelSuDeckAdapter,
    SunmmioRuntimeScalar,
    SunmmioSuDeckLibraryGenerator,
    SunmmioSunsimLibraryGenerator,
)
from tilelang.jit.adapter.sunmmio import libgen as sunmmio_libgen
from tilelang.jit.adapter.sunmmio.libgen import (
    NpuirTools,
    SunmmioKernelArtifact,
    SunmmioToolchain,
)
from tilelang.jit.adapter.sunmmio.host_wrapper import SunmmioSuDeckSourceWrapper
from tilelang.jit.execution_backend import allowed_backends_for_target, resolve_execution_backend
from tilelang.layout import make_row_major, make_zz_layout
from tilelang.utils.target import determine_target, target_is_sunmmio


def _load_sunmmio_elementwise_example():
    example_path = Path(__file__).resolve().parents[4] / "examples" / "sunmmio" / "elementwise" / "elementwise_add.py"
    spec = importlib.util.spec_from_file_location("tilelang_sunmmio_elementwise_add_example", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sunmmio_dynamic_elementwise_example():
    example_path = Path(__file__).resolve().parents[4] / "examples" / "sunmmio" / "elementwise" / "elementwise_add_dynamic.py"
    spec = importlib.util.spec_from_file_location("tilelang_sunmmio_dynamic_elementwise_add_example", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_sudeck_install_root(root: Path, mesh: str = "4x4") -> tuple[Path, Path, Path]:
    include_dir = root / "include"
    lib_dir = root / "lib"
    cmake_dir = lib_dir / "cmake" / "SuDeck"
    (include_dir / "sudeck").mkdir(parents=True)
    lib_dir.mkdir(parents=True)
    cmake_dir.mkdir(parents=True)
    (include_dir / "sudeck" / "context.h").write_text("// fake sudeck header\n", encoding="utf-8")
    (cmake_dir / "SuDeckConfig.cmake").write_text("# fake sudeck cmake package\n", encoding="utf-8")
    (cmake_dir / f"SuDeckTargets_{mesh}.cmake").write_text("# fake sudeck cmake target\n", encoding="utf-8")
    library_path = lib_dir / f"libsudeck_{mesh}.so"
    library_path.write_bytes(b"fake sudeck")
    return root, include_dir, library_path


def _load_sunmmio_dynamic_exp2_example():
    example_path = Path(__file__).resolve().parents[4] / "examples" / "sunmmio" / "elementwise" / "elementwise_exp2_dynamic.py"
    spec = importlib.util.spec_from_file_location("tilelang_sunmmio_dynamic_elementwise_exp2_example", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sunmmio_online_softmax_example():
    example_path = Path(__file__).resolve().parents[4] / "examples" / "sunmmio" / "softmax" / "online_softmax.py"
    spec = importlib.util.spec_from_file_location("tilelang_sunmmio_online_softmax_example", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_npuir_tools():
    try:
        return NpuirTools.resolve()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _require_sunmmio_toolchain():
    try:
        return SunmmioToolchain.resolve()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _require_sunmmio_codegen():
    build = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("Sunmmio SUVM codegen is not available in this build. Rebuild TileLang with USE_SUNMMIO=ON.")
    return build


def _runtime_scalar_sources(abi):
    return [(scalar.name, scalar.source_param_index, scalar.source_kind, scalar.source_dim) for scalar in abi.runtime_scalars]


def test_sunmmio_abi_reorders_public_args_to_device_order():
    abi = SunmmioKernelABI(
        kernel_name="main_kernel",
        public_arg_count=3,
        public_param_names=("X", "LSE", "Y"),
        device_param_names=("LSE", "X", "Y"),
        runtime_scalars=(),
        device_param_dtypes=("handle", "handle", "handle"),
    )

    assert abi.materialize_runtime_args(["x", "lse", "y"], lambda *_: None) == ["lse", "x", "y"]


def test_sunmmio_abi_reorders_public_args_before_hidden_scalars():
    class Marker:
        def __init__(self, shape):
            self.shape = shape

    abi = SunmmioKernelABI(
        kernel_name="main_kernel",
        public_arg_count=3,
        public_param_names=("X", "LSE", "Y"),
        device_param_names=("LSE", "X", "Y", "m"),
        runtime_scalars=(SunmmioRuntimeScalar("m", source_param_index=0, source_kind="shape", source_dim=0),),
        device_param_dtypes=("handle", "handle", "handle", "int32"),
    )
    x = Marker((512, 128))
    lse = Marker((512, 128))
    y = Marker((512, 128))

    runtime_args = abi.materialize_runtime_args(
        [x, lse, y],
        lambda marker, source_kind, dim_index: marker.shape[dim_index] if source_kind == "shape" else None,
    )

    assert runtime_args == [lse, x, y, 512]
    assert abi.materialize_runtime_args([x, lse, y, 512], lambda *_: None) == [lse, x, y, 512]


def test_sunmmio_abi_rejects_public_arg_count_mismatch():
    with pytest.raises(ValueError, match="public_arg_count"):
        SunmmioKernelABI(
            kernel_name="k",
            public_arg_count=3,
            public_param_names=("X", "Y"),
            device_param_names=("X", "Y"),
            runtime_scalars=(),
            device_param_dtypes=("handle", "handle"),
        )


def test_sunmmio_abi_rejects_duplicate_public_names():
    with pytest.raises(ValueError, match="duplicate public"):
        SunmmioKernelABI(
            kernel_name="k",
            public_arg_count=2,
            public_param_names=("X", "X"),
            device_param_names=("X", "X"),
            runtime_scalars=(),
            device_param_dtypes=("handle", "handle"),
        )


def test_sunmmio_abi_rejects_duplicate_scalar_names():
    with pytest.raises(ValueError, match="duplicate scalar"):
        SunmmioKernelABI(
            kernel_name="k",
            public_arg_count=1,
            public_param_names=("X",),
            device_param_names=("X", "m"),
            runtime_scalars=(
                SunmmioRuntimeScalar("m", source_param_index=0, source_kind="shape", source_dim=0),
                SunmmioRuntimeScalar("m", source_param_index=0, source_kind="shape", source_dim=1),
            ),
            device_param_dtypes=("handle", "int32", "int32"),
        )


def test_sunmmio_abi_rejects_public_scalar_name_collision():
    with pytest.raises(ValueError, match="collide"):
        SunmmioKernelABI(
            kernel_name="k",
            public_arg_count=1,
            public_param_names=("m",),
            device_param_names=("m",),
            runtime_scalars=(SunmmioRuntimeScalar("m", source_param_index=0, source_kind="shape", source_dim=0),),
            device_param_dtypes=("handle",),
        )


def test_sunmmio_abi_rejects_non_permutation_device_order():
    with pytest.raises(ValueError, match="not a permutation"):
        SunmmioKernelABI(
            kernel_name="k",
            public_arg_count=2,
            public_param_names=("X", "Y"),
            device_param_names=("X", "Z"),
            runtime_scalars=(),
            device_param_dtypes=("handle", "handle"),
        )


def test_sunmmio_runtime_scalar_rejects_partial_source():
    with pytest.raises(ValueError, match="partial source binding"):
        SunmmioRuntimeScalar("m", source_param_index=0, source_kind="shape")


def test_sunmmio_abi_from_json_dict_rejects_inconsistent_metadata():
    with pytest.raises(ValueError, match="not a permutation"):
        SunmmioKernelABI.from_json_dict(
            {
                "kernel_name": "k",
                "public_arg_count": 2,
                "public_param_names": ["X", "Y"],
                "device_param_names": ["X", "Z"],
                "device_param_dtypes": ["handle", "handle"],
                "runtime_scalars": [],
            }
        )


def test_sunmmio_abi_from_modules_preserves_public_and_device_orders():
    x_buffer = tvm.tir.decl_buffer((128, 128), "bfloat16", name="X")
    lse_buffer = tvm.tir.decl_buffer((128, 128), "float32", name="LSE")
    y_buffer = tvm.tir.decl_buffer((128, 128), "float32", name="Y")
    prim_func = tvm.tir.PrimFunc(
        [x_buffer.data, lse_buffer.data, y_buffer.data],
        tvm.tir.Evaluate(0),
        buffer_map={
            x_buffer.data: x_buffer,
            lse_buffer.data: lse_buffer,
            y_buffer.data: y_buffer,
        },
    ).with_attr("global_symbol", "main")
    params = [KernelParam.from_buffer(buffer) for buffer in (x_buffer, lse_buffer, y_buffer)]

    device_func = tvm.tir.PrimFunc(
        [
            tvm.tir.Var("LSE", "handle"),
            tvm.tir.Var("X", "handle"),
            tvm.tir.Var("Y", "handle"),
        ],
        tvm.tir.Evaluate(0),
    ).with_attr("tir.is_global_func", True)
    device_mod = tvm.IRModule({"main_kernel": device_func})

    abi = SunmmioKernelABI.from_modules(
        func_or_mod=prim_func,
        host_mod=None,
        device_mod=device_mod,
        params=params,
    )

    assert abi.kernel_name == "main_kernel"
    assert abi.public_param_names == ("X", "LSE", "Y")
    assert abi.device_param_names == ("LSE", "X", "Y")
    assert abi.device_param_dtypes == ("handle", "handle", "handle")
    assert abi.materialize_runtime_args(["x", "lse", "y"], lambda *_: None) == ["lse", "x", "y"]


def _make_sunmmio_abi_with_host_scalar_expr(host_expr_builder):
    m = tvm.tir.Var("m", "int32")
    x_buffer = tvm.tir.decl_buffer((m,), "float32", name="X")
    prim_func = tvm.tir.PrimFunc(
        [x_buffer.data],
        tvm.tir.Evaluate(0),
        buffer_map={x_buffer.data: x_buffer},
    ).with_attr("global_symbol", "main")
    params = [KernelParam.from_buffer(x_buffer)]

    device_func = tvm.tir.PrimFunc(
        [tvm.tir.Var("X", "handle"), tvm.tir.Var("m", "int32")],
        tvm.tir.Evaluate(0),
    ).with_attr("tir.is_global_func", True)
    device_mod = tvm.IRModule({"main_kernel": device_func})

    host_call = tvm.tir.call_extern("int32", "main_kernel", x_buffer.data, host_expr_builder(m))
    host_func = tvm.tir.PrimFunc([x_buffer.data], tvm.tir.Evaluate(host_call))
    host_mod = tvm.IRModule({"host": host_func})

    return SunmmioKernelABI.from_modules(
        func_or_mod=prim_func,
        host_mod=host_mod,
        device_mod=device_mod,
        params=params,
    )


def test_sunmmio_abi_binds_identity_cast_scalar_expr():
    abi = _make_sunmmio_abi_with_host_scalar_expr(lambda m: tvm.tir.Cast("int32", m))

    class Marker:
        shape = (129,)

    marker = Marker()

    assert _runtime_scalar_sources(abi) == [("m", 0, "shape", 0)]
    assert abi.materialize_runtime_args(
        [marker],
        lambda value, source_kind, dim_index: value.shape[dim_index] if source_kind == "shape" else None,
    ) == [marker, 129]


def test_sunmmio_abi_leaves_non_identity_scalar_expr_unbound():
    abi = _make_sunmmio_abi_with_host_scalar_expr(lambda m: m + 1)

    class Marker:
        shape = (129,)

    marker = Marker()

    assert _runtime_scalar_sources(abi) == [("m", None, None, None)]
    with pytest.raises(ValueError, match="cannot be inferred"):
        abi.materialize_runtime_args(
            [marker],
            lambda value, source_kind, dim_index: value.shape[dim_index] if source_kind == "shape" else None,
        )
    assert abi.materialize_runtime_args([marker, 130], lambda *_: None) == [marker, 130]


def test_sunmmio_backend_resolution_accepts_lowercase_target():
    target = determine_target("sunmmio", return_object=True)

    assert target_is_sunmmio(target)
    assert allowed_backends_for_target(target) == ["sunmmio", "sunmmio_sunsim"]
    assert resolve_execution_backend(None, target) == "sunmmio"
    assert resolve_execution_backend("auto", target) == "sunmmio"
    assert resolve_execution_backend("Sunmmio", target) == "sunmmio"
    assert resolve_execution_backend("sunmmio_sunsim", target) == "sunmmio_sunsim"


def test_sunmmio_backends_share_cache_class():
    from tilelang.cache import _dispatch_map
    from tilelang.jit.adapter.sunmmio.kernel_cache import SunmmioKernelCache

    assert type(_dispatch_map["sunmmio"]) is SunmmioKernelCache
    assert type(_dispatch_map["sunmmio_sunsim"]) is SunmmioKernelCache


def test_sunmmio_toolchain_resolves_only_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv("SUNMMIO_TOOLCHAIN", raising=False)

    with pytest.raises(FileNotFoundError, match="SUNMMIO_TOOLCHAIN is not set"):
        SunmmioToolchain.resolve()

    toolchain_root = tmp_path / "toolchain"
    clangxx = toolchain_root / "clang" / "bin" / "clang++"
    clangxx.parent.mkdir(parents=True)
    clangxx.write_text("#!/bin/sh\n", encoding="utf-8")
    clangxx.chmod(0o755)
    monkeypatch.setenv("SUNMMIO_TOOLCHAIN", str(toolchain_root))

    toolchain = SunmmioToolchain.resolve()

    assert toolchain.clangxx == clangxx
    assert toolchain.cflags()[0] == "--target=riscv64-sunmmio-elf"
    assert all(not flag.startswith("--sysroot=") for flag in toolchain.cflags())


def test_sunmmio_adapter_resolves_target_before_libgen(tmp_path, monkeypatch):
    prim_func = tvm.tir.PrimFunc([], tvm.tir.Evaluate(0)).with_attr("global_symbol", "kernel")
    observed_targets = []

    original_init = SunmmioSuDeckLibraryGenerator.__init__

    def record_init(self, target, kernel_name, verbose=False):
        observed_targets.append(target)
        original_init(self, target, kernel_name, verbose)

    monkeypatch.setattr(SunmmioSuDeckLibraryGenerator, "__init__", record_init)
    monkeypatch.setattr(SunmmioSuDeckLibraryGenerator, "compile_lib", lambda self: None)
    monkeypatch.setattr(SunmmioSuDeckLibraryGenerator, "load_lib", lambda self, lib_path=None: None)
    device_mod = tvm.IRModule({"kernel": tvm.tir.PrimFunc([], tvm.tir.Evaluate(0)).with_attr("tir.is_global_func", True)})

    adapter = SunmmioKernelSuDeckAdapter(
        params=[],
        result_idx=[],
        target="llvm -mcpu=sunmmio-test -mattr=device_mesh_nrow_4,device_mesh_ncol_4",
        func_or_mod=prim_func,
        device_mod=device_mod,
        device_kernel_source="",
    )
    toolchain = SunmmioToolchain(root=tmp_path, clangxx=tmp_path / "clang++")
    cflags = adapter.lib_generator._compile_flags(toolchain)

    assert len(observed_targets) == 1
    assert target_is_sunmmio(observed_targets[0])
    assert str(observed_targets[0].attrs["mcpu"]) == "sunmmio-test"
    assert str(adapter.target.attrs["mcpu"]) == "sunmmio-test"
    assert [flag for flag in cflags if flag.startswith("-mcpu=")] == ["-mcpu=sunmmio-test"]
    assert "kernel" in adapter.lib_generator.device_tir_source


def test_sunmmio_cache_persists_artifacts_through_libgen(tmp_path):
    from tilelang.jit.adapter.sunmmio.kernel_cache import SunmmioKernelCache

    class Kernel:
        pass

    class Adapter:
        pass

    target = determine_target("sunmmio", return_object=True)

    kernel = Kernel()
    kernel.execution_backend = "sunmmio"
    kernel.adapter = Adapter()
    kernel.adapter.abi = SunmmioKernelABI(
        kernel_name="kernel",
        public_arg_count=0,
        public_param_names=(),
        device_param_names=(),
        runtime_scalars=(),
        device_param_dtypes=(),
    )
    kernel.adapter.lib_generator = SunmmioSuDeckLibraryGenerator(target, "kernel")
    sudeck_build_dir = tmp_path / "sudeck-build"
    sudeck_build_dir.mkdir()
    src_sudeck_elf = sudeck_build_dir / "kernel.elf"
    src_sudeck_mlir = sudeck_build_dir / "kernel.mlir"
    src_sudeck_tir = sudeck_build_dir / "kernel.tir"
    src_sudeck_ll = sudeck_build_dir / "kernel.ll"
    src_sudeck_elf.write_bytes(b"SUDECK_ELF")
    src_sudeck_mlir.write_text("module {}", encoding="utf-8")
    src_sudeck_tir.write_text("# device tir\n", encoding="utf-8")
    src_sudeck_ll.write_text("define void @kernel() { ret void }", encoding="utf-8")
    (sudeck_build_dir / "sudeck_launcher.so").write_bytes(b"LAUNCHER")
    (sudeck_build_dir / "sudeck_launch.py").write_text("def call(*args):\n    return None\n", encoding="utf-8")
    kernel.adapter.lib_generator.artifact = SunmmioKernelArtifact(
        elf_path=src_sudeck_elf,
        mlir_path=src_sudeck_mlir,
        llvm_ir_path=src_sudeck_ll,
        build_dir=sudeck_build_dir,
        runtime_kernel_name="kernel",
        tir_path=src_sudeck_tir,
    )

    sunmmio_cache_dir = tmp_path / "sunmmio"
    sunmmio_cache_dir.mkdir()
    SunmmioKernelCache()._save_so_cubin_to_disk(kernel, str(sunmmio_cache_dir))

    kernel_elf = sunmmio_cache_dir / "kernel.elf"
    kernel_mlir = sunmmio_cache_dir / "kernel.mlir"
    kernel_tir = sunmmio_cache_dir / "kernel.tir"
    kernel_ll = sunmmio_cache_dir / "kernel.ll"
    assert kernel_elf.read_bytes() == b"SUDECK_ELF"
    assert kernel_mlir.read_text(encoding="utf-8") == "module {}"
    assert kernel_tir.read_text(encoding="utf-8") == "# device tir\n"
    assert kernel_ll.read_text(encoding="utf-8") == "define void @kernel() { ret void }"
    assert json.loads((sunmmio_cache_dir / "abi.json").read_text(encoding="utf-8")) == kernel.adapter.abi.to_json_dict()
    assert (sunmmio_cache_dir / "sudeck_launcher.so").read_bytes() == b"LAUNCHER"
    assert (sunmmio_cache_dir / "sudeck_launch.py").exists()
    assert kernel.adapter.lib_generator.artifact.llvm_ir_path == src_sudeck_ll
    assert kernel.adapter.lib_generator.libpath is None

    sunsim_build_dir = tmp_path / "sunsim-build"
    sunsim_build_dir.mkdir()
    src_elf = sunsim_build_dir / "kernel.elf"
    src_mlir = sunsim_build_dir / "kernel.mlir"
    src_tir = sunsim_build_dir / "kernel.tir"
    src_ll = sunsim_build_dir / "kernel.ll"
    src_elf.write_bytes(b"ELF")
    src_mlir.write_text("module {}", encoding="utf-8")
    src_tir.write_text("# sunsim device tir\n", encoding="utf-8")
    src_ll.write_text("define void @kernel() { ret void }", encoding="utf-8")
    kernel.execution_backend = "sunmmio_sunsim"
    kernel.adapter.lib_generator = SunmmioSunsimLibraryGenerator(target, "kernel")
    kernel.adapter.lib_generator.artifact = SunmmioKernelArtifact(
        elf_path=src_elf,
        mlir_path=src_mlir,
        llvm_ir_path=src_ll,
        build_dir=sunsim_build_dir,
        runtime_kernel_name="kernel",
        tir_path=src_tir,
    )

    sunsim_cache_dir = tmp_path / "sunsim"
    sunsim_cache_dir.mkdir()
    SunmmioKernelCache()._save_so_cubin_to_disk(kernel, str(sunsim_cache_dir))

    assert (sunsim_cache_dir / "kernel.elf").read_bytes() == b"ELF"
    assert (sunsim_cache_dir / "kernel.mlir").read_text(encoding="utf-8") == "module {}"
    assert (sunsim_cache_dir / "kernel.tir").read_text(encoding="utf-8") == "# sunsim device tir\n"
    assert (sunsim_cache_dir / "kernel.ll").read_text(encoding="utf-8") == "define void @kernel() { ret void }"
    assert json.loads((sunsim_cache_dir / "abi.json").read_text(encoding="utf-8")) == kernel.adapter.abi.to_json_dict()
    artifact = kernel.adapter.lib_generator.artifact
    assert artifact.elf_path == src_elf
    assert artifact.llvm_ir_path == src_ll


def test_sunmmio_sudeck_dependency_uses_explicit_root(tmp_path, monkeypatch):
    root, include_dir, library_path = _make_sudeck_install_root(tmp_path / "sudeck", sunmmio_libgen.SUNMMIO_SUDECK_MESH)

    monkeypatch.setenv(sunmmio_libgen.SUNMMIO_SUDECK_ROOT_ENV, str(root))

    sudeck = sunmmio_libgen.SuDeckToolchain.resolve()

    assert sudeck.root == root.resolve()
    assert sudeck.library_path == library_path.resolve()
    assert sudeck.library_dir == library_path.parent.resolve()
    assert sudeck.cmake_component == "mesh_4x4"
    assert sudeck.cmake_target == "sudeck::sudeck_4x4"
    assert str(root.resolve()) in sudeck.cmake_prefix_path().split(";")
    assert include_dir.exists()


def test_sunmmio_sudeck_dependency_requires_explicit_source(monkeypatch):
    monkeypatch.delenv(sunmmio_libgen.SUNMMIO_SUDECK_ROOT_ENV, raising=False)

    with pytest.raises(FileNotFoundError, match=sunmmio_libgen.SUNMMIO_SUDECK_ROOT_ENV):
        sunmmio_libgen.SuDeckToolchain.resolve()


def test_sunmmio_toolchain_resolves_device_ld(tmp_path):
    device_ld = tmp_path / "sysroot" / "riscv64-unknown-elf" / "lib" / "sunmmio" / "sunmmio-a4e" / "device.ld"
    device_ld.parent.mkdir(parents=True)
    device_ld.write_text("SECTIONS {}\n", encoding="utf-8")
    toolchain = SunmmioToolchain(root=tmp_path, clangxx=tmp_path / "clang++")

    assert toolchain.resolve_device_ld("sunmmio-a4e") == device_ld


def test_sunmmio_sudeck_linker_script_preserves_runner_owned_symbols(tmp_path):
    device_ld = tmp_path / "sysroot" / "riscv64-unknown-elf" / "lib" / "sunmmio" / "sunmmio-a4e" / "device.ld"
    device_ld.parent.mkdir(parents=True)
    device_ld.write_text(
        """\
SECTIONS {
  .dtcm.scratch (NOLOAD) : { _dtcm_scratch_start = .; } > DTCM_SCRATCH
  .dtcm.tagtrace (NOLOAD) : { _dtcm_tagtrace_start = .; } > DTCM_TAGTRACE
  .stack (NOLOAD) : { _stack = .; } > STACK
}
PROVIDE(__stack_start = _stack);
PROVIDE(__stack_end = _stack_end);
""",
        encoding="utf-8",
    )
    toolchain = SunmmioToolchain(root=tmp_path, clangxx=tmp_path / "clang++")

    output = sunmmio_libgen._write_sudeck_linker_script(toolchain, "sunmmio-a4e", tmp_path / "sudeck.ld")
    text = output.read_text(encoding="utf-8")

    assert ".dtcm.scratch" not in text
    assert ".dtcm.tagtrace" not in text
    assert ".stack (NOLOAD)" not in text
    assert "PROVIDE(__stack_start = DTCM_STACK_START);" in text
    assert "PROVIDE(__stack_end = DTCM_STACK_START + DTCM_STACK_SIZE);" in text
    assert "PROVIDE(_dtcm_scratch_start = DTCM_SCRATCH_START);" in text
    assert "PROVIDE(_dtcm_scratch_end = DTCM_SCRATCH_START + DTCM_SCRATCH_SIZE);" in text
    assert "PROVIDE(_dtcm_tagtrace_start = DTCM_TAGTRACE_START);" in text
    assert "PROVIDE(_dtcm_tagtrace_end = DTCM_TAGTRACE_START + DTCM_TAGTRACE_SIZE);" in text


def test_sunmmio_npuir_compile_error_reports_build_dir(tmp_path, monkeypatch):
    target = determine_target("sunmmio", return_object=True)
    generator = SunmmioSunsimLibraryGenerator(target, "kernel")
    mlir_path = tmp_path / "kernel.mlir"
    llvm_path = tmp_path / "kernel.ll"
    mlir_path.write_text("module {}", encoding="utf-8")
    captured = {}

    def fake_run_command(command, **kwargs):
        captured["description"] = kwargs["description"]
        raise RuntimeError("npuir failed")

    monkeypatch.setattr(sunmmio_libgen, "_run_command", fake_run_command)

    with pytest.raises(RuntimeError, match="npuir failed"):
        generator._run_npuir_compile(NpuirTools(compile=tmp_path / "npuir-compile"), mlir_path, llvm_path)

    assert f"build_dir: {tmp_path}" in captured["description"]
    assert f"mlir: {mlir_path}" in captured["description"]


def test_sunmmio_load_lib_restores_optional_tir_path(tmp_path):
    target = determine_target("sunmmio", return_object=True)
    generator = SunmmioSunsimLibraryGenerator(target, "kernel")
    elf_path = tmp_path / "kernel.elf"
    mlir_path = tmp_path / "kernel.mlir"
    ll_path = tmp_path / "kernel.ll"
    tir_path = tmp_path / "kernel.tir"
    elf_path.write_bytes(b"ELF")
    mlir_path.write_text("module {}", encoding="utf-8")
    ll_path.write_text("define void @kernel() { ret void }", encoding="utf-8")
    tir_path.write_text("# device tir\n", encoding="utf-8")

    generator.load_lib(elf_path)

    assert generator.artifact.tir_path == tir_path


def test_sunmmio_sudeck_launcher_accepts_explicit_stream_handle():
    wrapper = SunmmioSuDeckSourceWrapper(
        params=[("X", "tensor"), ("n", "int32")],
        kernel_name="kernel",
        elf_name="kernel.elf",
        launcher_lib_name="sudeck_launcher.so",
    )

    launcher_cpp = wrapper.generate_launcher_cpp()
    launch_module = wrapper.generate_python_module()

    assert "void launch_kernel(int64_t a0, int32_t a1, int64_t stream_handle, tvm::ffi::Bytes elf, tvm::ffi::Bytes name)" in launcher_cpp
    assert "int64_t stream_handle" in launcher_cpp
    assert "default_stream" not in launcher_cpp
    assert "reinterpret_cast<const Stream *>" in launcher_cpp
    assert "Stream stream = *stream_ptr;" in launcher_cpp
    assert "import torch" not in launch_module
    assert "torch.sunmmio" not in launch_module
    assert "def call(a0, a1, stream_handle):" in launch_module
    assert "_load()(a0, a1, stream_handle, _ELF, _NAME)" in launch_module


def test_sunmmio_sudeck_launcher_formats_empty_device_arg_list():
    wrapper = SunmmioSuDeckSourceWrapper(
        params=[],
        kernel_name="kernel",
        elf_name="kernel.elf",
        launcher_lib_name="sudeck_launcher.so",
    )

    launcher_cpp = wrapper.generate_launcher_cpp()
    launch_module = wrapper.generate_python_module()

    assert "void launch_kernel(int64_t stream_handle, tvm::ffi::Bytes elf, tvm::ffi::Bytes name)" in launcher_cpp
    assert "def call(stream_handle):" in launch_module
    assert "_load()(stream_handle, _ELF, _NAME)" in launch_module


def test_sunmmio_sudeck_adapter_passes_torch_current_stream(monkeypatch):
    stream = object()
    torch_mod = types.ModuleType("torch")
    torch_mod.sunmmio = types.SimpleNamespace(
        current_stream=lambda: stream,
        unsafe_get_sudeck_stream_handle=lambda value: 0x1234 if value is stream else 0,
    )
    torch_sunmmio_mod = types.ModuleType("torch_sunmmio")
    torch_sunmmio_sunmmio_mod = types.ModuleType("torch_sunmmio.sunmmio")
    torch_sunmmio_sunmmio_mod.unsafe_get_sutensor_handle = lambda value: 0x5678
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch_sunmmio", torch_sunmmio_mod)
    monkeypatch.setitem(sys.modules, "torch_sunmmio.sunmmio", torch_sunmmio_sunmmio_mod)

    calls = []
    adapter = object.__new__(SunmmioKernelSuDeckAdapter)
    adapter.result_idx = []
    adapter.abi = SunmmioKernelABI(
        kernel_name="kernel",
        public_arg_count=1,
        public_param_names=("X",),
        device_param_names=("X",),
        runtime_scalars=(),
        device_param_dtypes=("handle",),
    )
    adapter.lib_generator = types.SimpleNamespace(pymodule=types.SimpleNamespace(call=lambda *args: calls.append(args)))

    func = SunmmioKernelSuDeckAdapter._convert_torch_func(adapter)
    func("tensor")

    assert calls == [(0x5678, 0x1234)]


def test_sunmmio_sudeck_compile_emits_kernel_elf_without_thunk(tmp_path, monkeypatch):
    target = determine_target("sunmmio", return_object=True)
    generator = SunmmioSuDeckLibraryGenerator(target, "kernel")
    generator.update_mlir_source("module {}")
    generator.update_device_tir_source("# device tir\n")
    llvm_source = "define void @kernel(ptr %0) #0 !sunmmio.kernel_meta !1 { ret void }"

    sudeck_root, _, sudeck_library_path = _make_sudeck_install_root(tmp_path / "sudeck", sunmmio_libgen.SUNMMIO_SUDECK_MESH)
    ffi_include = tmp_path / "ffi" / "include"
    ffi_lib_dir = tmp_path / "ffi" / "lib"
    ffi_library = ffi_lib_dir / "libtvm_ffi.so"
    ffi_include.mkdir(parents=True)
    ffi_lib_dir.mkdir(parents=True)
    ffi_library.write_bytes(b"fake tvm ffi")
    fake_toolchain = SunmmioToolchain(root=tmp_path, clangxx=tmp_path / "clang++")
    commands = []

    def fake_resolve():
        return fake_toolchain

    def fake_run_command(command, **kwargs):
        commands.append([str(part) for part in command])
        if command[0] == "cmake":
            if "--build" in command:
                (tmp_path / "sudeck_launcher.so").write_bytes(b"LAUNCHER")
        else:
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_bytes(b"ELF")

        class Result:
            stdout = ""

        return Result()

    def fake_mlir_to_llvm_ir(mlir_path, llvm_path):
        Path(llvm_path).write_text(llvm_source, encoding="utf-8")

    def fake_write_sudeck_linker_script(toolchain, mcpu, out_path):
        Path(out_path).write_text("SECTIONS {}\n", encoding="utf-8")
        return Path(out_path)

    monkeypatch.setenv(sunmmio_libgen.SUNMMIO_SUDECK_ROOT_ENV, str(sudeck_root))
    monkeypatch.setattr(sunmmio_libgen.SunmmioToolchain, "resolve", fake_resolve)
    monkeypatch.setattr(sunmmio_libgen, "_run_command", fake_run_command)
    monkeypatch.setattr(sunmmio_libgen, "_write_sudeck_linker_script", fake_write_sudeck_linker_script)
    monkeypatch.setattr(sunmmio_libgen, "_tvm_ffi_paths", lambda: (ffi_include, ffi_library))
    monkeypatch.setattr(generator, "_mlir_to_llvm_ir", fake_mlir_to_llvm_ir)

    generator.compile_lib(output_dir=tmp_path)

    assert (tmp_path / "kernel.ll").read_text(encoding="utf-8") == generator.artifact.llvm_ir_path.read_text(encoding="utf-8")
    assert (tmp_path / "kernel.tir").read_text(encoding="utf-8") == "# device tir\n"
    assert generator.artifact.tir_path == tmp_path / "kernel.tir"
    assert (tmp_path / "kernel.elf").read_bytes() == b"ELF"
    assert not (tmp_path / "main_thunk.cpp").exists()
    assert not (tmp_path / "main_thunk.o").exists()
    assert len(commands) == 4
    assert commands[0][:2] == [str(fake_toolchain.clangxx), "-c"]
    assert "--target=riscv64-sunmmio-elf" in commands[0]
    assert all(not flag.startswith("--sysroot=") for command in commands for flag in command)
    assert str(tmp_path / "kernel.ll") in commands[0]
    assert str(tmp_path / "kernel.o") in commands[0]
    assert commands[1][0] == str(fake_toolchain.clangxx)
    assert "-c" not in commands[1]
    assert str(tmp_path / "kernel.o") in commands[1]
    assert str(tmp_path / "kernel.elf") in commands[1]
    assert "-nostartfiles" in commands[1]
    assert "-lnosys" in commands[1]
    assert commands[2][:5] == ["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "sudeck_launcher_cmake")]
    assert f"-DCMAKE_PREFIX_PATH={sudeck_root.resolve()}" in commands[2]
    assert commands[3] == [
        "cmake",
        "--build",
        str(tmp_path / "sudeck_launcher_cmake"),
        "--target",
        "tilelang_sudeck_launcher",
        "--config",
        "Release",
    ]
    cmake_lists = (tmp_path / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "find_package(SuDeck REQUIRED COMPONENTS mesh_4x4)" in cmake_lists
    assert "sudeck::sudeck_4x4" in cmake_lists
    assert str(ffi_library) in cmake_lists
    assert str(sudeck_library_path.parent.resolve()) in cmake_lists
    assert "/usr/include/sunmmio" not in cmake_lists
    for command in commands:
        assert "main_thunk.o" not in command


@pytest.mark.sunmmio_toolchain
@pytest.mark.sunmmio_sunsim
@pytest.mark.sunmmio_closed_runtime
def test_sunmmio_sunsim_elementwise_add_example_executes(tmp_path):
    _require_sunmmio_codegen()
    _require_npuir_tools()
    _require_sunmmio_toolchain()

    pytest.importorskip("ml_dtypes")
    pytest.importorskip("sunsim")

    from tilelang.cache import _dispatch_map

    example = _load_sunmmio_elementwise_example()
    cache_root = tmp_path / "cache"
    tmp_root = tmp_path / "tmp"
    old_cache_dir = tilelang.env.TILELANG_CACHE_DIR
    old_tmp_dir = tilelang.env.TILELANG_TMP_DIR
    was_cache_enabled = tilelang.env.is_cache_enabled()

    try:
        tilelang.env.TILELANG_CACHE_DIR = str(cache_root)
        tilelang.env.TILELANG_TMP_DIR = str(tmp_root)
        tilelang.env.enable_cache()
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        example.elementwise_add._kernel_cache.clear()
        _dispatch_map["sunmmio_sunsim"]._memory_cache.clear()

        shapes = [(64, 64), (256, 64)]
        for shape in shapes:
            result = example.main(*shape)
            assert result.exit_code == 0

        elf_files = list(cache_root.glob("*/kernel.elf"))

        assert len(elf_files) == len(shapes)
    finally:
        example.elementwise_add._kernel_cache.clear()
        _dispatch_map["sunmmio_sunsim"]._memory_cache.clear()
        tilelang.env.TILELANG_CACHE_DIR = old_cache_dir
        tilelang.env.TILELANG_TMP_DIR = old_tmp_dir
        if was_cache_enabled:
            tilelang.env.enable_cache()
        else:
            tilelang.env.disable_cache()


@pytest.mark.sunmmio_toolchain
@pytest.mark.sunmmio_sunsim
@pytest.mark.sunmmio_closed_runtime
def test_sunmmio_sunsim_dynamic_elementwise_add_example_executes(tmp_path):
    _require_sunmmio_codegen()
    _require_npuir_tools()
    _require_sunmmio_toolchain()

    pytest.importorskip("ml_dtypes")
    pytest.importorskip("sunsim")

    from tilelang.cache import _dispatch_map

    example = _load_sunmmio_dynamic_elementwise_example()
    cache_root = tmp_path / "cache"
    tmp_root = tmp_path / "tmp"
    old_cache_dir = tilelang.env.TILELANG_CACHE_DIR
    old_tmp_dir = tilelang.env.TILELANG_TMP_DIR
    was_cache_enabled = tilelang.env.is_cache_enabled()

    try:
        tilelang.env.TILELANG_CACHE_DIR = str(cache_root)
        tilelang.env.TILELANG_TMP_DIR = str(tmp_root)
        tilelang.env.enable_cache()
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        example.elementwise_add_dynamic._kernel_cache.clear()
        _dispatch_map["sunmmio_sunsim"]._memory_cache.clear()

        shapes = [(64, 64), (96, 160), (129, 257)]
        for shape in shapes:
            result = example.main(*shape)
            assert result.exit_code == 0

        elf_files = list(cache_root.glob("*/kernel.elf"))

        assert len(elf_files) == 1
    finally:
        example.elementwise_add_dynamic._kernel_cache.clear()
        _dispatch_map["sunmmio_sunsim"]._memory_cache.clear()
        tilelang.env.TILELANG_CACHE_DIR = old_cache_dir
        tilelang.env.TILELANG_TMP_DIR = old_tmp_dir
        if was_cache_enabled:
            tilelang.env.enable_cache()
        else:
            tilelang.env.disable_cache()


@pytest.mark.sunmmio_toolchain
@pytest.mark.sunmmio_sunsim
@pytest.mark.sunmmio_closed_runtime
def test_sunmmio_sunsim_dynamic_elementwise_exp2_example_executes(tmp_path):
    _require_sunmmio_codegen()
    _require_npuir_tools()
    _require_sunmmio_toolchain()

    pytest.importorskip("sunsim")

    from tilelang.cache import _dispatch_map

    example = _load_sunmmio_dynamic_exp2_example()
    cache_root = tmp_path / "cache"
    tmp_root = tmp_path / "tmp"
    old_cache_dir = tilelang.env.TILELANG_CACHE_DIR
    old_tmp_dir = tilelang.env.TILELANG_TMP_DIR
    was_cache_enabled = tilelang.env.is_cache_enabled()

    try:
        tilelang.env.TILELANG_CACHE_DIR = str(cache_root)
        tilelang.env.TILELANG_TMP_DIR = str(tmp_root)
        tilelang.env.enable_cache()
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        example.elementwise_exp2_dynamic._kernel_cache.clear()
        _dispatch_map["sunmmio_sunsim"]._memory_cache.clear()

        shapes = [(64, 64), (96, 160), (129, 257)]
        for shape in shapes:
            result = example.main(*shape)
            assert result.exit_code == 0

        elf_files = list(cache_root.glob("*/kernel.elf"))

        assert len(elf_files) == 1
    finally:
        example.elementwise_exp2_dynamic._kernel_cache.clear()
        _dispatch_map["sunmmio_sunsim"]._memory_cache.clear()
        tilelang.env.TILELANG_CACHE_DIR = old_cache_dir
        tilelang.env.TILELANG_TMP_DIR = old_tmp_dir
        if was_cache_enabled:
            tilelang.env.enable_cache()
        else:
            tilelang.env.disable_cache()


def test_sunmmio_sunsim_dynamic_elementwise_add_jit_binds_shape_abi():
    _require_sunmmio_codegen()

    example = _load_sunmmio_dynamic_elementwise_example()
    prim_func = example.elementwise_add_dynamic.get_tir(
        block_M=32,
        block_N=32,
        in_dtype=T.bfloat16,
        out_dtype=T.float32,
    )
    artifact = tilelang.lower(
        prim_func,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )

    host_source = artifact.host_mod.script()
    device_source = artifact.kernel_source

    assert example.elementwise_add_dynamic.execution_backend == "sunmmio_sunsim"
    assert len(artifact.params) == 3
    assert all(not param.is_scalar() for param in artifact.params)
    assert 'with T.LetStmt(T.Cast("int32", elem_add_A_shape_1[0]), var=m)' in host_source
    assert 'with T.LetStmt(T.Cast("int32", elem_add_A_shape_1[1]), var=n)' in host_source
    assert 'T.call_extern("int32", "elem_add_kernel", A.data, B.data, C.data, m, n)' in host_source
    assert "func.func @elem_add_kernel" in device_source
    assert "%arg3: i32" in device_source
    assert "%arg4: i32" in device_source


def test_sunmmio_sunsim_dynamic_elementwise_exp2_jit_binds_shape_abi():
    _require_sunmmio_codegen()

    example = _load_sunmmio_dynamic_exp2_example()
    prim_func = example.elementwise_exp2_dynamic.get_tir(
        block_M=32,
        block_N=32,
        in_dtype=T.float32,
        out_dtype=T.float32,
    )
    artifact = tilelang.lower(
        prim_func,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )

    host_source = artifact.host_mod.script()
    device_source = artifact.kernel_source

    assert example.elementwise_exp2_dynamic.execution_backend == "sunmmio_sunsim"
    assert len(artifact.params) == 2
    assert all(not param.is_scalar() for param in artifact.params)
    assert 'with T.LetStmt(T.Cast("int32", elem_exp2_A_shape_1[0]), var=m)' in host_source
    assert 'with T.LetStmt(T.Cast("int32", elem_exp2_A_shape_1[1]), var=n)' in host_source
    assert 'T.call_extern("int32", "elem_exp2_kernel", A.data, B.data, m, n)' in host_source
    assert "func.func @elem_exp2_kernel" in device_source
    assert "%arg2: i32" in device_source
    assert "%arg3: i32" in device_source


def test_sunmmio_abi_extracts_dynamic_shape_bindings():
    _require_sunmmio_codegen()

    example = _load_sunmmio_dynamic_elementwise_example()
    prim_func = example.elementwise_add_dynamic.get_tir(
        block_M=32,
        block_N=32,
        in_dtype=T.bfloat16,
        out_dtype=T.float32,
    )
    artifact = tilelang.lower(
        prim_func,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )

    abi = SunmmioKernelABI.from_modules(
        func_or_mod=prim_func,
        host_mod=artifact.host_mod,
        device_mod=artifact.device_mod,
        params=artifact.params,
    )

    assert abi.kernel_name == "elem_add_kernel"
    assert abi.public_arg_count == 3
    assert abi.public_param_names == ("A", "B", "C")
    assert abi.runtime_scalar_names == ("m", "n")
    assert _runtime_scalar_sources(abi) == [
        ("m", 0, "shape", 0),
        ("n", 0, "shape", 1),
    ]

    class Marker:
        shape = (129, 257)

    args = [Marker(), Marker(), Marker()]
    runtime_args = abi.materialize_runtime_args(
        args,
        lambda marker, source_kind, dim_index: marker.shape[dim_index] if source_kind == "shape" else None,
    )

    assert runtime_args[:3] == args
    assert runtime_args[3:] == [129, 257]
    assert abi.materialize_runtime_args([*args, 129, 257], lambda *_: None) == [*args, 129, 257]


def test_sunmmio_abi_requires_lowered_device_module_for_dynamic_kernel():
    _require_sunmmio_codegen()

    example = _load_sunmmio_dynamic_elementwise_example()
    prim_func = example.elementwise_add_dynamic.get_tir(
        block_M=32,
        block_N=32,
        in_dtype=T.bfloat16,
        out_dtype=T.float32,
    )
    artifact = tilelang.lower(
        prim_func,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )

    with pytest.raises(ValueError, match="requires device_mod"):
        SunmmioKernelABI.from_modules(
            func_or_mod=prim_func,
            host_mod=None,
            device_mod=None,
            params=artifact.params,
        )


def test_sunmmio_sunsim_cache_restores_abi_kernel_name_from_metadata(tmp_path):
    from tilelang.jit.adapter.sunmmio.kernel_cache import SunmmioKernelCache

    example = _load_sunmmio_dynamic_elementwise_example()
    prim_func = example.elementwise_add_dynamic.get_tir(
        block_M=32,
        block_N=32,
        in_dtype=T.bfloat16,
        out_dtype=T.float32,
    )
    params = [KernelParam.from_buffer(prim_func.buffer_map[param]) for param in prim_func.params]

    elf_path = tmp_path / "kernel.elf"
    mlir_path = tmp_path / "kernel.mlir"
    ll_path = tmp_path / "kernel.ll"
    elf_path.write_bytes(b"ELF")
    mlir_path.write_text("module {}", encoding="utf-8")
    ll_path.write_text(
        "define void @elem_add_kernel(ptr %0, ptr %1, ptr %2, i32 %3, i32 %4) #0 !sunmmio.kernel_meta !1 { ret void }",
        encoding="utf-8",
    )
    cached_abi = SunmmioKernelABI(
        kernel_name="elem_add_kernel",
        public_arg_count=3,
        public_param_names=("A", "B", "C"),
        device_param_names=("B", "A", "C", "m", "n"),
        runtime_scalars=(
            SunmmioRuntimeScalar("m", source_param_index=0, source_kind="shape", source_dim=0),
            SunmmioRuntimeScalar("n", source_param_index=0, source_kind="shape", source_dim=1),
        ),
        device_param_dtypes=("handle", "handle", "handle", "int32", "int32"),
    )
    (tmp_path / "abi.json").write_text(json.dumps(cached_abi.to_json_dict()), encoding="utf-8")

    kernel = SunmmioKernelCache()._build_kernel(
        func=prim_func,
        host_kernel_source=None,
        device_kernel_source="module {}",
        kernel_lib_path=str(elf_path),
        kernel_params=params,
        target="sunmmio",
        target_host=None,
        out_idx=[],
        execution_backend="sunmmio_sunsim",
        pass_configs=None,
        compile_flags=None,
    )
    assert kernel is not None
    adapter = kernel.adapter

    assert adapter.abi.kernel_name == "elem_add_kernel"
    assert adapter.kernel_name == "elem_add_kernel"
    assert adapter.abi.public_param_names == ("A", "B", "C")
    assert adapter.abi.device_param_names == ("B", "A", "C", "m", "n")
    assert adapter.abi.runtime_scalar_names == ("m", "n")
    assert _runtime_scalar_sources(adapter.abi) == [
        ("m", 0, "shape", 0),
        ("n", 0, "shape", 1),
    ]


def test_sunmmio_elementwise_device_tir_is_tokenless():
    _require_sunmmio_codegen()

    static_example = _load_sunmmio_elementwise_example()
    static_prim = static_example.elementwise_add.get_tir(
        128,
        128,
        block_M=32,
        block_N=32,
        in_dtype=T.bfloat16,
        out_dtype=T.float32,
    )
    static_artifact = tilelang.lower(
        static_prim,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )
    static_source = static_artifact.device_mod.script()

    assert static_source.count("T.dma_copy") == 3
    assert "token" not in static_source
    assert static_source.count("T.odma_unit(") == 3

    static_multi_prim = static_example.elementwise_add.get_tir(
        128,
        256,
        block_M=32,
        block_N=32,
        in_dtype=T.bfloat16,
        out_dtype=T.float32,
    )
    static_multi_artifact = tilelang.lower(
        static_multi_prim,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )
    static_multi_source = static_multi_artifact.device_mod.script()

    assert static_multi_source.count("T.dma_copy") == 3
    assert "token" not in static_multi_source
    assert static_multi_source.count("T.odma_unit(") == 3

    dynamic_example = _load_sunmmio_dynamic_elementwise_example()
    dynamic_prim = dynamic_example.elementwise_add_dynamic.get_tir(
        block_M=32,
        block_N=32,
        in_dtype=T.bfloat16,
        out_dtype=T.float32,
    )
    dynamic_artifact = tilelang.lower(
        dynamic_prim,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )
    dynamic_source = dynamic_artifact.device_mod.script()

    assert dynamic_source.count("T.dma_copy") == 3
    assert "token" not in dynamic_source
    assert dynamic_source.count("T.odma_unit(") == 3


def test_sunmmio_softmax_device_tir_preserves_odma_units_without_tokens():
    _require_sunmmio_codegen()

    example = _load_sunmmio_online_softmax_example()
    prim = example.online_softmax.get_tir(
        2048,
        2048,
        block_M=256,
        block_N=128,
        dtype=T.bfloat16,
    )
    artifact = tilelang.lower(
        prim,
        target="sunmmio",
        enable_host_codegen=False,
        enable_device_compile=False,
    )
    lines = artifact.device_mod.script().splitlines()

    source = "\n".join(lines)
    assert "token" not in source

    output_dma_idx, output_dma_line = next((idx, line) for idx, line in enumerate(lines) if "T.dma_copy(T.region(Y_shared[0, 0]" in line)
    store_idx = max(idx for idx, line in enumerate(lines[:output_dma_idx]) if "Y_shared[" in line and "] =" in line)
    input_dma_idx, input_dma_line = max((idx, line) for idx, line in enumerate(lines[:store_idx]) if "T.dma_copy(T.region(X_" in line)
    tile_store_loop_idx = next(
        idx
        for idx, line in enumerate(lines[input_dma_idx:store_idx], start=input_dma_idx)
        if "for i in T.serial" in line and "tile.domain" in line
    )
    assert 'T.odma_unit("odma0")' in input_dma_line
    assert 'T.odma_unit("odma0")' in output_dma_line
    assert input_dma_idx < tile_store_loop_idx < store_idx < output_dma_idx


def test_sunmmio_base_adapter_does_not_expose_sunsim_runtime_surface(tmp_path):
    assert not hasattr(SunmmioKernelAdapter, "get_llvm_ir")
    assert not hasattr(SunmmioKernelAdapter, "lower_to_llvm_ir")
    assert not hasattr(SunmmioKernelAdapter, "libpath")
    assert not hasattr(SunmmioKernelAdapter, "llvm_ir_artifact")
    assert not hasattr(SunmmioKernelAdapter, "llvm_ir_source")
    assert not hasattr(SunmmioKernelAdapter, "llvm_ir_path")
    assert not hasattr(SunmmioKernelAdapter, "build_sunsim_elf")
    assert not hasattr(SunmmioKernelAdapter, "sunsim_artifact")
    assert not hasattr(SunmmioKernelAdapter, "sunsim_elf_path")
    assert not hasattr(SunmmioKernelAdapter, "sunsim_build_dir")

    target = determine_target("sunmmio", return_object=True)
    generator = SunmmioSunsimLibraryGenerator(target, "kernel")
    llvm_path = tmp_path / "kernel.ll"

    generator.artifact = SunmmioKernelArtifact(
        elf_path=tmp_path / "kernel.elf",
        mlir_path=tmp_path / "kernel.mlir",
        llvm_ir_path=llvm_path,
        build_dir=tmp_path,
        runtime_kernel_name="kernel",
    )

    assert generator.artifact.llvm_ir_path == llvm_path
    assert generator.get_lib_path() is None


def test_sunmmio_sunsim_thunk_uses_abi_kernel_name(tmp_path, monkeypatch):
    target = determine_target("sunmmio", return_object=True)
    generator = SunmmioSunsimLibraryGenerator(target, "elem_add_kernel")
    generator.update_mlir_source(
        """
module attributes {suvm.device_arch = #suvm.device_arch<a4e>} {
  func.func @elem_add_kernel() {
    return
  }
}
"""
    )
    llvm_source = """
define void @elem_add_kernel(ptr addrspace(4) %0) #0 !sunmmio.kernel_meta !1 {
  ret void
}
"""

    def fake_mlir_to_llvm_ir(mlir_path, llvm_path):
        Path(llvm_path).write_text(llvm_source, encoding="utf-8")

    fake_toolchain = SunmmioToolchain(root=tmp_path, clangxx=tmp_path / "clang++")
    commands = []

    def fake_resolve():
        return fake_toolchain

    def fake_run_command(command, **kwargs):
        commands.append([str(part) for part in command])
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_bytes(b"artifact")

        class Result:
            stdout = ""

        return Result()

    monkeypatch.setattr(sunmmio_libgen.SunmmioToolchain, "resolve", fake_resolve)
    monkeypatch.setattr(sunmmio_libgen, "_run_command", fake_run_command)
    monkeypatch.setattr(generator, "_mlir_to_llvm_ir", fake_mlir_to_llvm_ir)

    generator.compile_lib(output_dir=tmp_path)
    thunk = (tmp_path / "main_thunk.cpp").read_text(encoding="utf-8")

    assert generator.runtime_kernel_name == "elem_add_kernel"
    assert "void elem_add_kernel(void *args);" in thunk
    assert "elem_add_kernel(const_cast<unsigned char *>(_kernel_arg_start));" in thunk
    assert '#include "pwln_setup.h"' not in thunk
    assert 'section(".r.sram.common")' in thunk
    assert "static inline void pwln_init()" in thunk
    assert len(commands) == 3
    assert all("--target=riscv64-sunmmio-elf" in command for command in commands)
    assert all(not flag.startswith("--sysroot=") for command in commands for flag in command)


@tilelang.jit(target="sunmmio")
def elementwise_add_jit(M, N, block_M, block_N, in_dtype, out_dtype):
    """JIT version of examples/sunmmio/elementwise/elementwise_add.py."""

    zz_layout = make_zz_layout((M, N))
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def elem_add(
        A: T.MeshTensor((M, N), placement, in_dtype, layout=zz_layout),
        B: T.MeshTensor((M, N), placement, in_dtype, layout=zz_layout),
        C: T.MeshTensor((M, N), placement, out_dtype, layout=zz_layout),
    ):
        with T.Kernel() as _cid:
            sharded_M, sharded_N = A.local_shape

            A_shared = T.alloc_shared((block_M, block_N), in_dtype)
            B_shared = T.alloc_shared((block_M, block_N), in_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                for by in T.serial(T.ceildiv(sharded_N, block_N)):
                    T.copy(A[bx * block_M, by * block_N], A_shared)
                    T.copy(B[bx * block_M, by * block_N], B_shared)
                    for i, j in T.Tiles([block_M, block_N]):
                        C_shared[i, j] = A_shared[i, j] + B_shared[i, j]
                    T.copy(C_shared, C[bx * block_M, by * block_N])

    return elem_add


@pytest.mark.sunmmio_toolchain
@pytest.mark.sunmmio_closed_runtime
def test_sunmmio_jit_lowercase_target_generates_suvm_source():
    _require_sunmmio_codegen()
    _require_npuir_tools()
    _require_sunmmio_toolchain()

    kernel = elementwise_add_jit(64, 64, 32, 32, T.bfloat16, T.float32)
    source = kernel.get_kernel_source()

    assert kernel.execution_backend == "sunmmio"
    assert isinstance(kernel.adapter, SunmmioKernelSuDeckAdapter)
    assert target_is_sunmmio(kernel.target)
    assert "suvm.device_arch" in source
    assert "func.func @elem_add" in source


@tilelang.jit(target="sunmmio")
def row_major_copy_jit(M, N, dtype):
    rsram_layout = make_row_major((M, N))

    @T.prim_func
    def copy_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel():
            A_shared = T.alloc_shared((M, N), dtype, scope="shared.rsram")
            T.annotate_layout({A_shared: rsram_layout})

            T.copy(A[0, 0], A_shared)
            T.copy(A_shared, B[0, 0])

    return copy_kernel


@pytest.mark.sunmmio_toolchain
@pytest.mark.sunmmio_closed_runtime
def test_sunmmio_jit_row_major_copy_lowers_to_llvm_ir(tmp_path):
    _require_sunmmio_codegen()
    _require_npuir_tools()
    _require_sunmmio_toolchain()

    kernel = row_major_copy_jit(32, 64, T.float32)
    mlir_path = tmp_path / "row_major_copy_jit.mlir"
    llvm_path = tmp_path / "row_major_copy_jit.ll"

    kernel.adapter.lib_generator._dump_mlir(mlir_path)
    kernel.adapter.lib_generator._mlir_to_llvm_ir(mlir_path, llvm_path)
    llvm_ir = llvm_path.read_text(encoding="utf-8")

    assert llvm_path.read_text(encoding="utf-8") == llvm_ir
    assert "suvm.device_arch" in kernel.get_kernel_source()
    assert "define void @copy_kernel" in llvm_ir
    assert "sunmmio.kernel_meta" in llvm_ir
    assert "su_odma_submit_direct" in llvm_ir


@pytest.mark.sunmmio_toolchain
@pytest.mark.sunmmio_closed_runtime
def test_sunmmio_jit_cache_saves_llvm_ir(tmp_path, monkeypatch):
    _require_sunmmio_codegen()
    _require_npuir_tools()
    _require_sunmmio_toolchain()

    from tilelang.cache import _dispatch_map

    cache_root = tmp_path / "cache"
    tmp_root = tmp_path / "tmp"
    old_cache_dir = tilelang.env.TILELANG_CACHE_DIR
    old_tmp_dir = tilelang.env.TILELANG_TMP_DIR
    was_cache_enabled = tilelang.env.is_cache_enabled()

    try:
        tilelang.env.TILELANG_CACHE_DIR = str(cache_root)
        tilelang.env.TILELANG_TMP_DIR = str(tmp_root)
        tilelang.env.enable_cache()
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        row_major_copy_jit._kernel_cache.clear()
        _dispatch_map["sunmmio"]._memory_cache.clear()

        kernel = row_major_copy_jit(32, 64, T.float32)
        elf_files = sorted(cache_root.glob("*/kernel.elf"))
        ll_files = sorted(cache_root.glob("*/kernel.ll"))

        assert len(elf_files) == 1
        assert len(ll_files) == 1
        assert elf_files[0].read_bytes()
        llvm_ir = ll_files[0].read_text(encoding="utf-8")
        artifact = kernel.adapter.lib_generator.artifact
        assert artifact is not None
        assert artifact.llvm_ir_path.read_text(encoding="utf-8") == llvm_ir
        assert artifact.elf_path.read_bytes() == elf_files[0].read_bytes()
        assert "define void @copy_kernel" in llvm_ir
        assert "sunmmio.kernel_meta" in llvm_ir

        def fail_resolve(*args, **kwargs):
            raise AssertionError("cached Sunmmio LLVM IR should not require NPU-IR tool resolution")

        monkeypatch.setattr(NpuirTools, "resolve", fail_resolve)
        row_major_copy_jit._kernel_cache.clear()
        _dispatch_map["sunmmio"]._memory_cache.clear()
        cached_kernel = row_major_copy_jit(32, 64, T.float32)

        cached_artifact = cached_kernel.adapter.lib_generator.artifact
        assert cached_artifact is not None
        assert cached_artifact.llvm_ir_path == ll_files[0]
        assert cached_artifact.llvm_ir_path.read_text(encoding="utf-8") == llvm_ir
        assert cached_artifact.elf_path == elf_files[0]
    finally:
        row_major_copy_jit._kernel_cache.clear()
        _dispatch_map["sunmmio"]._memory_cache.clear()
        tilelang.env.TILELANG_CACHE_DIR = old_cache_dir
        tilelang.env.TILELANG_TMP_DIR = old_tmp_dir
        if was_cache_enabled:
            tilelang.env.enable_cache()
        else:
            tilelang.env.disable_cache()


if __name__ == "__main__":
    tilelang.testing.main()
