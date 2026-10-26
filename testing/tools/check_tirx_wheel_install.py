"""Check standalone installed wheels without importing either source tree."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def check_wheel_contents(directory: Path) -> None:
    tilelang_wheels = list(directory.glob("tilelang-*.whl"))
    dependency_wheels = list(directory.glob("apache_tvm_tirx-*.whl"))
    if not tilelang_wheels or not dependency_wheels:
        raise RuntimeError("Both TileLang and apache-tvm-tirx wheels are required")
    for path in tilelang_wheels + dependency_wheels:
        with zipfile.ZipFile(path) as wheel:
            names = wheel.namelist()
            legacy = [name for name in names if name.startswith("tvm/") or "3rdparty/tvm/" in name]
            if legacy:
                raise RuntimeError(f"Legacy TVM contents in {path.name}: {legacy}")
            if path in tilelang_wheels:
                duplicate = [name for name in names if Path(name).name.startswith(("libtvm_tirx", "libtvm_runtime", "libtvm_ffi"))]
                if duplicate:
                    raise RuntimeError(f"Bundled dependency libraries in {path.name}: {duplicate}")
                metadata_path = next(name for name in names if name.endswith(".dist-info/METADATA"))
                metadata = wheel.read(metadata_path).decode()
                if "Requires-Dist: apache-tvm-tirx" not in metadata:
                    raise RuntimeError(f"Missing standalone dependency in {path.name}")
        print(f"Wheel contents verified: {path.name}", flush=True)


def check_loaded_libraries(prefix: Path) -> set[Path]:
    mappings = Path("/proc/self/maps")
    if not mappings.is_file():
        return set()
    libraries = set()
    for line in mappings.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5]).resolve()
        if path.name.startswith(("libtilelang", "libtvm_tirx", "libtvm_runtime", "libtvm_ffi")):
            if not path.is_relative_to(prefix):
                raise RuntimeError(f"Compiler/runtime library loaded outside the installed environment: {path}")
            libraries.add(path)
    for library in sorted(libraries):
        print(f"Installed library verified: {library}", flush=True)
    return libraries


def check_symbol_ownership(libraries: set[Path]) -> None:
    native = {path for path in libraries if path.name.startswith(("libtvm_tirx", "libtvm_runtime", "libtvm_ffi"))}
    extensions = {path for path in libraries if path.name.startswith("libtilelang.so")}
    if sys.platform != "linux" or shutil.which("nm") is None:
        print("Strong-symbol ownership check unavailable: requires Linux and nm", flush=True)
        return
    if not extensions or not native:
        raise RuntimeError("Both TileLang and native dependency libraries must be loaded for the symbol check")

    def symbols(paths):
        result = set()
        for path in paths:
            output = subprocess.check_output(["nm", "-D", "--defined-only", "--format=posix", str(path)], text=True)
            for line in output.splitlines():
                fields = line.split()
                if len(fields) >= 2 and fields[1] in {"T", "D", "B", "R", "S"}:
                    result.add(fields[0])
        return result - {"__bss_start", "_edata", "_end"}

    overlap = symbols(extensions) & symbols(native)
    if overlap:
        raise RuntimeError(f"TileLang defines native dependency implementation symbols: {sorted(overlap)}")
    print("Strong implementation symbol overlap: 0", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheels", type=Path)
    parser.add_argument("--extra-site-packages", type=Path, action="append", default=[])
    parser.add_argument("--test-root", type=Path, default=Path(__file__).resolve().parents[1] / "python")
    arguments = parser.parse_args()
    if not sys.flags.isolated:
        raise RuntimeError("Run this checker with python -I")
    for variable in ("PYTHONPATH", "TVM_TIRX_LIBRARY_PATH", "TILELANG_TIRX_TEST_LIBRARY", "TILELANG_TIRX_TEST_HELPERS_LIBRARY"):
        if os.environ.get(variable):
            raise RuntimeError(f"Unset {variable} before checking installed wheels")
    if importlib.util.find_spec("tvm") is not None:
        raise RuntimeError("The isolated environment must not contain a legacy tvm package")
    if arguments.wheels:
        check_wheel_contents(arguments.wheels.resolve())
    for path in arguments.extra_site_packages:
        sys.path.append(str(path.resolve()))
    test_root = arguments.test_root.resolve()
    prefix = Path(sys.prefix).resolve()
    with tempfile.TemporaryDirectory(prefix="tilelang-tirx-wheel-") as directory:
        os.chdir(directory)
        os.environ["TILELANG_CACHE_DIR"] = str(Path(directory) / "cache")
        os.environ["TILELANG_DISABLE_CACHE"] = "0"
        native_functions = {}
        native_types = None
        for name in ("tvm_ffi", "tvm_tirx", "tilelang"):
            package = importlib.import_module(name)
            path = Path(package.__file__).resolve()
            if not path.is_relative_to(prefix):
                raise RuntimeError(f"{name} is not installed inside {prefix}: {path}")
            print(f"Installed package verified: {name}: {path}", flush=True)
            if name == "tvm_tirx":
                import tvm_ffi

                native_functions = {name: tvm_ffi.get_global_func(name) for name in tvm_ffi.registry.list_global_func_names()}
                native_types = (package.sym.Analyzer, package.ir.PrimType, package.ir.prim.Cast)
        check_loaded_libraries(prefix)
        from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

        for name, directory in (("TileLang templates", TILELANG_TEMPLATE_PATH), ("CUTLASS headers", CUTLASS_INCLUDE_DIR)):
            if not directory or not Path(directory).resolve().is_relative_to(prefix):
                raise RuntimeError(f"{name} must come from the installed environment: {directory}")
            print(f"Installed headers verified: {name}: {directory}", flush=True)
        import pytest

        result = pytest.main(
            [
                "-q",
                "-rs",
                str(test_root / "tirx" / "test_tirx_frontend.py"),
                str(test_root / "tirx" / "codegen" / "cuda" / "test_thread_context.py"),
                str(test_root / "tirx" / "codegen" / "cuda" / "test_storage_sync.py"),
                str(test_root / "tirx" / "sym" / "test_let_queries.py"),
                str(test_root / "tirx" / "sym" / "test_query_adapters.py"),
                str(test_root / "tirx" / "test_tirx_safe_memory.py") + "::test_annotated_pointer_keeps_its_actual_offset",
                str(test_root / "tirx" / "test_tirx_storage.py") + "::test_storage_metadata_keeps_its_logical_element_units",
                str(test_root / "tirx" / "test_tirx_lower_tile_op.py") + "::test_async_copy_injection_preserves_zero_fill_and_visibility",
                str(test_root / "tirx" / "test_tirx_cuda_host_services.py") + "::test_im2col_tensor_map_host_service_executes",
                str(test_root / "tirx" / "test_tirx_cuda_sync_passes.py")
                + "::test_blackwell_two_cta_preserves_native_typed_call_attributes",
                str(test_root / "cache"),
                "--tb=short",
                "--show-capture=no",
            ]
        )
        libraries = check_loaded_libraries(prefix)
        if libraries:
            check_symbol_ownership(libraries)
        replaced = [name for name, function in native_functions.items() if not function.same_as(tvm_ffi.get_global_func(name))]
        if replaced:
            raise RuntimeError(f"TileLang replaced native FFI registrations: {replaced}")
        native_package = sys.modules["tvm_tirx"]
        current_types = (native_package.sym.Analyzer, native_package.ir.PrimType, native_package.ir.prim.Cast)
        if any(current is not original for current, original in zip(current_types, native_types, strict=True)):
            raise RuntimeError("TileLang replaced native Python IR/analysis classes")
        print("Native FFI registrations and Python types unchanged", flush=True)
        if "tvm" in sys.modules:
            raise RuntimeError("A legacy tvm module was imported during installed-wheel validation")
        os.chdir(prefix)
        return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
