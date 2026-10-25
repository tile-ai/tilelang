"""Check standalone installed wheels without importing either source tree."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
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


def check_loaded_libraries(prefix: Path) -> None:
    mappings = Path("/proc/self/maps")
    if not mappings.is_file():
        return
    libraries = set()
    for line in mappings.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5]).resolve()
        if path.name.startswith(("libtilelang.so", "libtvm_tirx", "libtvm_runtime", "libtvm_ffi")):
            if not path.is_relative_to(prefix):
                raise RuntimeError(f"Compiler/runtime library loaded outside the installed environment: {path}")
            libraries.add(path)
    for library in sorted(libraries):
        print(f"Installed library verified: {library}", flush=True)


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
        for name in ("tilelang", "tvm_tirx", "tvm_ffi"):
            package = importlib.import_module(name)
            path = Path(package.__file__).resolve()
            if not path.is_relative_to(prefix):
                raise RuntimeError(f"{name} is not installed inside {prefix}: {path}")
            print(f"Installed package verified: {name}: {path}", flush=True)
        check_loaded_libraries(prefix)
        import pytest

        result = pytest.main(
            [
                "-q",
                str(test_root / "tirx" / "test_tirx_frontend.py"),
                str(test_root / "cache"),
                "--tb=short",
                "--show-capture=no",
            ]
        )
        check_loaded_libraries(prefix)
        if "tvm" in sys.modules:
            raise RuntimeError("A legacy tvm module was imported during installed-wheel validation")
        os.chdir(prefix)
        return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
