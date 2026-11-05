#!/usr/bin/env python3
"""Verify a paired installation in a fresh process, outside the source checkout.

Run once for each --import-order. --execute-cpu additionally runs the public
TileLang DSL/JIT through the C backend and checks exact results using Torch.
The default checks package ownership, registration and actual library loading.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys


def _loaded_paths():
    if sys.platform.startswith("linux"):
        paths = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/"):
                # The pathname is one field even when it contains whitespace.
                # Retain a kernel " (deleted)" suffix: such a mapping must not
                # authenticate a replacement file at the original pathname.
                paths.add(Path(fields[5]).resolve())
        return paths
    if sys.platform == "darwin":
        loader = ctypes.CDLL(None)
        loader._dyld_image_count.restype = ctypes.c_uint32
        loader._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        loader._dyld_get_image_name.restype = ctypes.c_char_p
        return {Path(loader._dyld_get_image_name(i).decode()).resolve() for i in range(loader._dyld_image_count())}
    if sys.platform == "win32":
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        process = kernel.GetCurrentProcess()
        enum = kernel.K32EnumProcessModules
        enum.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        get_name = kernel.GetModuleFileNameW
        get_name.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        count = 256
        while True:
            modules = (wintypes.HMODULE * count)()
            needed = wintypes.DWORD()
            if not enum(process, modules, ctypes.sizeof(modules), ctypes.byref(needed)):
                raise ctypes.WinError(ctypes.get_last_error())
            if needed.value <= ctypes.sizeof(modules):
                break
            count = needed.value // ctypes.sizeof(wintypes.HMODULE)
        paths = set()
        for handle in modules[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
            buffer = ctypes.create_unicode_buffer(32768)
            if not get_name(handle, buffer, len(buffer)):
                raise ctypes.WinError(ctypes.get_last_error())
            paths.add(Path(buffer.value).resolve())
        return paths
    raise RuntimeError(f"Library mapping verification is not implemented on {sys.platform}")


def _owned_files(distribution):
    return {Path(distribution.locate_file(item)).resolve() for item in distribution.files or ()}


def _verify_record(distribution):
    """Match installed bytes to their own wheel, including each FFI ABI variant."""
    files = distribution.files
    if not files:
        raise RuntimeError(f"Missing installed wheel RECORD for {distribution.metadata['Name']}")
    checked = 0
    for item in files:
        # Wheel RECORD cannot hash itself. Installers may also list generated
        # bytecode without a hash; it is not a compiler or header artifact.
        if item.hash is None:
            if item.suffix == ".pyc" or item.name in {"RECORD", "RECORD.jws", "RECORD.p7s"}:
                continue
            raise RuntimeError(f"Unhashed installed artifact: {distribution.metadata['Name']} {item}")
        path = Path(distribution.locate_file(item))
        if not path.is_file():
            raise RuntimeError(f"Missing installed artifact: {distribution.metadata['Name']} {item}")
        data = path.read_bytes()
        checksum = base64.urlsafe_b64encode(hashlib.new(item.hash.mode, data).digest()).rstrip(b"=").decode("ascii")
        if (item.size is not None and len(data) != item.size) or checksum != item.hash.value:
            raise RuntimeError(f"Installed artifact differs from wheel RECORD: {distribution.metadata['Name']} {item}")
        checked += 1
    return {"checked_files": checked, "version": distribution.version}


def _execute_cpu():
    import tilelang
    import tilelang.cpu.language as T
    import torch
    from tvm_tirx.target import Target

    n = T.dynamic("n", "int64")

    @T.prim_func
    def add_one(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            for i in T.Parallel(n):
                B[i] = A[i] + 1.0

    target = Target("c")
    kernel = tilelang.compile(add_one, out_idx=[1], target=target, target_host=target, execution_backend="tvm_ffi")
    for size in (0, 1, 17, 257):
        inputs = torch.arange(size, dtype=torch.float32)
        outputs = kernel(inputs)
        torch.testing.assert_close(outputs, inputs + 1, rtol=0, atol=0)
        assert outputs.shape == inputs.shape
    return {"sizes": [0, 1, 17, 257], "dtype": "float32", "rtol": 0, "atol": 0, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-order", choices=["tilelang-first", "tirx-first"], default="tilelang-first")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source", type=Path, help="native-source.json or a staging manifest")
    parser.add_argument("--allow-editable", action="store_true")
    parser.add_argument("--execute-cpu", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new verification output: {args.output}")

    # Do this before loading compiler code: path ownership and version strings
    # alone do not detect a replaced library or headers. Each distribution's
    # RECORD naturally admits the bytes of its own CPython/platform variant.
    distributions = {name: metadata.distribution(name) for name in ("tilelang", "apache-tvm-tirx", "apache-tvm-ffi")}
    integrity = {name: _verify_record(dist) for name, dist in distributions.items()}
    owners = {name: _owned_files(dist) for name, dist in distributions.items()}
    assert not owners["tilelang"] & owners["apache-tvm-tirx"], "Both wheels own the same path"
    assert not owners["tilelang"] & owners["apache-tvm-ffi"], "TileLang duplicates FFI files"
    assert not owners["apache-tvm-tirx"] & owners["apache-tvm-ffi"], "The compiler provider duplicates FFI files"

    if args.import_order == "tirx-first":
        import tvm_tirx
        import tilelang
    else:
        import tilelang
        import tvm_tirx
    import tvm_ffi
    from tvm_ffi import libinfo as ffi_libinfo
    from tilelang._build_info import COMPILER_INFO
    from tvm_tirx import base

    assert "tvm" not in sys.modules, "A legacy top-level TVM package was imported"
    assert not hasattr(tilelang, "tvm")
    assert not any(name == "tilelang.tvm" or name.startswith("tilelang.tvm.") for name in sys.modules)
    assert tilelang._LIB is tvm_tirx.base._LOADED_LIBS["tvm_tirx_compiler"]
    assert not base._RUNTIME_ONLY
    native_info = dict(tvm_ffi.get_global_func("tl.CompilerBuildInfo")())
    for key in ("source_digest", "provider_version", "ffi_ref"):
        assert native_info[key] == COMPILER_INFO[key], (key, native_info, COMPILER_INFO)
    if args.expected_source:
        expected = json.loads(args.expected_source.read_text())
        expected = expected.get("identity", expected)
        for key in ("source_digest", "provider_version", "ffi_ref", "ffi_version"):
            assert COMPILER_INFO[key] == expected[key], (key, COMPILER_INFO, expected)

    assert distributions["apache-tvm-tirx"].version == COMPILER_INFO["provider_version"]
    assert distributions["apache-tvm-ffi"].version == COMPILER_INFO["ffi_version"]
    assert Path(tvm_tirx.__file__).resolve() in owners["apache-tvm-tirx"]
    assert Path(tvm_ffi.__file__).resolve() in owners["apache-tvm-ffi"]
    if not args.allow_editable:
        assert Path(tilelang.__file__).resolve() in owners["tilelang"]

    libraries = {name: Path(lib._name).resolve() for name, lib in base._LOADED_LIBS.items()}
    for path in libraries.values():
        assert path in owners["apache-tvm-tirx"], ("Library not owned by the provider", str(path))
    ffi_path = Path(ffi_libinfo.find_libtvm_ffi()).resolve()
    assert ffi_path in owners["apache-tvm-ffi"]
    cpu_result = _execute_cpu() if args.execute_cpu else None
    mapped = {path for path in _loaded_paths() if "tvm" in path.name.lower()}
    assert ffi_path in mapped and all(path in mapped for path in libraries.values())
    assert not {path for path in mapped if path.name.lower() in {"libtvm.so", "tvm.dll", "libtvm.dylib"}}
    for expected_path in [ffi_path, *libraries.values()]:
        matches = {path for path in mapped if path.name.lower() == expected_path.name.lower()}
        assert matches == {expected_path}, ("Multiple loaded copies", matches)
    assert "tvm" not in sys.modules
    result = {
        "status": "passed",
        "import_order": args.import_order,
        "python": sys.executable,
        "identity": COMPILER_INFO,
        "native_info": native_info,
        "cpu_execution": cpu_result,
        "distributions": {name: dist.version for name, dist in distributions.items()},
        "distribution_integrity": integrity,
        "python_modules": {"tilelang": tilelang.__file__, "tvm_tirx": tvm_tirx.__file__, "tvm_ffi": tvm_ffi.__file__},
        "loaded_libraries": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(mapped)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
