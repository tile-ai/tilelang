"""Check the CUDA runtime stub in a fresh process without importing a compiler.

The CLI form accepts a built or installed libstub_cudart.so.  The pytest entry
uses the CUDA sidecar selected by the current provider, then starts that CLI in
an isolated child.  A real CUDA runtime must be discoverable by the loader.
"""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def _mapped_libraries():
    result = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            result.add(str(Path(fields[5]).resolve()))
    return sorted(result)


def _symbol_owner(symbol):
    class DlInfo(ctypes.Structure):
        _fields_ = [
            ("filename", ctypes.c_char_p),
            ("base", ctypes.c_void_p),
            ("symbol_name", ctypes.c_char_p),
            ("symbol_address", ctypes.c_void_p),
        ]

    api = ctypes.CDLL(None)
    api.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    api.dladdr.restype = ctypes.c_int
    info = DlInfo()
    assert api.dladdr(ctypes.cast(symbol, ctypes.c_void_p), ctypes.byref(info))
    return str(Path(os.fsdecode(info.filename)).resolve())


def probe(stub_path):
    """Exercise real forwarding before opening libcudart directly for comparison."""
    assert sys.platform.startswith("linux"), "This probe inspects Linux loader mappings"
    assert not {"torch", "tilelang", "tvm_tirx", "tvm_ffi"}.intersection(sys.modules)
    stub_path = Path(stub_path).resolve()
    before = _mapped_libraries()
    assert not any(Path(path).name.startswith("libcudart.so") for path in before)
    library = ctypes.CDLL(str(stub_path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    loaded = _mapped_libraries()
    assert not any(Path(path).name.startswith("libcudart.so") for path in loaded)
    forwarded = library.cudaGetErrorName
    forwarded.argtypes = [ctypes.c_int]
    forwarded.restype = ctypes.c_char_p
    assert _symbol_owner(forwarded) == str(stub_path)

    # Error 1 distinguishes real forwarding from the stub's Unknown fallback.
    expected = {0: "cudaSuccess", 1: "cudaErrorInvalidValue"}
    actual = {code: forwarded(code).decode() for code in expected}
    assert actual == expected, actual
    after = _mapped_libraries()
    runtime_paths = [path for path in after if Path(path).name.startswith("libcudart.so")]
    assert len(runtime_paths) == 1, runtime_paths
    runtime = ctypes.CDLL(runtime_paths[0], mode=os.RTLD_NOW | os.RTLD_LOCAL)
    direct = runtime.cudaGetErrorName
    direct.argtypes = [ctypes.c_int]
    direct.restype = ctypes.c_char_p
    assert _symbol_owner(direct) == runtime_paths[0]
    reference = {code: direct(code).decode() for code in expected}
    assert actual == reference
    ffi_paths = [path for path in after if Path(path).name.startswith("libtvm_ffi.so")]
    assert len(ffi_paths) <= 1, ffi_paths
    assert not {"torch", "tilelang", "tvm_tirx", "tvm_ffi"}.intersection(sys.modules)
    return {
        "status": "passed",
        "cases": len(expected),
        "expected": expected,
        "forwarded": actual,
        "direct": reference,
        "stub": str(stub_path),
        "stub_sha256": hashlib.sha256(stub_path.read_bytes()).hexdigest(),
        "real_cudart": runtime_paths[0],
        "real_cudart_sha256": hashlib.sha256(Path(runtime_paths[0]).read_bytes()).hexdigest(),
        "runtime_absent_before_first_call": True,
        "runtime_loaded_by_stub_call": True,
        "ffi_libraries": ffi_paths,
        "new_mappings": sorted(set(after) - set(before)),
        "gpu_execution": False,
    }


def test_cudart_error_name_forwarding(tmp_path):
    import pytest

    if not sys.platform.startswith("linux"):
        pytest.skip("This loader-mapping oracle is for Linux")
    from tvm_tirx import base

    runtime = base._LOADED_LIBS.get("tvm_runtime_cuda")
    if runtime is None:
        pytest.skip("The selected provider has no CUDA runtime sidecar")
    stub = Path(runtime._name).resolve().parent / "libstub_cudart.so"
    if not stub.is_file():
        pytest.skip("The selected CUDA provider does not use the runtime stub")
    output = tmp_path / "forwarding.json"
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "LD_PRELOAD"):
        environment.pop(name, None)
    subprocess.run(
        [sys.executable, "-I", "-B", __file__, str(stub), str(output)],
        env=environment,
        check=True,
    )
    assert json.loads(output.read_text())["status"] == "passed"


if __name__ == "__main__":
    output = Path(sys.argv[2])
    if output.exists():
        raise FileExistsError(output)
    result = probe(sys.argv[1])
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
