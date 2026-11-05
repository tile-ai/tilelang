"""Persistent native JIT cache: real export, fresh-process load, and repair."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import tvm_tirx

import tilelang
import tilelang.cpu.language as T


def two_outputs(callee):
    n = T.dynamic("n", "int64")

    @T.prim_func
    def main(B: T.Tensor((n,), "float32"), A: T.Buffer((n,), "float32", align=4), C: T.Tensor((n,), "int32")):
        with T.Kernel(1):
            for i in T.Parallel(n):
                B[i] = A[i] + 1.0
                C[i] = T.int32(i) * 3

    return main.with_attr("tilelang_callee_allocated_outputs", callee)


def cache_process(mode, kind, callee, record_path):
    from tilelang.jit.kernel import JITKernel
    from tilelang._provider import FFI_REF, FFI_VERSION

    tilelang.enable_cache()
    original_compile = JITKernel._compile_and_create_adapter
    compilations = 0

    def observe_compile(self, *args, **kwargs):
        nonlocal compilations
        compilations += 1
        if mode == "hit":
            raise AssertionError("A fresh-process cache hit must not recompile the kernel")
        return original_compile(self, *args, **kwargs)

    JITKernel._compile_and_create_adapter = observe_compile
    kernel = tilelang.compile(
        two_outputs(callee),
        out_idx=[2, 0],
        target=tvm_tirx.target.Target({"kind": kind, **({"mcpu": "generic"} if kind == "llvm" else {})}),
        target_host=tvm_tirx.target.Target({"kind": kind, **({"mcpu": "generic"} if kind == "llvm" else {})}),
        execution_backend="tvm_ffi",
        verbose=True,
    )
    cache_path = Path(kernel._tilelang_cache_path)
    assert (cache_path / "executable.so").is_file(), list(cache_path.iterdir())
    assert (cache_path / "manifest.json").is_file()
    if mode == "hit":
        assert Path(kernel.adapter.libpath) == cache_path / "executable.so"
    assert compilations == (0 if mode == "hit" else 1)
    for extent in (0, 1, 17, 257):
        # The nonzero storage offset exercises the loaded ABI, not only shape.
        values = torch.arange(extent + 5, dtype=torch.float32)[5:]
        outputs = kernel(values)
        assert isinstance(outputs, list) and len(outputs) == 2
        torch.testing.assert_close(outputs[0], torch.arange(extent, dtype=torch.int32) * 3, rtol=0, atol=0)
        torch.testing.assert_close(outputs[1], values + 1.0, rtol=0, atol=0)
    assert "tvm" not in sys.modules
    mapped = []
    maps = Path("/proc/self/maps")
    if maps.is_file():
        mapped = sorted({line.split()[-1] for line in maps.read_text().splitlines() if "libtvm" in line})
        assert not any("/3rdparty/tvm/" in path for path in mapped)
    Path(record_path).write_text(
        json.dumps(
            {
                "mode": mode,
                "kind": kind,
                "callee": callee,
                "compilations": compilations,
                "key": kernel._tilelang_cache_key,
                "cache_path": str(cache_path),
                "ffi_ref": FFI_REF,
                "ffi_version": FFI_VERSION,
                "mapped_libraries": mapped,
                "extents": [0, 1, 17, 257],
                "status": "passed",
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("kind", ["llvm", "c"])
@pytest.mark.parametrize("callee", [False, True])
def test_persistent_cache_hit_and_corruption_repair(kind, callee, tmp_path):
    if kind == "llvm" and not tvm_tirx.runtime.enabled("llvm"):
        pytest.skip("Optional LLVM code generation is disabled")
    environment = os.environ.copy()
    environment.update(TILELANG_CACHE_DIR=str(tmp_path / "cache"), TILELANG_DISABLE_CACHE="0")

    def run(mode):
        record = tmp_path / f"{mode}.json"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), mode, kind, str(int(callee)), str(record)],
            env=environment,
            text=True,
            capture_output=True,
            timeout=180,
        )
        (tmp_path / f"{mode}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(record.read_text())
        assert payload["status"] == "passed"
        return payload

    original = run("populate")
    loaded = run("hit")
    assert loaded["key"] == original["key"]
    assert loaded["compilations"] == 0
    # Damage persisted metadata without updating its manifest. The actual
    # cache must detect the mismatch and rebuild before any descriptor is used.
    params = Path(original["cache_path"]) / "params.json"
    content = params.read_bytes()
    params.write_bytes(content[: len(content) // 2])
    repaired = run("repair")
    assert repaired["key"] == original["key"] and repaired["compilations"] == 1


if __name__ == "__main__":
    cache_process(sys.argv[1], sys.argv[2], bool(int(sys.argv[3])), sys.argv[4])
