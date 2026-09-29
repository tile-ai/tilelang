"""Kernel caching for the torch Metal backend.

Compiled Metal kernels are cached like every other backend: a memory hit
returns the same kernel, a disk hit in the same or a fresh process recreates
the adapter from the cached source and launch metadata without lowering
again, damaged entries are repaired by recompiling, and compiling a Metal
kernel no longer disables caching for the rest of the process.
"""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.cache import _dispatch_map
from tilelang.env import env
from tilelang.jit.kernel import JITKernel

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="PyTorch MPS device is required")

SIZE = 96


def affine(symbol: str):
    @T.prim_func
    def main(A: T.Tensor((SIZE,), "float32"), B: T.Tensor((SIZE,), "float32")):
        with T.Kernel(T.ceildiv(SIZE, 32), threads=32) as block:
            i = block * 32 + T.get_thread_binding(0)
            if i < SIZE:
                B[i] = A[i] * 2 + 1

    return main.with_attr("global_symbol", symbol)


def compile_metal(func):
    return tilelang.compile(func, target="metal", execution_backend="torch")


def run(kernel):
    source = torch.arange(SIZE, dtype=torch.float32, device="mps")
    output = torch.zeros(SIZE, device="mps")
    kernel(source, output)
    torch.mps.synchronize()
    torch.testing.assert_close(output.cpu(), torch.arange(SIZE, dtype=torch.float32) * 2 + 1)


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.delenv("TILELANG_DISABLE_CACHE", raising=False)
    original = env.TILELANG_CACHE_DIR
    root = tmp_path / "tilelang-cache"
    root.mkdir()
    env.TILELANG_CACHE_DIR = str(root)
    tilelang.enable_cache()
    _dispatch_map["torch"]._memory_cache.clear()
    try:
        yield root
    finally:
        _dispatch_map["torch"]._memory_cache.clear()
        env.TILELANG_CACHE_DIR = original


@pytest.fixture
def lowerings(monkeypatch):
    """Count lowering passes; a cache hit must not lower again."""
    calls = []
    original = JITKernel._compile_artifact

    def counting(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(JITKernel, "_compile_artifact", counting)
    return calls


def test_memory_and_disk_hits_reuse_the_cached_source(cache_root, lowerings):
    func = affine(f"affine_{uuid.uuid4().hex[:8]}")
    first = compile_metal(func)
    assert len(lowerings) == 1
    assert env.is_cache_enabled(), "compiling a Metal kernel must not disable the cache"
    run(first)

    assert compile_metal(func) is first
    assert len(lowerings) == 1

    entry = Path(first._tilelang_cache_path)
    assert entry.is_relative_to(cache_root)
    for name in ("device_kernel.metal", "host_program.py", "params.json", "launch.json", "manifest.json"):
        assert (entry / name).is_file(), name
    metadata = json.loads((entry / "launch.json").read_text())
    assert metadata == first.adapter.launch_metadata

    _dispatch_map["torch"]._memory_cache.clear()
    second = compile_metal(func)
    assert len(lowerings) == 1, "a disk hit must not lower again"
    assert second is not first
    assert second.adapter.launch_metadata == first.adapter.launch_metadata
    assert second.get_kernel_source() == first.get_kernel_source()
    assert second.get_host_source() == first.get_host_source()
    assert second.params == first.params
    run(second)
    assert compile_metal(func) is second


def test_fresh_process_loads_the_entry_without_lowering(cache_root, lowerings):
    symbol = f"affine_{uuid.uuid4().hex[:8]}"
    first = compile_metal(affine(symbol))
    assert len(lowerings) == 1
    script = f"""
import json, sys
sys.path.insert(0, {str(Path(__file__).parent)!r})
import torch
import tilelang
from tilelang.jit.kernel import JITKernel
import test_metal_kernel_cache as fixture

def refuse(self, *args, **kwargs):
    raise AssertionError("cache miss: the kernel was lowered again")

JITKernel._compile_artifact = refuse
tilelang.enable_cache()
kernel = fixture.compile_metal(fixture.affine({symbol!r}))
fixture.run(kernel)
print(json.dumps({{"launch": kernel.adapter.launch_metadata, "host": kernel.get_host_source()}}))
"""
    environment = {key: value for key, value in os.environ.items() if key != "TILELANG_DISABLE_CACHE"}
    environment["TILELANG_CACHE_DIR"] = str(cache_root)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded["launch"] == first.adapter.launch_metadata
    assert loaded["host"] == first.get_host_source()


def test_damaged_entries_are_recompiled_and_repaired(cache_root, lowerings):
    func = affine(f"affine_{uuid.uuid4().hex[:8]}")
    first = compile_metal(func)
    entry = Path(first._tilelang_cache_path)
    launch = entry / "launch.json"
    launch.write_text("{not json")

    _dispatch_map["torch"]._memory_cache.clear()
    second = compile_metal(func)
    assert len(lowerings) == 2, "a corrupted entry must be recompiled"
    assert json.loads(launch.read_text()) == second.adapter.launch_metadata
    run(second)

    (entry / "params.json").unlink()
    _dispatch_map["torch"]._memory_cache.clear()
    third = compile_metal(func)
    assert len(lowerings) == 3, "an incomplete entry must be recompiled"
    assert (entry / "params.json").is_file()
    run(third)


def test_cached_kernel_keeps_the_public_call_contract(cache_root):
    func = affine(f"affine_{uuid.uuid4().hex[:8]}")
    compile_metal(func)
    _dispatch_map["torch"]._memory_cache.clear()
    kernel = compile_metal(func)
    assert kernel.execution_backend == "torch"
    assert kernel.artifact is None
    assert "#include <metal_stdlib>" in kernel.get_kernel_source()
    assert kernel.out_idx == []
    run(kernel)


if __name__ == "__main__":
    tilelang.testing.main()
