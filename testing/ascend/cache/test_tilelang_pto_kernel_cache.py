"""PTO tvm_ffi kernel-cache round-trip: layout and warm hit in a fresh process.

Mirrors the generic kernel-cache coverage in testing/python/cache/ for
``target="pto"`` with ``execution_backend="tvm_ffi"``.
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
import tilelang.testing
from tilelang.ascend import language as T
from tilelang.ascend.kernel_cache import PTOTVMFFIKernelCache
from tilelang.env import env

pytestmark = pytest.mark.pto

REPO_ROOT = Path(__file__).resolve().parents[3]

N = 64
TILE = 32

EXPECTED_FILES = [
    "device_kernel.ptodsl.py",
    "executable.so",
    "host_kernel.c",
    "manifest.json",
    "params.json",
]


def _vecadd(n, tile, symbol):
    @T.prim_func
    def main(
        a: T.Tensor((n,), T.float32),
        b: T.Tensor((n,), T.float32),
        c: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(n // tile) as bx, T.SimtVF(threads=tile):
            for i in T.Parallel(tile):
                idx = bx * tile + i
                c[idx] = a[idx] + b[idx]

    return main.with_attr("global_symbol", symbol)


def _unique_symbol(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _set_cache_dirs(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0")
    tilelang.enable_cache()
    # The in-memory kernel cache is shared across backends; drop it so every
    # compile below goes through the isolated on-disk cache.
    PTOTVMFFIKernelCache()._memory_cache.clear()
    return cache_dir


def _kernel_entries(cache_dir):
    entries = [path.parent for path in cache_dir.rglob("params.json")]
    assert len(entries) == 1, f"expected exactly one kernel-cache entry, found {entries}"
    return entries[0]


def _compile(func):
    return tilelang.compile(func, out_idx=[2], target="pto", execution_backend="tvm_ffi")


def _deterministic_inputs(n):
    a = torch.arange(n, dtype=torch.float32, device="npu")
    b = torch.arange(n, dtype=torch.float32, device="npu") * 2
    return a, b


def test_pto_kernel_cache_layout(monkeypatch, tmp_path):
    """A cold compile stores exactly the PTOTVMFFIKernelCache artifacts."""
    cache_dir = _set_cache_dirs(monkeypatch, tmp_path)

    _compile(_vecadd(N, TILE, _unique_symbol("pto_cache_layout")))

    entry = _kernel_entries(cache_dir)
    assert entry.parent.name == "kernels"
    assert sorted(path.name for path in entry.iterdir()) == EXPECTED_FILES

    device_source = (entry / "device_kernel.ptodsl.py").read_text()
    assert "@pto.jit" in device_source

    host_source = (entry / "host_kernel.c").read_text()
    assert host_source.startswith("// tilelang target:")

    manifest = json.loads((entry / "manifest.json").read_text())
    assert set(manifest["files"]) == set(EXPECTED_FILES) - {"manifest.json"}
    for name, meta in manifest["files"].items():
        assert (entry / name).stat().st_size == meta["size"]


@tilelang.testing.requires_ascend
def test_pto_kernel_cache_hit_in_fresh_process(monkeypatch, tmp_path):
    """A fresh process reuses the on-disk entry and produces bit-exact outputs."""
    cache_dir = _set_cache_dirs(monkeypatch, tmp_path)
    symbol = _unique_symbol("pto_cache_subprocess")
    _compile(_vecadd(N, TILE, symbol))

    probe = tmp_path / "warm_compile_probe.py"
    probe.write_text(
        """
import sys

test_dir, symbol, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, test_dir)

import torch

import tilelang
import test_tilelang_pto_kernel_cache as cache_test
from tilelang.contrib import pto_toolchain


toolchain_calls = []
real_compile = pto_toolchain.compile_ptodsl_to_aibin


def counting_compile(code, pto_arch, pto_flags, verbose=False):
    toolchain_calls.append(code)
    return real_compile(code, pto_arch, pto_flags, verbose=verbose)


pto_toolchain.compile_ptodsl_to_aibin = counting_compile

kernel = cache_test._compile(cache_test._vecadd(cache_test.N, cache_test.TILE, symbol))
a, b = cache_test._deterministic_inputs(cache_test.N)
out = kernel(a, b)
torch.npu.synchronize()
torch.save(out.cpu(), output_path)
print(f"toolchain_calls={len(toolchain_calls)}")
"""
    )
    output = tmp_path / "probe_output.pt"

    probe_env = dict(os.environ)
    probe_env["TILELANG_CACHE_DIR"] = str(cache_dir)
    probe_env["TILELANG_DISABLE_CACHE"] = "0"
    pythonpath = probe_env.get("PYTHONPATH")
    probe_env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{pythonpath}" if pythonpath else str(REPO_ROOT)

    result = subprocess.run(
        [sys.executable, str(probe), str(Path(__file__).resolve().parent), symbol, str(output)],
        env=probe_env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"

    calls_lines = [line for line in result.stdout.splitlines() if line.startswith("toolchain_calls=")]
    assert calls_lines, f"probe printed no toolchain count:\n{result.stdout}\n{result.stderr}"
    assert int(calls_lines[-1].split("=")[1]) == 0, "fresh process must hit the kernel cache without recompiling"

    warm_output = torch.load(output, weights_only=True)
    a, b = _deterministic_inputs(N)
    torch.testing.assert_close(warm_output, a.cpu() + b.cpu(), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
