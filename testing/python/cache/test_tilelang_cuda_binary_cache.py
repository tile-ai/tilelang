from __future__ import annotations

import cloudpickle
import importlib
import os

import tilelang
import tilelang.cache.kernel_cache as kernel_cache_mod
from tilelang.cache.cuda_binary_cache import CUDABinaryCache
from tilelang.cache.kernel_cache import KernelCache
from tilelang.env import env
from tvm.target import Target


def _set_cache_dirs(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    tmp_dir = tmp_path / "tmp"
    cache_dir.mkdir()
    tmp_dir.mkdir()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(env, "TILELANG_TMP_DIR", str(tmp_dir))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0")
    tilelang.enable_cache()
    KernelCache._get_cache_namespace.cache_clear()
    CUDABinaryCache._get_tilelang_lib_stamp.cache_clear()
    return cache_dir


def test_kernel_cache_namespace_includes_host_platform(monkeypatch):
    monkeypatch.setattr(kernel_cache_mod, "__version__", "1.2.3+cuda.gitabc")
    monkeypatch.setattr(kernel_cache_mod.sys, "platform", "linux")
    monkeypatch.setattr(kernel_cache_mod.platform, "machine", lambda: "aarch64")
    KernelCache._get_cache_namespace.cache_clear()

    assert KernelCache._get_cache_namespace() == os.path.join("1.2.3_cuda_gitabc", "linux-aarch64")


def test_cuda_binary_cache_hit_skips_nvcc_compile(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    lower = importlib.import_module("tilelang.engine.lower")
    monkeypatch.setattr(env, "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", "0")

    compile_calls = []

    def fake_compile_cuda(code, target_format, arch, options=None, verbose=False):
        compile_calls.append((code, target_format, tuple(arch), tuple(options or ())))
        return bytearray(b"fake-cubin")

    monkeypatch.setattr(lower.nvcc, "compile_cuda", fake_compile_cuda)

    target = Target({"kind": "cuda", "arch": "sm_90a"})
    source = 'extern "C" __global__ void kernel() {}'

    fast_math_pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["--extra-device-vectorization"],
    }

    first = lower.tilelang_callback_cuda_compile(source, target)
    second = lower.tilelang_callback_cuda_compile(source, target)
    # Different compiler options (e.g. --use_fast_math) change the generated
    # SASS without changing the source, so they must NOT share a cache entry.
    third = lower.tilelang_callback_cuda_compile(source, target, fast_math_pass_configs)
    fourth = lower.tilelang_callback_cuda_compile(source, target, fast_math_pass_configs)

    assert bytes(first) == b"fake-cubin"
    assert bytes(second) == b"fake-cubin"
    assert bytes(third) == b"fake-cubin"
    assert bytes(fourth) == b"fake-cubin"
    # first compiles, second hits; third compiles (new options), fourth hits
    assert len(compile_calls) == 2
    assert compile_calls[0][3] != compile_calls[1][3]
    cache_files = list((tmp_path / "cache").glob("*/cuda-binaries/*.cubin"))
    assert len(cache_files) == 2


def test_disk_cache_load_failure_is_cache_miss(monkeypatch, tmp_path):
    _set_cache_dirs(monkeypatch, tmp_path)
    cache = KernelCache()
    key = "bad-host-executable"
    cache_path = tmp_path / "cache" / KernelCache._get_cache_namespace() / "kernels" / key
    cache_path.mkdir(parents=True)
    (cache_path / cache.device_kernel_path).write_text("// device")
    (cache_path / cache.host_kernel_path).write_text("// host")
    (cache_path / cache.kernel_lib_path).write_bytes(b"not-loadable")
    with (cache_path / cache.params_path).open("wb") as f:
        cloudpickle.dump(["param"], f)

    def fail_from_database(*args, **kwargs):
        raise RuntimeError("bad host executable")

    monkeypatch.setattr(kernel_cache_mod.JITKernel, "from_database", classmethod(fail_from_database))

    loaded = cache._load_kernel_from_disk(
        key,
        target="cuda",
        target_host=None,
        out_idx=[0],
        execution_backend="tvm_ffi",
        pass_configs=None,
        compile_flags=None,
        func=None,
    )

    assert loaded is None
    assert not cache_path.exists()
