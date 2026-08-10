"""Tests for auto-tuner cache-key identity."""

import inspect
import threading
from dataclasses import replace

import tilelang.autotuner.param as param_mod
from tilelang.autotuner import AutoTuner
from tilelang.autotuner.param import CompileArgs, ProfileArgs


def _kernel(block_size=128):
    return block_size


def _cache_key(compile_args=None, profile_args=None):
    tuner = AutoTuner(_kernel, configs=[{"block_size": 128}])
    tuner.compile_args = compile_args or CompileArgs()
    tuner.profile_args = profile_args or ProfileArgs()
    return tuner.generate_cache_key(inspect.signature(_kernel).parameters, {})


def test_cache_key_includes_output_indices():
    base_args = CompileArgs(out_idx=[0])

    assert _cache_key(compile_args=base_args) != _cache_key(compile_args=replace(base_args, out_idx=[1]))


def test_cache_key_tracks_tvm_ffi_callable_abi(monkeypatch):
    tvm_ffi_key = _cache_key(compile_args=CompileArgs(execution_backend="tvm_ffi"))
    auto_key = _cache_key(compile_args=CompileArgs(execution_backend="auto"))
    cython_key = _cache_key(compile_args=CompileArgs(execution_backend="cython"))

    monkeypatch.setattr(
        param_mod,
        "TVM_FFI_KERNEL_ABI_VERSION",
        param_mod.TVM_FFI_KERNEL_ABI_VERSION + 1,
    )

    assert _cache_key(compile_args=CompileArgs(execution_backend="tvm_ffi")) != tvm_ffi_key
    assert _cache_key(compile_args=CompileArgs(execution_backend="auto")) != auto_key
    assert _cache_key(compile_args=CompileArgs(execution_backend="cython")) == cython_key


def test_cache_key_tracks_tvm_ffi_torch_storage_abi(monkeypatch):
    tvm_ffi_key = _cache_key(compile_args=CompileArgs(execution_backend="tvm_ffi"))
    auto_key = _cache_key(compile_args=CompileArgs(execution_backend="auto"))
    cython_key = _cache_key(compile_args=CompileArgs(execution_backend="cython"))

    monkeypatch.setattr(param_mod, "get_tvm_ffi_torch_storage_abi_tag", lambda: "different-storage-abi")

    assert _cache_key(compile_args=CompileArgs(execution_backend="tvm_ffi")) != tvm_ffi_key
    assert _cache_key(compile_args=CompileArgs(execution_backend="auto")) != auto_key
    assert _cache_key(compile_args=CompileArgs(execution_backend="cython")) == cython_key


def test_cache_key_includes_profile_validation_and_input_behavior():
    base_args = ProfileArgs(skip_check=False, cache_input_tensors=False)

    variants = (
        replace(base_args, skip_check=True),
        replace(base_args, cache_input_tensors=True),
    )

    base_key = _cache_key(profile_args=base_args)
    assert all(base_key != _cache_key(profile_args=variant) for variant in variants)


def test_cache_key_is_disabled_for_profile_callbacks():
    lock = threading.Lock()

    def callback(value):
        with lock:
            return value

    for callback_field in ("ref_prog", "supply_prog", "manual_check_prog"):
        assert _cache_key(profile_args=ProfileArgs(**{callback_field: callback})) is None
