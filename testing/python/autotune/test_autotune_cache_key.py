"""Tests for auto-tuner cache-key identity."""

import inspect
import threading
from dataclasses import replace

import pytest

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


@pytest.fixture(params=["target", "target_host"])
def target_config(request):
    if request.param == "target":
        return request.param, {"kind": "cuda", "arch": "sm_80"}, "arch", "sm_90"
    return request.param, {"kind": "llvm", "mcpu": "x86-64"}, "mcpu", "haswell"


def test_equal_target_dicts_have_equal_compile_argument_hashes(target_config):
    target_field, config, _, _ = target_config
    first = CompileArgs(**{target_field: config})
    second = CompileArgs(**{target_field: dict(reversed(config.items()))})

    assert first == second
    assert hash(first) == hash(second)
    assert {first: "cached"}[second] == "cached"


def test_cache_key_ignores_target_dict_insertion_order(target_config):
    target_field, config, _, _ = target_config
    first = CompileArgs(**{target_field: config})
    second = CompileArgs(**{target_field: dict(reversed(config.items()))})

    assert _cache_key(compile_args=first) == _cache_key(compile_args=second)


def test_cache_key_distinguishes_target_dict_values(target_config):
    target_field, config, key, value = target_config
    first = CompileArgs(**{target_field: config})
    second = CompileArgs(**{target_field: {**config, key: value}})

    assert _cache_key(compile_args=first) != _cache_key(compile_args=second)


def test_target_dict_hash_accepts_target_objects():
    from tilelang import tvm

    host = tvm.target.Target({"kind": "llvm", "mcpu": "x86-64"})
    config = {"kind": "cuda", "arch": "sm_80", "host": host}
    first = CompileArgs(target=config)
    second = CompileArgs(target=dict(reversed(config.items())))

    assert hash(first) == hash(second)
