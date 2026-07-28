"""Tests for auto-tuner cache-key identity."""

import inspect
from dataclasses import replace

from tilelang.autotuner import AutoTuner
from tilelang.autotuner.param import CompileArgs, ProfileArgs


def _kernel(block_size=128):
    return block_size


def _make_callback(offset):
    def callback(value):
        return value + offset

    return callback


def _cache_key(compile_args=None, profile_args=None):
    tuner = AutoTuner(_kernel, configs=[{"block_size": 128}])
    tuner.compile_args = compile_args or CompileArgs()
    tuner.profile_args = profile_args or ProfileArgs()
    return tuner.generate_cache_key(inspect.signature(_kernel).parameters, {})


def test_cache_key_includes_output_indices():
    base_args = CompileArgs(out_idx=[0])

    assert _cache_key(compile_args=base_args) != _cache_key(compile_args=replace(base_args, out_idx=[1]))


def test_cache_key_includes_profile_validation_and_input_behavior():
    base_args = ProfileArgs(
        ref_prog=_make_callback(1),
        supply_prog=_make_callback(2),
        manual_check_prog=_make_callback(3),
        skip_check=False,
        cache_input_tensors=False,
    )

    variants = (
        replace(base_args, ref_prog=_make_callback(4)),
        replace(base_args, supply_prog=_make_callback(4)),
        replace(base_args, manual_check_prog=_make_callback(4)),
        replace(base_args, skip_check=True),
        replace(base_args, cache_input_tensors=True),
    )

    base_key = _cache_key(profile_args=base_args)
    assert all(base_key != _cache_key(profile_args=variant) for variant in variants)


def test_cache_key_fingerprints_callback_content():
    first_args = ProfileArgs(ref_prog=_make_callback(1))
    equivalent_args = ProfileArgs(ref_prog=_make_callback(1))

    assert first_args.ref_prog is not equivalent_args.ref_prog
    assert _cache_key(profile_args=first_args) == _cache_key(profile_args=equivalent_args)
