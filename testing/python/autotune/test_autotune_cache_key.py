"""Tests for auto-tuner cache-key identity."""

import inspect
import threading
from dataclasses import replace

import tilelang
import tilelang.language as T
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


def _decorated(warmup, rep, timeout):
    """Build an autotuned kernel whose profiler settings differ per call."""

    @tilelang.autotune(configs=[{"block_size": 128}], warmup=warmup, rep=rep, timeout=timeout)
    @tilelang.jit
    def kernel(N: int = 256, block_size: int = 128):
        @T.prim_func
        def main(A: T.Tensor((N,), "float32")):
            with T.Kernel(T.ceildiv(N, block_size), threads=block_size):
                T.evaluate(0)

        return main

    return kernel


def test_decorator_profile_settings_reach_the_cache_identity():
    """warmup/rep/timeout set on the decorator must reach the tuning cache identity.

    The cache key is built from ``hash(profile_args)``, which already covers these
    three fields. When the decorator did not forward them, the tuner kept the
    dataclass defaults, so two kernels configured with different measurement
    settings produced the same key and the second silently reused the first's
    tuning result.
    """
    short = _decorated(warmup=10, rep=20, timeout=5).get_tunner()
    long = _decorated(warmup=500, rep=1000, timeout=60).get_tunner()

    assert (short.profile_args.warmup, short.profile_args.rep, short.profile_args.timeout) == (10, 20, 5)
    assert (long.profile_args.warmup, long.profile_args.rep, long.profile_args.timeout) == (500, 1000, 60)
    assert hash(short.profile_args) != hash(long.profile_args)
