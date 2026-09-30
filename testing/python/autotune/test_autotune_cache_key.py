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


def test_cache_key_includes_compile_flags():
    """Flags forwarded to the device compiler build a different binary."""
    precise = CompileArgs(compile_flags=["--prec-div=true"])
    fast = CompileArgs(compile_flags=["--use_fast_math", "--prec-div=false"])

    assert _cache_key(compile_args=precise) != _cache_key(compile_args=fast)
    assert _cache_key(compile_args=precise) != _cache_key()
    # Same flags must still resolve to the same key, whether spelled as a list
    # or as a single string.
    assert _cache_key(compile_args=CompileArgs(compile_flags=["--use_fast_math"])) == _cache_key(
        compile_args=CompileArgs(compile_flags="--use_fast_math")
    )


def _decorated_with_flags(compile_flags):
    """Build an autotuned kernel with the given device-compiler flags.

    The target is pinned to a host target so the test does not depend on a
    device being present to auto-detect one; nothing here is lowered.
    """

    @tilelang.autotune(configs=[{"block_size": 128}])
    @tilelang.jit(compile_flags=compile_flags, target="c")
    def kernel(N: int = 256, block_size: int = 128):
        @T.prim_func
        def main(A: T.Tensor((N,), "float32")):
            with T.Kernel(T.ceildiv(N, block_size), threads=block_size):
                T.evaluate(0)

        return main

    return kernel


def test_decorator_compile_flags_reach_the_cache_identity():
    """compile_flags set on the decorator must reach the tuning cache identity.

    The flags are part of the compiled binary, so two kernels built from the
    same source with different flags must not share a cache entry. The reload
    path has to rebuild with the same flags the entry was produced under.
    """
    fast_flags = ["--use_fast_math", "--prec-div=false"]

    plain = _decorated_with_flags(None).get_tunner()
    fast = _decorated_with_flags(fast_flags).get_tunner()

    assert plain.compile_args.compile_flags is None
    assert list(fast.compile_args.compile_flags) == fast_flags
    assert hash(plain.compile_args) != hash(fast.compile_args)


def test_cache_key_is_disabled_for_profile_callbacks():
    lock = threading.Lock()

    def callback(value):
        with lock:
            return value

    for callback_field in ("ref_prog", "supply_prog", "manual_check_prog"):
        assert _cache_key(profile_args=ProfileArgs(**{callback_field: callback})) is None
