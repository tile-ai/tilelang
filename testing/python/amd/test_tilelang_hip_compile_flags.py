"""compile_flags and tl.device_compile_flags reach hipcc on HIP."""

import pytest
import tilelang
import tilelang.rocm.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey

# The kernel only compiles when the flag defines the macro, and its result is the macro's value.
_PRELUDE = """
#ifndef TL_TEST_FLAG_VALUE
#error "compile flag did not reach hipcc"
#endif
"""


def _make():
    @T.prim_func
    def main(out: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=64, prelude=_PRELUDE):
            for i in T.Parallel(1):
                out[i] = 0

    return main


@tilelang.testing.requires_rocm
def test_missing_flag_fails_to_compile():
    with pytest.raises(Exception, match="compile flag did not reach hipcc"):
        tilelang.compile(_make(), out_idx=[0], target="hip")


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("flags", [["-DTL_TEST_FLAG_VALUE=1"], "-DTL_TEST_FLAG_VALUE=1", ["-O3 -DTL_TEST_FLAG_VALUE=1"]])
def test_compile_flags_reach_hipcc(flags):
    assert tilelang.compile(_make(), out_idx=[0], target="hip", compile_flags=flags) is not None


@tilelang.testing.requires_rocm
def test_pass_config_flags_reach_hipcc():
    kernel = tilelang.compile(
        _make(),
        out_idx=[0],
        target="hip",
        pass_configs={PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["-DTL_TEST_FLAG_VALUE=1"]},
    )
    assert kernel is not None


@tilelang.testing.requires_rocm
def test_nvcc_only_flag_is_dropped_with_warning():
    with pytest.warns(UserWarning, match="--use_fast_math"):
        kernel = tilelang.compile(
            _make(),
            out_idx=[0],
            target="hip",
            compile_flags=["--use_fast_math", "-DTL_TEST_FLAG_VALUE=1"],
        )
    assert kernel is not None


def test_hip_compile_flags_split_and_filter():
    from tilelang.rocm.backend import hip_compile_flags

    assert hip_compile_flags(None) == []
    assert hip_compile_flags("-O3 -DX=1") == ["-O3", "-DX=1"]
    assert hip_compile_flags(["-mllvm -some-option", "-DX=1"]) == ["-mllvm", "-some-option", "-DX=1"]
    with pytest.warns(UserWarning):
        assert hip_compile_flags(["--use_fast_math", "-DX=1"]) == ["-DX=1"]


if __name__ == "__main__":
    tilelang.testing.main()
