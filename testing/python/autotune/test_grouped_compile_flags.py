"""The grouped autotune path must forward compile_flags to the device codegen.

``compile_grouped_unit_tvm_ffi`` compiles the merged device module itself and
then builds its per-config ``JITKernel`` objects with ``from_database=True``, so
``JITKernel``'s own merge of ``self.compile_flags`` never runs there. The device
backends read the flags from the pass configs only, so without an explicit merge
the flags that the cache identity was computed from never reach the compiler and
a kernel built without them is stored under a key that claims they were used.
"""

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.autotuner.grouped_compile import _with_device_compile_flags, compile_grouped_unit_tvm_ffi
from tilelang.autotuner.param import CompileArgs
from tilelang.transform.pass_config import PassConfigKey

FLAGS_KEY = PassConfigKey.TL_DEVICE_COMPILE_FLAGS


def test_no_flags_leaves_the_configs_untouched():
    configs = {PassConfigKey.TL_ENABLE_DUMP_IR: True}

    result = _with_device_compile_flags(configs, None)

    assert FLAGS_KEY not in result


def test_a_single_string_is_wrapped_into_a_list():
    result = _with_device_compile_flags({}, "-O3")

    assert result[FLAGS_KEY] == ["-O3"]


def test_a_list_is_forwarded_in_order():
    result = _with_device_compile_flags({}, ["-O3", "-DFOO"])

    assert result[FLAGS_KEY] == ["-O3", "-DFOO"]


def test_existing_entries_keep_their_place_before_the_explicit_flags():
    result = _with_device_compile_flags({FLAGS_KEY: ["-DEXISTING"]}, ["-DNEW"])

    assert result[FLAGS_KEY] == ["-DEXISTING", "-DNEW"]


def test_an_existing_single_string_is_normalized():
    result = _with_device_compile_flags({FLAGS_KEY: "-DEXISTING"}, ["-DNEW"])

    assert result[FLAGS_KEY] == ["-DEXISTING", "-DNEW"]


def test_other_configs_survive_the_merge():
    result = _with_device_compile_flags({PassConfigKey.TL_ENABLE_DUMP_IR: True}, ["-O3"])

    assert result[PassConfigKey.TL_ENABLE_DUMP_IR] is True
    assert result[FLAGS_KEY] == ["-O3"]


def _make_grouped_kernel(size):
    @T.prim_func
    def kernel(A: T.Tensor((size,), T.float32), B: T.Tensor((size,), T.float32)):
        with T.Kernel(1, threads=1):
            for i in T.serial(size):
                B[i] = A[i] + 1.0

    return kernel


@tilelang.testing.requires_cuda
def test_grouped_compile_forwards_the_flags_through_the_pass_configs():
    """The grouped path itself, not just the merge helper, must carry the flags.

    ``compile_grouped_unit_tvm_ffi`` compiles the merged device module once and
    then builds each per-config ``JITKernel`` with ``from_database=True``, so the
    pass configs are the only carrier the device codegen and the adapter see.
    """
    compile_args = CompileArgs(
        out_idx=[1],
        execution_backend="tvm_ffi",
        target=tvm.target.Target("cuda"),
        compile_flags=["-DGROUPED_FLAG_PROBE=1"],
    )

    results = compile_grouped_unit_tvm_ffi(
        unit_items=[(0, {"size": 17}), (1, {"size": 23})],
        compile_args=compile_args,
        elaborate_func=_make_grouped_kernel,
    )

    assert len(results) == 2
    for _, _, kernel, error in results:
        if error is not None:
            raise error
        assert list(kernel.pass_configs[FLAGS_KEY]) == ["-DGROUPED_FLAG_PROBE=1"]
