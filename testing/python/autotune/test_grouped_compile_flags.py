"""The grouped autotune path must forward compile_flags to the device codegen.

``compile_grouped_unit_tvm_ffi`` compiles the merged device module itself and
then builds its per-config ``JITKernel`` objects with ``from_database=True``, so
``JITKernel``'s own merge of ``self.compile_flags`` never runs there. The device
backends read the flags from the pass configs only, so without an explicit merge
the flags that the cache identity was computed from never reach the compiler and
a kernel built without them is stored under a key that claims they were used.
"""

from tilelang.autotuner.grouped_compile import _with_device_compile_flags
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
