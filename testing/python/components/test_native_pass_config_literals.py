"""Public JIT configuration literals cross the native POD boundary explicitly."""

import tvm_tirx

import pytest

import tilelang
import tilelang.cpu.language as T
from tilelang.transform import PassConfigKey
from tilelang.transform.pass_config import normalize_pass_configs
from tvm_tirx import ir
from tvm_tirx.ir.transform import PassContext
from tvm_tirx.target import Target


@pytest.mark.parametrize("key", [PassConfigKey.TL_DISABLE_DATA_RACE_CHECK, "tirx.disable_vectorize"])
@pytest.mark.parametrize("value", [0, 2, -1])
def test_integer_literals_for_boolean_configs(key, value):
    configs = normalize_pass_configs({key: ir.IntImm("int32", value)})
    assert configs[key] is (value != 0)
    with PassContext(config=configs) as context:
        assert context.config[key] is (value != 0)


@pytest.mark.parametrize("value", [-(1 << 63), 32, (1 << 63) - 1])
def test_integer_config_preserves_literal_value(value):
    key = "tl.cpu_parallel_min_trip"
    configs = normalize_pass_configs({key: ir.IntImm("int64", value)})
    assert type(configs[key]) is int and configs[key] == value
    with PassContext(config=configs) as context:
        assert context.config[key] == value


def test_integer_config_rejects_overflow():
    with pytest.raises(OverflowError, match="signed 64-bit"):
        normalize_pass_configs({"tl.cpu_parallel_min_trip": ir.IntImm("uint64", 1 << 63)})


@pytest.mark.parametrize("value", [ir.FloatImm("float32", 1), ir.Var("dynamic", ir.PrimType("int32")), "1", None])
def test_non_integer_values_keep_native_diagnostics(value):
    key = "tl.disable_data_race_check"
    configs = normalize_pass_configs({key: value})
    assert configs[key] is value
    with pytest.raises((AttributeError, TypeError)):
        PassContext(config=configs)


def test_unknown_key_is_not_coerced_or_silently_accepted():
    literal = ir.IntImm("int32", 1)
    configs = normalize_pass_configs({"tl.not_a_registered_config": literal})
    assert configs["tl.not_a_registered_config"] is literal
    with pytest.raises(AttributeError, match="Invalid config option"):
        PassContext(config=configs)


@pytest.mark.skipif(not tvm_tirx.runtime.enabled("llvm"), reason="LLVM backend is required")
def test_cpu_jit_accepts_literal_pass_configs():
    import torch

    @T.prim_func
    def add_one(A: T.Tensor((17,), "float32"), B: T.Tensor((17,), "float32")):
        with T.Kernel(1):
            for i in T.Parallel(17):
                B[i] = A[i] + 1.0

    target = Target({"kind": "llvm", "mcpu": "generic"})
    kernel = tilelang.compile(
        add_one,
        out_idx=[1],
        target=target,
        target_host=target,
        pass_configs={"tl.disable_data_race_check": T.int32(2), "tl.config_index_bitwidth": T.int32(32)},
    )
    inputs = torch.arange(17, dtype=torch.float32)
    torch.testing.assert_close(kernel(inputs), inputs + 1, rtol=0, atol=0)
