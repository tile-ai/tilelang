from types import SimpleNamespace

import pytest
import torch

import tilelang.language as T
from tilelang.engine.param import KernelParam
from tvm import tirx


PTO_TARGET = SimpleNamespace(kind=SimpleNamespace(name="ascend"), keys=("ascend", "pto"))
CUDA_TARGET = SimpleNamespace(kind=SimpleNamespace(name="cuda"), keys=("cuda",))


@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"),
    reason="PyTorch float4_e2m1fn_x2 dtype is unavailable",
)
def test_scalar_fp4_uses_packed_torch_storage_shape():
    scalar_fp4 = KernelParam(T.float4_e2m1fn, [4, 256])
    packed_fp4 = KernelParam(T.float4_e2m1fnx2, [4, 128])

    assert scalar_fp4.storage_packing_factor(target=PTO_TARGET) == 2
    assert scalar_fp4.storage_shape(target=PTO_TARGET) == [4, 128]
    assert packed_fp4.storage_packing_factor(target=PTO_TARGET) == 1
    assert packed_fp4.storage_shape(target=PTO_TARGET) == [4, 128]


def test_int4_uses_pto_packed_storage_shape():
    int4 = KernelParam(T.int4, [4, 256])

    assert int4.storage_packing_factor(target=PTO_TARGET) == 2
    assert int4.storage_shape(target=PTO_TARGET) == [4, 128]


def test_bool_uses_its_existing_byte_per_element_abi():
    boolean = KernelParam(T.bool, [4, 4])

    assert boolean.storage_packing_factor(target=PTO_TARGET) == 1
    assert boolean.storage_shape(target=PTO_TARGET) == [4, 4]


@pytest.mark.parametrize("dtype", [T.float4_e2m1fn, T.int4])
@pytest.mark.parametrize("innermost", [tirx.IntImm("int32", 255), tirx.Var("dynamic_extent", "int32")])
def test_packed_storage_requires_static_even_innermost_extent(dtype, innermost):
    packed = KernelParam(dtype, [4, innermost])

    with pytest.raises(ValueError, match="innermost dimension"):
        packed.storage_shape(target=PTO_TARGET)


@pytest.mark.parametrize("dtype", [T.float4_e2m1fn, T.int4, T.dtype("uint4")])
def test_packed_storage_abi_is_pto_only(dtype):
    packed = KernelParam(dtype, [4, 256])

    assert packed.storage_packing_factor(target=CUDA_TARGET) == 1
    assert packed.storage_shape(target=CUDA_TARGET) == [4, 256]
