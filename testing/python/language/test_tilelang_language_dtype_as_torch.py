import pytest
import torch

import tilelang.language as T
from tilelang.language.dtypes import resolve_torch_storage_dtype


@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"),
    reason="PyTorch float4_e2m1fn_x2 dtype is unavailable",
)
def test_float4_e2m1fnx2_as_torch_uses_storage_dtype_name():
    assert T.float4_e2m1fnx2.as_torch() is torch.float4_e2m1fn_x2
    assert T.float4_e2m1fn.as_torch() is torch.float4_e2m1fn_x2


def test_resolve_torch_storage_dtype_round_trips_through_torch():
    assert resolve_torch_storage_dtype(T.float16) == T.float16
    assert resolve_torch_storage_dtype(T.int4) == T.int8
    assert resolve_torch_storage_dtype(T.dtype("uint4")) == T.uint8

    expected_fp4 = T.float4_e2m1fnx2 if hasattr(torch, "float4_e2m1fn_x2") else T.int8
    assert resolve_torch_storage_dtype(T.float4_e2m1fn) == expected_fp4
