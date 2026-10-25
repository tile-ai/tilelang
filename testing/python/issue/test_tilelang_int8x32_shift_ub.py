import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _run_non_broadcast_8bit_vector_add(lanes, dtype, torch_dtype):
    n = lanes

    @T.prim_func
    def main(
        a: T.Tensor((n,), dtype),
        b: T.Tensor((n,), dtype),
        c: T.Tensor((n,), dtype),
    ):
        with T.Kernel(1, threads=1):
            for i in T.vectorized(n):
                c[i] = a[i] + b[i]

    kernel = tilelang.compile(main, target="cuda")
    source = kernel.get_kernel_source()

    if torch_dtype is torch.int8:
        a = torch.arange(-(lanes // 2), lanes // 2, dtype=torch_dtype, device="cuda")
    else:
        a = torch.arange(128, 128 + n, dtype=torch_dtype, device="cuda")
    b = torch.full((n,), 3, dtype=torch_dtype, device="cuda")
    c = torch.empty_like(a)
    kernel(a, b, c)
    torch.cuda.synchronize()

    expected = (a.to(torch.int16) + b.to(torch.int16)).to(torch_dtype)
    torch.testing.assert_close(c, expected, rtol=0, atol=0)
    return source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    ("lanes", "dtype", "torch_dtype"),
    [(lanes, dtype, torch_dtype) for lanes in [8, 16] for dtype, torch_dtype in [("int8", torch.int8), ("uint8", torch.uint8)]],
)
def test_non_broadcast_8bit_small_vector_add_is_packed(lanes, dtype, torch_dtype):
    source = _run_non_broadcast_8bit_vector_add(lanes, dtype, torch_dtype)
    packed = re.search(r"\b(?:u?int|u?char)(?:2|4)\s+(__\w+);", source)
    assert packed is not None
    vector = packed.group(1)
    for field in "xyzw"[: lanes // 4]:
        pattern = rf"reinterpret_cast<(?:signed char|unsigned char|uchar)\*>\(&\({re.escape(vector)}\.{field}\)\)\[(\d+)\]\s*="
        writes = [line for line in source.splitlines() if re.search(pattern, line)]
        assert [int(re.search(pattern, line).group(1)) for line in writes] == list(range(4))
        assert all(not re.search(rf"\b{re.escape(vector)}\b", line.split("=", 1)[1]) for line in writes)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@pytest.mark.parametrize(
    ("dtype", "torch_dtype", "expected_vector_type"),
    [("int8", torch.int8, "longlong4"), ("uint8", torch.uint8, "ulonglong4")],
)
def test_non_broadcast_8bit_wide_vector_add_is_packed(dtype, torch_dtype, expected_vector_type):
    source = _run_non_broadcast_8bit_vector_add(32, dtype, torch_dtype)
    packed = re.search(rf"\b{expected_vector_type}\s+(__\w+);", source)
    assert packed is not None
    vector = packed.group(1)
    pattern = rf"reinterpret_cast<(?:signed char|unsigned char|uchar)\*>\(&\({re.escape(vector)}\)\)\[(\d+)\]\s*="
    writes = [line for line in source.splitlines() if re.search(pattern, line)]
    assert [int(re.search(pattern, line).group(1)) for line in writes] == list(range(32))
    assert all(not re.search(rf"\b{re.escape(vector)}\b", line.split("=", 1)[1]) for line in writes)
    assert "0x000000ffu <<" not in source
