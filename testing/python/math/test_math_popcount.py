import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


_POPCOUNT_DTYPES = [T.int8, T.int16, T.int32, T.int64, T.uint8, T.uint16, T.uint32, T.uint64]


def _make_popcount_kernel(n):
    @T.prim_func
    def main(
        A0: T.Tensor((n,), T.int8),
        A1: T.Tensor((n,), T.int16),
        A2: T.Tensor((n,), T.int32),
        A3: T.Tensor((n,), T.int64),
        A4: T.Tensor((n,), T.uint8),
        A5: T.Tensor((n,), T.uint16),
        A6: T.Tensor((n,), T.uint32),
        A7: T.Tensor((n,), T.uint64),
        B0: T.Tensor((n,), T.int8),
        B1: T.Tensor((n,), T.int16),
        B2: T.Tensor((n,), T.int32),
        B3: T.Tensor((n,), T.int64),
        B4: T.Tensor((n,), T.uint8),
        B5: T.Tensor((n,), T.uint16),
        B6: T.Tensor((n,), T.uint32),
        B7: T.Tensor((n,), T.uint64),
    ):
        inputs = [A0, A1, A2, A3, A4, A5, A6, A7]
        outputs = [B0, B1, B2, B3, B4, B5, B6, B7]
        for src, dst in zip(inputs, outputs):
            with T.Kernel(1, threads=32):
                for i in T.Parallel(n):
                    dst[i] = T.popcount(src[i])

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("elements_per_thread", [1, 4])
def test_popcount(elements_per_thread):
    n = 32 * elements_per_thread
    inputs = []
    expected = []
    for dtype in _POPCOUNT_DTYPES:
        torch_dtype = dtype.as_torch()
        bits = torch.iinfo(torch_dtype).bits
        mask = (1 << bits) - 1
        patterns = [0, 1, 7, mask, mask - 1, 1 << (bits - 1), mask >> 1, mask // 3]
        patterns *= n // len(patterns)
        signed = dtype in (T.int8, T.int16, T.int32, T.int64)
        values = [x - (1 << bits) if signed and x >= (1 << (bits - 1)) else x for x in patterns]
        inputs.append(torch.tensor(values, dtype=torch_dtype, device="cuda"))
        expected.append([x.bit_count() for x in patterns])

    kernel = tilelang.compile(_make_popcount_kernel(n), out_idx=list(range(8, 16)), target="cuda")
    results = kernel(*inputs)
    for result, reference in zip(results, expected):
        assert result.cpu().tolist() == reference


if __name__ == "__main__":
    tilelang.testing.main()
