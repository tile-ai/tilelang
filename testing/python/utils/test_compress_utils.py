# Copyright (c) Tile-AI Organization.
# Licensed under the MIT License.
import torch
import tilelang
from tilelang.utils.sparse import compress_sm90


def generate_2_to_4_sparse_tensor(shape, dtype=torch.float32, device='cpu'):
    if shape[-1] % 4 != 0:
        raise ValueError("Last dimension must be divisible by 4 for 2:4 sparsity.")

    full_tensor = torch.randn(shape, dtype=torch.float32, device=device)
    mask = torch.zeros_like(full_tensor, dtype=torch.bool)

    group_count = shape[-1] // 4
    group_shape = shape[:-1] + (group_count, 4)

    reshaped = full_tensor.view(*group_shape)

    for idx in range(reshaped.numel() // 4):
        flat_idx = torch.randint(0, 4, (2,), dtype=torch.int64)
        while flat_idx[0] == flat_idx[1]:
            flat_idx[1] = torch.randint(0, 4, (1,), dtype=torch.int64)
        i = idx // group_count
        j = idx % group_count
        mask.view(*group_shape)[i, j, flat_idx[0]] = True
        mask.view(*group_shape)[i, j, flat_idx[1]] = True

    sparse_tensor = full_tensor * mask
    return sparse_tensor.to(dtype)


def _test_compress_sm90(M, K, block_k, dtype):
    A = generate_2_to_4_sparse_tensor((M, K), dtype=dtype, device='cuda')
    A_sparse, E = compress_sm90(A, block_k, False)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_compress_sm90():
    _test_compress_sm90(1024, 1024, 128, torch.float16)
    _test_compress_sm90(1024, 1024, 64, torch.float16)
    _test_compress_sm90(1024, 1024, 32, torch.float16)

    _test_compress_sm90(1024, 1024, 128, torch.bfloat16)
    _test_compress_sm90(1024, 1024, 64, torch.bfloat16)
    _test_compress_sm90(1024, 1024, 32, torch.bfloat16)

    _test_compress_sm90(1024, 1024, 64, torch.float32)
    _test_compress_sm90(1024, 1024, 32, torch.float32)
    _test_compress_sm90(1024, 1024, 16, torch.float32)

    _test_compress_sm90(1024, 1024, 256, torch.float8_e4m3fn)
    _test_compress_sm90(1024, 1024, 128, torch.float8_e4m3fn)
    _test_compress_sm90(1024, 1024, 64, torch.float8_e4m3fn)

    _test_compress_sm90(1024, 1024, 256, torch.float8_e5m2)
    _test_compress_sm90(1024, 1024, 128, torch.float8_e5m2)
    _test_compress_sm90(1024, 1024, 64, torch.float8_e5m2)


if __name__ == "__main__":
    test_compress_sm90()
    print("All tests passed.")
