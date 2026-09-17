import pytest
import torch

import tilelang.language as T
import tilelang.testing
from tilelang.tileir.checks import TileIRDependencyError, check_tileir_available

from sparse_attn_fwd_tileir import check_output, sparse_attn_fwd


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "heads,head_tile,seq_len,dim,num_ctas,padded",
    [
        (128, 128, 8, 128, 2, False),
        (256, 128, 7, 64, 2, True),
        (128, 64, 5, 128, 1, True),
        (16, 16, 1, 256, 2, True),
        (32, 32, 3, 128, 2, True),
    ],
)
def test_sparse_attn_fwd_tileir(dtype, heads, head_tile, seq_len, dim, num_ctas, padded):
    try:
        check_tileir_available()
    except TileIRDependencyError as exc:
        pytest.skip(str(exc))
    torch.manual_seed(42)
    batch, seq_len_kv, topk = 2, 256, 128
    torch_dtype = getattr(torch, dtype)
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch_dtype)
    kv = torch.randn(batch, seq_len_kv, dim, device="cuda", dtype=torch_dtype)
    indices = torch.randint(0, seq_len_kv, (batch, seq_len, topk), device="cuda", dtype=torch.int32)
    sinks = torch.randn(heads, device="cuda", dtype=torch_dtype)
    if padded:
        indices[:, :, ::3] = -1
        indices[0, 0, :] = -1
    kernel = sparse_attn_fwd(
        batch,
        heads,
        seq_len,
        seq_len_kv,
        dim,
        topk,
        dtype=T.dtype(dtype),
        H_per_block=head_tile,
        num_ctas=num_ctas,
    )
    output = kernel(q, kv, indices, sinks)
    check_output(output, q, kv, indices, sinks)
    if padded:
        torch.testing.assert_close(output[0, 0], torch.zeros_like(output[0, 0]), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
