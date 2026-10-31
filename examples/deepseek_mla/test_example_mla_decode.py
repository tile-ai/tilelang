import os
import pytest
import torch
import tilelang.testing
import example_mla_decode

_is_cutedsl = os.environ.get("TILELANG_DEFAULT_TARGET", "").lower() == "cutedsl"


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_example_mla_decode():
    example_mla_decode.main()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.parametrize("heads", [1, 24])
@pytest.mark.parametrize("num_split", [1, 2])
def test_mla_decode_partial_head_tile(heads, num_split):
    """Both decode paths must write every head in a partial MMA tile."""
    if _is_cutedsl and num_split > 1:
        pytest.skip("CuTeDSL backend does not support alloc_global yet")
    batch, kv_heads, kv_ctx, dim, pe_dim = 1, 1, 256, 512, 64
    block_n, block_h, threads = example_mla_decode.get_default_config(heads, kv_heads)
    kernel = example_mla_decode.flashattn(
        batch, heads, kv_heads, kv_ctx, dim, pe_dim, block_n, block_h, num_split, (dim + pe_dim) ** -0.5, threads=threads
    )
    shapes = [(batch, heads, dim), (batch, heads, pe_dim), (batch, kv_ctx, kv_heads, dim), (batch, kv_ctx, kv_heads, pe_dim)]
    inputs = [torch.randn(shape, device="cuda", dtype=torch.float16) for shape in shapes]
    expected = example_mla_decode.ref_program(*(x.float() for x in inputs))
    actual = kernel(*inputs)
    torch.testing.assert_close(actual.float(), expected, rtol=1e-3, atol=2e-3)


if __name__ == "__main__":
    tilelang.testing.main()
