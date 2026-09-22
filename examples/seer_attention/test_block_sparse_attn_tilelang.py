import tilelang.testing

import block_sparse_attn_tilelang


@tilelang.testing.requires_cuda
def test_block_sparse_attn_tilelang():
    block_sparse_attn_tilelang.main()


@tilelang.testing.requires_rocm
def test_block_sparse_attn_tilelang_rocm():
    """Run the existing bounded Seer validations on ROCm."""
    block_sparse_attn_tilelang.test_topk_sparse_attention(batch=1, heads=2, sequence_length=128, topk=2)
    block_sparse_attn_tilelang.test_topk_sparse_attention_qlen_lt_klen()


if __name__ == "__main__":
    tilelang.testing.main()
