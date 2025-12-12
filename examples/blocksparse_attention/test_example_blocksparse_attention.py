import tilelang.testing
import block_sparse_attn_triton
import example_tilelang_block_sparse_attn
import example_tilelang_sparse_gqa_decode_varlen_indice
import example_tilelang_sparse_gqa_decode_varlen_mask
import example_triton_sparse_gqa_decode_varlen_indice
import example_triton_sparse_gqa_decode_varlen_mask


def test_block_sparse_attn_triton():
    block_sparse_attn_triton.main()


def test_example_tilelang_block_sparse_attn():
    example_tilelang_block_sparse_attn.main()


def test_example_tilelang_sparse_gqa_decode_varlen_indice():
    example_tilelang_sparse_gqa_decode_varlen_indice.main(batch=1, max_cache_seqlen=2048)


def test_example_tilelang_sparse_gqa_decode_varlen_mask():
    example_tilelang_sparse_gqa_decode_varlen_mask.main(batch=1, max_cache_seqlen=2048)


def test_example_triton_sparse_gqa_decode_varlen_indice():
    example_triton_sparse_gqa_decode_varlen_indice.main(
        batch=8, heads=8, heads_kv=4, max_cache_seqlen=2048, dim=128, dim_v=128, sparse_ratio=0.8, block_size=32
    )


def test_example_triton_sparse_gqa_decode_varlen_mask():
    example_triton_sparse_gqa_decode_varlen_mask.main(
        batch=16, heads=16, heads_kv=8, max_cache_seqlen=1024, dim=128, dim_v=128, sparse_ratio=0.8, block_size=32
    )


if __name__ == "__main__":
    tilelang.testing.main()
