# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
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
    example_tilelang_sparse_gqa_decode_varlen_indice.main()


def test_example_tilelang_sparse_gqa_decode_varlen_mask():
    example_tilelang_sparse_gqa_decode_varlen_mask.main()


def test_example_triton_sparse_gqa_decode_varlen_indice():
    example_triton_sparse_gqa_decode_varlen_indice.main()


def test_example_triton_sparse_gqa_decode_varlen_mask():
    example_triton_sparse_gqa_decode_varlen_mask.main()


if __name__ == "__main__":
    tilelang.testing.main()
