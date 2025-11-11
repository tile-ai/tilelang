import tilelang.tools.bench
import block_sparse_attn_triton
import example_tilelang_block_sparse_attn
import example_tilelang_sparse_gqa_decode_varlen_indice
import example_tilelang_sparse_gqa_decode_varlen_mask
import example_triton_sparse_gqa_decode_varlen_indice
import example_triton_sparse_gqa_decode_varlen_mask


def bench_block_sparse_attn_triton():
    tilelang.tools.bench.process_func(block_sparse_attn_triton.main)


def bench_example_tilelang_block_sparse_attn():
    tilelang.tools.bench.process_func(example_tilelang_block_sparse_attn.main)


def bench_example_tilelang_sparse_gqa_decode_varlen_indice():
    tilelang.tools.bench.process_func(example_tilelang_sparse_gqa_decode_varlen_indice.main)


def bench_example_tilelang_sparse_gqa_decode_varlen_mask():
    tilelang.tools.bench.process_func(example_tilelang_sparse_gqa_decode_varlen_mask.main)


def bench_example_triton_sparse_gqa_decode_varlen_indice():
    tilelang.tools.bench.process_func(example_triton_sparse_gqa_decode_varlen_indice.main)


def bench_example_triton_sparse_gqa_decode_varlen_mask():
    tilelang.tools.bench.process_func(example_triton_sparse_gqa_decode_varlen_mask.main)


def main():
    tilelang.tools.bench.main()


if __name__ == "__main__":
    main()
