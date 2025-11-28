import tilelang.tools.bench
import block_sparse_attn_triton
import example_tilelang_block_sparse_attn
import example_tilelang_sparse_gqa_decode_varlen_indice
import example_tilelang_sparse_gqa_decode_varlen_mask
import example_triton_sparse_gqa_decode_varlen_indice
import example_triton_sparse_gqa_decode_varlen_mask


def bench_block_sparse_attn_triton():
    tilelang.tools.bench.process_func(block_sparse_attn_triton.benchmark)


def bench_example_tilelang_block_sparse_attn():
    tilelang.tools.bench.process_func(example_tilelang_block_sparse_attn.benchmark)


def bench_example_tilelang_sparse_gqa_decode_varlen_indice():
    tilelang.tools.bench.process_func(
        example_tilelang_sparse_gqa_decode_varlen_indice.benchmark, batch=1, max_cache_seqlen=2048)


def bench_example_tilelang_sparse_gqa_decode_varlen_mask():
    tilelang.tools.bench.process_func(
        example_tilelang_sparse_gqa_decode_varlen_mask.benchmark, batch=1, max_cache_seqlen=2048)


def bench_example_triton_sparse_gqa_decode_varlen_indice():
    tilelang.tools.bench.process_func(
        example_triton_sparse_gqa_decode_varlen_indice.benchmark,
        batch=8,
        heads=8,
        heads_kv=4,
        max_cache_seqlen=2048,
        dim=128,
        dim_v=128,
        sparse_ratio=0.8,
        block_size=32)


def bench_example_triton_sparse_gqa_decode_varlen_mask():
    tilelang.tools.bench.process_func(
        example_triton_sparse_gqa_decode_varlen_mask.benchmark,
        batch=8,
        heads=8,
        heads_kv=4,
        max_cache_seqlen=2048,
        dim=128,
        dim_v=128,
        sparse_ratio=0.8,
        block_size=32)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
