import tilelang.tools.bench
import example_tilelang_block_sparse_attn
import example_tilelang_sparse_gqa_decode_varlen_indice
import example_tilelang_sparse_gqa_decode_varlen_mask

def bench_example_tilelang_block_sparse_attn():
    tilelang.tools.bench.process_func(example_tilelang_block_sparse_attn.benchmark)


def bench_example_tilelang_sparse_gqa_decode_varlen_indice():
    tilelang.tools.bench.process_func(
        example_tilelang_sparse_gqa_decode_varlen_indice.benchmark, batch=1, max_cache_seqlen=2048)


def bench_example_tilelang_sparse_gqa_decode_varlen_mask():
    tilelang.tools.bench.process_func(
        example_tilelang_sparse_gqa_decode_varlen_mask.benchmark, batch=1, max_cache_seqlen=2048)



if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
