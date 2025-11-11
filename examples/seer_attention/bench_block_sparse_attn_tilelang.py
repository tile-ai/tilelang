import tilelang.tools.bench
import block_sparse_attn_tilelang

@tilelang.testing.requires_cuda
def bench_block_sparse_attn_tilelang():
    tilelang.tools.bench.process_func(block_sparse_attn_tilelang.main)

def main():
    tilelang.tools.bench.main()

if __name__ == "__main__":
    main()
