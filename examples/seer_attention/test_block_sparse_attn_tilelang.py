import tilelang.testing

import block_sparse_attn_tilelang
import bench_block_sparse_attn_tilelang


@tilelang.testing.requires_cuda
def test_block_sparse_attn_tilelang():
    block_sparse_attn_tilelang.main()


def test_bench_block_sparse_attn_tilelang():
    bench_block_sparse_attn_tilelang.main()


if __name__ == "__main__":
    tilelang.testing.main()
