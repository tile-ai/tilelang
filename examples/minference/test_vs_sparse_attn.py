import tilelang.testing

import example_vertical_slash_sparse_attn
import bench_vs_sparse_attn


@tilelang.testing.requires_cuda
def test_vs_sparse_attn():
    example_vertical_slash_sparse_attn.main(argv=[])



def test_bench_vs_sparse_attn():
    bench_vs_sparse_attn.main()
if __name__ == "__main__":
    tilelang.testing.main()