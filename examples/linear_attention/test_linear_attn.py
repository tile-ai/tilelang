import tilelang.testing

import example_linear_attn_fwd
import example_linear_attn_bwd
import bench_linear_attn


@tilelang.testing.requires_cuda
def test_example_linear_attn_fwd():
    example_linear_attn_fwd.main()


@tilelang.testing.requires_cuda
def test_example_linear_attn_bwd():
    example_linear_attn_bwd.main()



def test_bench_linear_attn():
    bench_linear_attn.main()
if __name__ == "__main__":
    tilelang.testing.main()