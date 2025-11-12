import tilelang.testing

import example_linear_attn_fwd
import example_linear_attn_bwd


@tilelang.testing.requires_cuda
def test_example_linear_attn_fwd():
    example_linear_attn_fwd.main()


@tilelang.testing.requires_cuda
def test_example_linear_attn_bwd():
    example_linear_attn_bwd.main()


if __name__ == "__main__":
    tilelang.testing.main()
