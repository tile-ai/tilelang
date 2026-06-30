import tilelang.testing

import block_causal_attention


@tilelang.testing.requires_cuda
def test_block_causal_attention():
    block_causal_attention.main()


if __name__ == "__main__":
    tilelang.testing.main()
