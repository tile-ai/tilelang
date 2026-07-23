import tilelang.testing

import block_causal_attention
import block_causal_attention_varlen


@tilelang.testing.requires_cuda
def test_block_causal_attention_fixed():
    block_causal_attention.test_block_causal_attention_all_block_sizes()


@tilelang.testing.requires_cuda
def test_block_causal_attention_varlen():
    block_causal_attention_varlen.test_block_causal_attention_varlen()


if __name__ == "__main__":
    tilelang.testing.main()
