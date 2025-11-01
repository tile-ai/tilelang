import tilelang.testing

import example_gqa_decode
import example_mha_inference


# TODO(lei): fix the correctness of gqa decode on sm90
@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_le(8, 9)
def test_example_example_gqa_decode():
    example_gqa_decode.main()


def test_example_example_mha_inference():
    example_mha_inference.main(BATCH=1, H=32, Q_CTX=128, KV_CTX=2048, D_HEAD=128, causal=False)


if __name__ == "__main__":
    tilelang.testing.main()
