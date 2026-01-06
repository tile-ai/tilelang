import tilelang.testing
import example_gqa_decode
import example_mha_inference


def regression_example_gqa_decode():
    tilelang.testing.process_func(example_gqa_decode.run_regression_perf)


def regression_example_mha_inference():
    tilelang.testing.process_func(
        example_mha_inference.run_regression_perf, BATCH=1, H=32, Q_CTX=128, KV_CTX=2048, D_HEAD=128, causal=False
    )


if __name__ == "__main__":
    tilelang.testing.regression()
