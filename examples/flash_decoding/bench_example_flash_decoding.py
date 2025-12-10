import tilelang.tools.bench
import example_gqa_decode
import example_mha_inference


def bench_example_gqa_decode():
    tilelang.tools.bench.process_func(example_gqa_decode.benchmark)


def bench_example_mha_inference():
    tilelang.tools.bench.process_func(
        example_mha_inference.benchmark,
        BATCH=1,
        H=32,
        Q_CTX=128,
        KV_CTX=2048,
        D_HEAD=128,
        causal=False)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()
