import tilelang.tools.bench
import example_gqa_decode
import example_mha_inference


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_le(8, 9)
def bench_example_gqa_decode():
    tilelang.tools.bench.process_func(example_gqa_decode.main)


def bench_example_mha_inference():
    tilelang.tools.bench.process_func(example_mha_inference.main)


def main():
    tilelang.tools.bench.main()


if __name__ == "__main__":
    main()
