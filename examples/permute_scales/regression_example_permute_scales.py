import tilelang.testing
import example_permute_scales


def _record(name: str, **kwargs) -> None:
    tilelang.testing.process_func(example_permute_scales.run_regression_perf, name, **kwargs)


def regression_permute_scales_dsv3_w4a16():
    _record(
        "permute_scales_dsv3_e256_n2048_k7168_g32_w4a16",
        num_experts=256,
        size_n=2048,
        size_k=7168,
        group_size=32,
    )


def regression_permute_scales_dsv3_w4a8():
    _record(
        "permute_scales_dsv3_e256_n2048_k7168_g32_w4a8",
        num_experts=256,
        size_n=2048,
        size_k=7168,
        group_size=32,
        is_a8=True,
    )


def regression_permute_scales_llama4_w4a16():
    _record(
        "permute_scales_llama4_e8_n4096_k8192_g128_w4a16",
        num_experts=8,
        size_n=4096,
        size_k=8192,
        group_size=128,
    )


if __name__ == "__main__":
    tilelang.testing.regression()
