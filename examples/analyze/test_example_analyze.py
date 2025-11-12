import tilelang.testing
import example_gemm_analyze
import example_conv_analyze


def test_example_gemm_analyze():
    example_gemm_analyze.main()


def test_example_conv_analyze():
    example_conv_analyze.main()


if __name__ == "__main__":
    tilelang.testing.main()
