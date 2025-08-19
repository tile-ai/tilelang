import tilelang
import os


def test_env_var():
    # test default value
    assert tilelang.env.TILELANG_PRINT_ON_COMPILATION == "1"
    # test forced value
    os.environ["TILELANG_PRINT_ON_COMPILATION"] = "0"
    assert tilelang.env.TILELANG_PRINT_ON_COMPILATION == "0"
    # test forced value with class method
    tilelang.env.TILELANG_PRINT_ON_COMPILATION = "1"
    assert tilelang.env.TILELANG_PRINT_ON_COMPILATION == "1"


if __name__ == "__main__":
    test_env_var()
