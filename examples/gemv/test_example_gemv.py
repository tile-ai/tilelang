import example_gemv


def test_example_gemv():
    example_gemv.main(do_bench=False)


if __name__ == "__main__":
    test_example_gemv()
