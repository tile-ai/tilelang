import tilelang.testing

import example_gemv
import bench_example_gemv


def test_example_gemv():
    example_gemv.main(do_bench=False)



def test_bench_example_gemv():
    bench_example_gemv.main()
if __name__ == "__main__":
    tilelang.testing.main()