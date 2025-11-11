# ruff: noqa
import tilelang.testing

from example_tilelang_nsa_fwd import main as main_fwd
from example_tilelang_nsa_decode import main as main_fwd_decode
import bench_example_tilelang_nsa


def test_example_tilelang_nsa_fwd():
    main_fwd()


def test_example_tilelang_nsa_fwd_decode():
    main_fwd_decode()


def test_bench_example_tilelang_nsa():
    bench_example_tilelang_nsa.main()


if __name__ == "__main__":
    tilelang.testing.main()
