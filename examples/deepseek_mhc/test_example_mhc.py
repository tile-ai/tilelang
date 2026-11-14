import tilelang.testing

from example_mhc_bwd import check_unconverged
from example_mhc_post import main as main_post
from example_mhc_pre import main as main_pre


@tilelang.testing.requires_cuda
def test_mhc_post():
    main_post()


@tilelang.testing.requires_cuda
def test_mhc_pre():
    main_pre()


@tilelang.testing.requires_cuda
def test_mhc_bwd_unconverged():
    check_unconverged()


if __name__ == "__main__":
    tilelang.testing.main()
