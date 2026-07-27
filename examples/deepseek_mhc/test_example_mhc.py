import tilelang.testing

from example_mhc_post import main as main_post
from example_mhc_pre import main as main_pre


@tilelang.testing.requires_cuda
def test_mhc_post():
    main_post()


@tilelang.testing.requires_cuda
def test_mhc_pre():
    main_pre()


if __name__ == "__main__":
    tilelang.testing.main()
