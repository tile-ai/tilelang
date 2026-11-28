import pytest


@pytest.fixture
def cann_91():
    from tilelang.contrib.bisheng import get_cann_home, get_cann_version

    if not get_cann_version(get_cann_home()).startswith("9.1."):
        pytest.skip("This regression exercises the CANN 9.1.x compatibility path")
