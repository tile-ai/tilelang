import pytest


@pytest.fixture(autouse=True)
def downstream_codegen(tilelang_builders):
    yield
