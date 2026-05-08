import warnings

import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.jit.kernel import JITKernel


def test_disable_tma_lower_pass_context_compat():
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True}):
        assert bool(tvm.transform.PassContext.current().config["tl.disable_tma_lower"])

    with tvm.transform.PassContext(config={"tl.disable_tma_lower": True}):
        assert bool(tvm.transform.PassContext.current().config["tl.disable_tma_lower"])


def test_disable_tma_lower_warns_in_jit_entry():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        JITKernel(
            from_database=True,
            target="c",
            pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True},
        )

    assert any("tl.disable_tma_lower" in str(item.message) and "v0.1.10" in str(item.message) for item in caught)


if __name__ == "__main__":
    tilelang.testing.main()
