import pytest

from tilelang import tvm
from tilelang.jit.adapter.wrapper import TLCUDASourceWrapper


def test_adapter_source_wrapper_requires_lowered_modules():
    with pytest.raises(ValueError, match="missing: device_mod, host_mod"):
        TLCUDASourceWrapper(
            scheduled_ir_module=tvm.IRModule(),
            source="",
            target=tvm.target.Target("cuda"),
        )


if __name__ == "__main__":
    pytest.main([__file__])
