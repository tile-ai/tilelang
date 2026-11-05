import pytest

import tvm_tirx
from tilelang.jit.adapter.wrapper import TLCPUSourceWrapper


def test_adapter_source_wrapper_requires_lowered_modules():
    with pytest.raises(ValueError, match="missing: device_mod, host_mod"):
        TLCPUSourceWrapper(
            scheduled_ir_module=tvm_tirx.IRModule(),
            source="",
            target=tvm_tirx.target.Target("c"),
        )


if __name__ == "__main__":
    pytest.main([__file__])
