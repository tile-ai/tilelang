import torch

import tilelang
import tilelang.ascend.language as T
import tilelang.testing


@tilelang.testing.requires_ascend
def test_hf32_rounding_and_disable_update_control_bits():
    @T.prim_func
    def kernel(ctrl: T.Tensor((3,), "int64")):
        with T.Kernel(1), T.Cube():
            T.set_hf32_mode("nearest_zero")  # NEAREST_AWAY in the SDK
            ctrl[0] = T.cast(T.call_extern("uint64", "get_ctrl"), "int64")
            T.set_hf32_mode("nearest_even")
            ctrl[1] = T.cast(T.call_extern("uint64", "get_ctrl"), "int64")
            T.set_hf32_mode(None)
            ctrl[2] = T.cast(T.call_extern("uint64", "get_ctrl"), "int64")

    # Keep the control-register reads and writes on the explicitly chosen Cube core.
    compiled = tilelang.compile(kernel, target="ascend", pass_configs={"tl.enable_auto_schedule": False})
    ctrl = torch.tensor([0, 0, -1], dtype=torch.int64, device="npu")
    compiled(ctrl)
    away, even, disabled = ctrl.cpu().tolist()
    # CTRL bit 46 enables HF32. Bit 47 selects AWAY (1) or EVEN (0).
    assert (away >> 46) & 3 == 3
    assert (even >> 46) & 3 == 1
    assert (disabled >> 46) & 1 == 0
