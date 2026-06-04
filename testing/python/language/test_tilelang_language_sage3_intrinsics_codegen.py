from unittest import mock

import tilelang
import tilelang.testing
from tilelang import tvm

from examples.sage_attention_sm120.sageattn3_fp4 import sage3_packed_fp4_attention_raw_kernel


@tilelang.testing.requires_cuda
def test_sage3_sm120_codegen_uses_upstream_intrinsics_without_role_deadlock():
    factory = sage3_packed_fp4_attention_raw_kernel
    old_mode = factory.func.mode
    try:
        factory.func.mode = "lazy"
        with mock.patch(
            "examples.sage_attention_sm120.sageattn3_fp4.driver.get_num_sms",
            return_value=1,
        ):
            program = factory.func(128, 128, 128, 1, 1, 128)
    finally:
        factory.func.mode = old_mode

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_120a"})
    with tvm.transform.PassContext(), target:
        source = tilelang.lower(
            program,
            target=target,
            enable_device_compile=False,
        ).kernel_source

    assert "tl::sm120_mma_sync_blockscaled" in source
    assert "SM120MmaBlockScaledKind::kMxf4nvf4" in source
    assert "tl::tma_load(" in source
    assert "tl::tma_store(" in source
    assert "tl::warpgroup_reg_dealloc<24>()" in source
    assert "tl::warpgroup_reg_alloc<240>()" in source
    assert "make_float2(" in source
    assert "= float2(" not in source
    # The only CTA-wide barrier is the legal one after mbarrier initialization.
    assert source.count("__syncthreads()") == 1


if __name__ == "__main__":
    tilelang.testing.main()
