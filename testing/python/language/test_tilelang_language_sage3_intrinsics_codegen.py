from unittest import mock

import tilelang
import tilelang.testing
from tilelang import tvm

from examples.sage_attention_sm120.sageattn3_fp4 import sage3_packed_fp4_attention_raw_kernel


MAYFLY_IMAGE_GENERATION_CASES = ((4128, 4224), (4608, 4608))


def _lower_full_scale_case(valid_tokens: int, padded_tokens: int) -> str:
    factory = sage3_packed_fp4_attention_raw_kernel
    old_mode = factory.func.mode
    try:
        factory.func.mode = "lazy"
        with mock.patch(
            "examples.sage_attention_sm120.sageattn3_fp4.driver.get_num_sms",
            return_value=1,
        ):
            program = factory.func(padded_tokens, padded_tokens, valid_tokens, 30, 30, 128)
    finally:
        factory.func.mode = old_mode

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_120a"})
    with tvm.transform.PassContext(), target:
        return tilelang.lower(
            program,
            target=target,
            enable_device_compile=False,
        ).kernel_source


@tilelang.testing.requires_cuda
def test_sage3_sm120_codegen_uses_upstream_intrinsics_without_role_deadlock():
    # Keep Mayfly's complete 1024x1024 image-generation shapes: 4096 latent
    # tokens plus 32/512 context tokens.  The 4128 case follows the official
    # quantization path and pads Q/KV to 4224; 4608 is already aligned.
    for valid_tokens, padded_tokens in MAYFLY_IMAGE_GENERATION_CASES:
        source = _lower_full_scale_case(valid_tokens, padded_tokens)

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
        assert "__sync_thread_partial" not in source


if __name__ == "__main__":
    tilelang.testing.main()
