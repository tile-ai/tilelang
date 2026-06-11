import os
import pytest
import torch
import tilelang.testing
import example_mla_decode

_is_cutedsl = os.environ.get("TILELANG_TARGET", "").lower() == "cutedsl"
_DEFAULT_DYNAMIC_SMEM_BYTES = 296960


def _has_default_dynamic_smem_capacity():
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    optin = getattr(props, "shared_memory_per_block_optin", 0)
    return optin >= _DEFAULT_DYNAMIC_SMEM_BYTES


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.skipif(_is_cutedsl, reason="CuTeDSL backend does not support alloc_global yet")
@pytest.mark.skipif(
    not _has_default_dynamic_smem_capacity(),
    reason=f"default MLA decode example needs {_DEFAULT_DYNAMIC_SMEM_BYTES} bytes of dynamic shared memory",
)
def test_example_mla_decode():
    example_mla_decode.main()


if __name__ == "__main__":
    tilelang.testing.main()
