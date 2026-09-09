import sys
from functools import partial
from pathlib import Path

import tilelang
import tilelang.testing


_EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "flash_decoding"
sys.path.insert(0, str(_EXAMPLE_DIR))

import example_mha_inference as mha  # noqa: E402


@tilelang.testing.requires_rocm
def test_mha_flash_decoding_selects_launchable_tile():
    """Run the resource-selected MHA kernel against its PyTorch reference."""
    batch, heads, query_length, kv_length, head_dim = 1, 2, 128, 512, 128
    kernel, config = mha.compile_flashattn_for_current_device(
        batch,
        heads,
        query_length,
        kv_length,
        head_dim,
        False,
    )

    assert config in ({"block_M": 128, "block_N": 64}, {"block_M": 64, "block_N": 64})
    if "gfx942" in str(kernel.target.attrs.get("mcpu", "")):
        assert config == {"block_M": 64, "block_N": 64}
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    profiler.assert_allclose(partial(mha.ref_program, causal=False), rtol=1e-2, atol=1e-2)
