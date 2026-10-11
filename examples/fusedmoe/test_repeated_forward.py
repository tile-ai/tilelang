import pytest
import torch

from example_fusedmoe_tilelang import MoE, moe_forward_tilelang_routed, moe_forward_tilelang_shared
from example_fusedmoe_torch import generate_input, ref_kernel


@pytest.mark.parametrize("top_k", [1, 2])
def test_repeated_forward(top_k):
    x, weights, config = generate_input(256, 128, 2, 1, top_k, 1, 256, seed=81394)
    shared = moe_forward_tilelang_shared.compile(dhidden=256, dexpert=128, num_tokens=256, dtype="float16")
    routed = moe_forward_tilelang_routed.compile(
        dhidden=256,
        dexpert=128,
        n_routed_experts=2,
        dtype="float16",
        group_sum=256 * top_k,
        group_count=2,
        block_token=128,
        block_dhidden=128,
        block_dexpert=128,
        threads=256,
        num_stages=1,
    )
    model = MoE(config, shared, routed, weights)
    for call, inputs in enumerate((x, x, -x)):
        expected = ref_kernel((inputs, weights, config))
        actual = model(inputs)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2, msg=lambda msg, call=call: f"call {call + 1}: {msg}")
