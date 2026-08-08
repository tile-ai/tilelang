"""DeepSeek V4 activation-quantization coverage for ROCm targets."""

import sys
from pathlib import Path

import tilelang.testing


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepseek_v4"
sys.path.insert(0, str(EXAMPLE_DIR))

import act_quant  # noqa: E402


@tilelang.testing.requires_rocm
def test_deepseek_v4_activation_quantization():
    tilelang.testing.set_random_seed(0)
    act_quant.test_fp8_act_quant(M=64, N=256, block_size=128)
    act_quant.test_fp4_act_quant(M=64, N=256, block_size=32)
