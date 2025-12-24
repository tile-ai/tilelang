"""Compare the outputs of a GPTQ model to a Marlin model.

Note: GPTQ and Marlin do not have bitwise correctness.
As a result, in this test, we just confirm that the top selected tokens of the
Marlin/GPTQ models are in the top 3 selections of each other.

Note: Marlin internally uses locks to synchronize the threads. This can
result in very slight nondeterminism for Marlin. As a result, we re-run the test
up to 3 times to see if we pass.

Run `pytest tests/models/test_marlin.py`.
"""

from conftest import VllmRunner
import os
import argparse

# get the path of the current file
current_file_path = os.path.realpath(__file__)
current_dir = os.path.dirname(current_file_path)

ckpt_path = os.path.join(current_dir, "../models/ckpt_bitnet_b1_58-3B_bitblas")
parser = argparse.ArgumentParser(description="Inference with BitNet")
parser.add_argument(
    "--ckpt_path",
    type=str,
    default=ckpt_path,
    help="Path to the checkpoint",
)

args = parser.parse_args()

ckpt_path = args.ckpt_path
with VllmRunner(
    ckpt_path,
    dtype="half",
    quantization="bitblas",
    # set enforce_eager = False to enable cuda graph
    # set enforce_eager = True to disable cuda graph
    enforce_eager=False,
) as bitnet_model:
    bitbnet_outputs = bitnet_model.generate_greedy(["Hi, tell me about microsoft?"], max_tokens=1024)
    print("bitnet inference:")
    print(bitbnet_outputs[0][0])
    print(bitbnet_outputs[0][1])
