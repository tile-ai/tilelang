"""pytest test for example_manual_multibuffer.py.

Verifies the manual + auto multi-buffer kernel against the PyTorch reference on
the NPU. `run_manual_multibuffer(verify=True)` asserts closeness internally, so
a clean return means the kernel is correct.
"""

import pytest

from example_manual_multibuffer import run_manual_multibuffer


TARGETS = ["ascend", pytest.param("pto", marks=pytest.mark.pto)]


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("num_steps", [1, 7])
def test_manual_multibuffer(num_steps, target):
    run_manual_multibuffer(width=4096, num_steps=num_steps, num_rows=4096, verify=True, target=target)


if __name__ == "__main__":
    for s in [1, 7]:
        test_manual_multibuffer(s)
        print(f"PASS: test_manual_multibuffer num_steps={s}")
