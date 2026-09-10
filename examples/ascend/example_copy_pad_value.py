"""Ascend GM->UB copy padding: T.copy(pad_value=) and T.copy(data_select=).

An Ascend MTE GM->UB copy requires each row to be a multiple of 32B. When the
copied region's row is not 32B-aligned, T.copy can right-pad the row tail up to
the next 32B boundary and fill the pad lanes with a chosen value. Two opt-in
modes (mutually exclusive):

  1. pad_value=v      -- this copy sets the fill value AND pads. Emits
                         SetPadValue<T>(v) immediately before the padded copy.
  2. data_select=True -- this copy pads but reuses whatever the hardware pad
                         register already holds. The caller sets it once via
                         T.ascend_set_copy_pad_value(v, dtype=...) and can then
                         reuse it across several copies without re-setting.

Both require the destination UB buffer to be over-allocated to (at least) the
32B-aligned row width, so the padded rows do not overlap. Here we copy N=30
float32 columns (120B, not 32B-aligned) into a UB buffer of 32 columns (128B):
each row's tail lanes [30:32] are filled with the pad value.
"""

import argparse

import torch
import tilelang
import tilelang.ascend.language as T


def copy_pad_value(M, N, N_pad, fill):
    """T.copy(pad_value=fill): the copy sets the fill value and pads the tail."""

    @T.prim_func
    def main(
        A: T.Tensor((M, N), T.float32),
        B: T.Tensor((M, N_pad), T.float32),
    ):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((M, N_pad), T.float32)
            # 120B rows -> right-padded to 128B, tail lanes filled with `fill`.
            T.copy(A[:, :], a_ub[:, :N], pad_value=fill)
            T.copy(a_ub[:, :], B[:, :])

    return main


def copy_data_select(M, N, N_pad, fill):
    """T.copy(data_select=True): reuse a pad register set once beforehand."""

    @T.prim_func
    def main(
        A: T.Tensor((M, N), T.float32),
        B: T.Tensor((M, N_pad), T.float32),
    ):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((M, N_pad), T.float32)
            # Set the pad register once; the copy below reuses it (no SetPadValue
            # is emitted by the copy itself).
            T.ascend_set_copy_pad_value(fill, dtype="float32")
            T.copy(A[:, :], a_ub[:, :N], data_select=True)
            T.copy(a_ub[:, :], B[:, :])

    return main


def _check(program, M, N, N_pad, fill, target="ascend"):
    device = torch.device("npu")
    kernel = tilelang.compile(program, target=target, out_idx=-1)
    a = torch.randn(M, N, dtype=torch.float32, device=device)
    out = kernel(a)
    torch.npu.synchronize()

    # Valid region matches the source; padded tail equals the fill value.
    torch.testing.assert_close(out[:, :N], a)
    expected_tail = torch.full((M, N_pad - N), fill, dtype=torch.float32, device=device)
    torch.testing.assert_close(out[:, N:], expected_tail)
    return out


def run(target="ascend"):
    M, N, N_pad, fill = 4, 30, 32, -1.0

    print(f"--- {target} pad_value mode ---")
    _check(copy_pad_value(M, N, N_pad, fill), M, N, N_pad, fill, target)
    print("PASS")

    print(f"--- {target} data_select mode ---")
    _check(copy_data_select(M, N, N_pad, fill), M, N, N_pad, fill, target)
    print("PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GM-to-UB copy padding examples.")
    parser.add_argument("--target", choices=["ascend"], default="ascend")
    run(parser.parse_args().target)
