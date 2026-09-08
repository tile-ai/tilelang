"""NPU e2e coverage for PTO VMI tail masks and dtype conversion."""

from __future__ import annotations

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.language as T


LANES = 64
TAIL = 6
N = LANES + TAIL
DEVICE = "npu"


def pto_tail_convert():
    @T.prim_func
    def main(A: T.Buffer((N,), "float32"), C: T.Buffer((N,), "float16")):
        with T.Kernel(1) as _:
            a_ub = T.alloc_shared((LANES,), "float32")
            c_ub = T.alloc_shared((LANES,), "float16")

            T.copy(A[0:LANES], a_ub)
            with T.SimdVF():
                full_mask = T.vmi.create_mask(LANES, size=LANES)
                full_vec = T.vmi.vload(a_ub[0], size=LANES)
                full_cast = T.vmi.vcvt(full_vec, "float16")
                T.vmi.vstore(full_cast, c_ub[0], full_mask)
            T.copy(c_ub, C[0:LANES])

            T.copy(A[LANES:N], a_ub[0:TAIL])
            with T.SimdVF():
                tail_mask = T.vmi.create_mask(TAIL, size=LANES)
                tail_vec = T.vmi.vload(a_ub[0], size=LANES)
                tail_cast = T.vmi.vcvt(tail_vec, "float16")
                T.vmi.vstore(tail_cast, c_ub[0], tail_mask)
            T.copy(c_ub[0:TAIL], C[LANES:N])

    return main


def cpu_input():
    generator = torch.Generator(device="cpu").manual_seed(0)
    return torch.randn(N, dtype=torch.float32, generator=generator)


@pytest.mark.pto
def test_pto_tail_mask_and_conversion_e2e():
    a_cpu = cpu_input()
    expected = a_cpu.to(torch.float16)
    a = a_cpu.to(DEVICE)

    kernel = tilelang.compile(pto_tail_convert(), target="pto", out_idx=-1)
    c = kernel(a)
    torch.npu.synchronize()

    torch.testing.assert_close(c.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
