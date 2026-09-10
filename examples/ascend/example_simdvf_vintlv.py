"""Verify T.simd.vintlv / vdintlv — interleave and de-interleave on Ascend NPU.

vintlv(x0, y0) → (a0, a1): interleaves even/odd 32-bit lanes of x0, y0
vdintlv(a0, a1) → (x0, y0): de-interleaves back, should recover originals
"""

import torch
import tilelang
import tilelang.ascend.language as T

LANES = 64
N = LANES


def vintlv_kernel(backend="asc"):

    @T.prim_func
    def main(
        X: T.Buffer((N,), "float32"),
        Y: T.Buffer((N,), "float32"),
        X_BAK: T.Buffer((N,), "float32"),
        Y_BAK: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((N,), "float32")
            y_ub = T.alloc_shared((N,), "float32")
            x_bak_ub = T.alloc_shared((N,), "float32")
            y_bak_ub = T.alloc_shared((N,), "float32")

            T.copy(X, x_ub)
            T.copy(Y, y_ub)

            with T.SimdVF():
                if backend == "pto":
                    mask = T.vmi.create_mask(LANES, size=LANES)
                    x0 = T.vmi.vload(x_ub[0], size=LANES)
                    y0 = T.vmi.vload(y_ub[0], size=LANES)
                    a0, a1 = T.vmi.vintlv(x0, y0, mask)
                    x0_back, y0_back = T.vmi.vdintlv(a0, a1, mask)
                    T.vmi.vstore(x0_back, x_bak_ub[0], mask)
                    T.vmi.vstore(y0_back, y_bak_ub[0], mask)
                else:
                    mask_32 = T.simd.pset(32)

                    # Load 64 elements each (one vector register)
                    x0 = T.simd.vld(x_ub[0])
                    y0 = T.simd.vld(y_ub[0])

                    # vintlv: interleave x0 and y0 → a0, a1
                    a0, a1 = T.simd.vintlv(x0, y0)

                    # vdintlv: de-interleave a0, a1 → should recover x0, y0
                    x0_back, y0_back = T.simd.vdintlv(a0, a1)

                    # Store recovered vectors
                    T.simd.vsts(x_bak_ub[0], x0_back, mask_32)
                    T.simd.vsts(y_bak_ub[0], y0_back, mask_32)

            T.copy(x_bak_ub, X_BAK)
            T.copy(y_bak_ub, Y_BAK)

    return main


def simulator_safe_randn(shape, *, dtype, device):
    return torch.randn(shape, dtype=dtype, device="cpu").to(device)


if __name__ == "__main__":
    print("Compiling vintlv_kernel...")
    kernel = tilelang.compile(vintlv_kernel())
    print("Compilation succeeded!")

    print("\n--- Generated Ascend Source ---")
    print(kernel.get_kernel_source())

    device = torch.device("npu")
    x = torch.randn(N, dtype=torch.float32, device=device)
    y = torch.randn(N, dtype=torch.float32, device=device)

    x_bak = torch.empty(N, dtype=torch.float32, device=device)
    y_bak = torch.empty(N, dtype=torch.float32, device=device)

    print("\nRunning kernel on NPU...")
    kernel(x, y, x_bak, y_bak)
    torch.npu.synchronize()

    ok_x = torch.equal(x_bak.cpu(), x.cpu())
    ok_y = torch.equal(y_bak.cpu(), y.cpu())

    assert ok_x and ok_y
    print("\nVerification PASSED! vintlv → vdintlv roundtrip works correctly.")
