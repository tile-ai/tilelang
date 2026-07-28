import torch.nn.functional as F

import tilelang
import tilelang.language as T


@tilelang.jit()
def pad():
    @T.prim_func()
    def pad(inp_0: T.Buffer((1, 8, 16, 16), "float16"), PadInput: T.Buffer((1, 8, 18, 18), "float16")):
        with T.Kernel(8, threads=324) as (ax0,):
            for i, j in T.Parallel(18, 18):
                PadInput[0, ax0, i, j] = T.if_then_else(
                    i >= 1 and i < 17 and j >= 1 and j < 17, inp_0[0, ax0, i - 1, j - 1], T.float16(0.0)
                )

    return pad


def main():
    kernel = pad()

    import torch

    a = torch.randn(1, 8, 16, 16).cuda().half()
    c = torch.randn(1, 8, 18, 18).cuda().half()

    kernel(a, c)

    ref_c = F.pad(a, (1, 1, 1, 1))

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("All check passed.")


if __name__ == "__main__":
    main()
