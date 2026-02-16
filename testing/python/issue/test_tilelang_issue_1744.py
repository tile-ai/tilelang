import torch
import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def _buggy_kernel(S: T.Tensor((8), T.bfloat16), D: T.Tensor((4, 64), T.bfloat16)):
    with T.Kernel(1, threads=128):
        S_shared = T.alloc_shared((8), T.bfloat16)
        S_fragment = T.alloc_fragment((8), T.float32)
        D_shared = T.alloc_shared((4, 64), T.bfloat16)

        T.copy(S, S_shared)
        T.copy(S_shared, S_fragment)
        for k in T.serial(64):
            for i in T.Parallel(4):
                D_shared[i, k] = S_fragment[i]
        T.copy(D_shared, D)


@tilelang.testing.requires_cuda
def test():
    test_S = torch.randn((8), dtype=torch.bfloat16, device="cuda")
    test_D = torch.zeros((4, 64), dtype=torch.bfloat16, device="cuda")
    _buggy_kernel(test_S, test_D)
    ref_D = test_S[:4].view(4, 1).repeat(1, 64)
    torch.testing.assert_close(test_D, ref_D)


if __name__ == "__main__":
    test()
