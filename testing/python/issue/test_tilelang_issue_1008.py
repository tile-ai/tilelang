import torch
import tilelang
import tilelang.testing
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def _fill_with_static_region_kernel():
    num_tokens = T.symbolic("num_tokens")

    @T.prim_func
    def buggy_kernel(x: T.Tensor[(num_tokens,), "int64"]):  # noqa: F821
        with T.Kernel(num_tokens, threads=128) as _:
            T.fill(x[0:128], 0)

    return buggy_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def _fill_with_dynamic_region_kernel():
    num_tokens = T.symbolic("num_tokens")

    @T.prim_func
    def buggy_kernel(x: T.Tensor[(num_tokens,), "int64"]):  # noqa: F821
        with T.Kernel(num_tokens, threads=128) as _:
            a, b = T.alloc_var(T.int), T.alloc_var(T.int)
            T.fill(x[a:b], 0)

    return buggy_kernel


def test_fill_with_static_region_kernel():
    kernel = _fill_with_static_region_kernel()
    x = torch.zeros((256,), dtype=torch.int64, device="cuda")
    kernel(x)


def test_fill_with_dynamic_region_kernel():
    kernel = _fill_with_dynamic_region_kernel()
    x = torch.zeros((256,), dtype=torch.int64, device="cuda")
    kernel(x)


if __name__ == "__main__":
    tilelang.testing.main()
