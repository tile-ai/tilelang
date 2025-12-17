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
def _cumsum_view_infer_layout(hidden):
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def buggy_kernel(x: T.Tensor[(num_tokens, hidden), T.float]):
        with T.Kernel(num_tokens, threads=128) as pid:
            smem = T.alloc_shared((hidden,), dtype=T.float32)
            T.copy(x[pid, :], smem)
            T.cumsum(T.view(smem, (1, hidden)), dim=1)

    return buggy_kernel


def test_cumsum_view_infer_layout():
    hidden = 128
    x = torch.randn(1, hidden, device="cuda", dtype=torch.float)
    kernel = _cumsum_view_infer_layout(hidden)
    kernel(x)


if __name__ == "__main__":
    tilelang.testing.main()
