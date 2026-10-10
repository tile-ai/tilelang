import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit(out_idx=[-1])
def subtract_first_element(width):
    @T.prim_func
    def main(x: T.Tensor((4, width), T.float16), y: T.Tensor((4, width), T.float16)):
        with T.Kernel(4, threads=128) as row:
            for j in T.Parallel(width):
                y[row, j] = x[row, j] - x[row, 0]

    return main


@tilelang.testing.requires_cuda
def test_scalar_read_at_a_wide_row_stride():
    import torch

    # The width is the coefficient of x[row, 0]'s index, one past DataType's 32767 lanes.
    width = 32768
    x = torch.randn(4, width).cuda().half()
    torch.testing.assert_close(subtract_first_element(width)(x), x - x[:, :1])


if __name__ == "__main__":
    tilelang.testing.main()
