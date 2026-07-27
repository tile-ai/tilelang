import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def test_runtime_unknown_sign_vector_negative_index_load():
    @T.prim_func
    def main(A: T.Tensor((1024,), T.float32), B: T.Tensor((4, 4), T.float32)):
        with T.Kernel(1, threads=1) as _:
            for t in T.serial(4):
                B[t, T.Ramp(0, 1, 4)] = A[T.Ramp(t - 2, 1, 4)]

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")

    a = torch.arange(1024, device="cuda", dtype=torch.float32)
    b = kernel(a)
    expected = torch.tensor(
        [
            [1022.0, 1023.0, 0.0, 1.0],
            [1023.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    torch.testing.assert_close(b, expected)


def test_runtime_unknown_sign_vector_negative_index_store():
    @T.prim_func
    def main(B: T.Tensor((4, 4), T.float32), A: T.Tensor((1024,), T.float32)):
        with T.Kernel(1, threads=1) as _:
            for i in T.serial(1024):
                A[i] = T.float32(-1)
            for t in T.serial(4):
                A[T.Ramp(t - 2, 1, 4)] = B[t, T.Ramp(0, 1, 4)]

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")

    b = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
    a = kernel(b)

    expected = torch.full((1024,), -1, device="cuda", dtype=torch.float32)
    expected[0] = b[2, 0]
    expected[1] = b[3, 0]
    expected[2] = b[3, 1]
    expected[3] = b[3, 2]
    expected[4] = b[3, 3]
    expected[1022] = b[0, 0]
    expected[1023] = b[1, 0]

    torch.testing.assert_close(a, expected)


if __name__ == "__main__":
    tilelang.testing.main()
