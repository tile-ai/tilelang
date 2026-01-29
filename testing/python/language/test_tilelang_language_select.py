import torch
import tilelang.testing
import tilelang.language as T


@tilelang.jit
def get_select_kernel_1():
    @T.prim_func
    def main(
        A: T.Tensor[(128, 8), T.float32],
        B: T.Tensor[(128, 8), T.float32],
    ):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding(0)
            tmp = T.alloc_var(T.bfloat16)
            tmp = A[tx, 3]

            B[tx, 0] = T.Select(True, A[tx, 0], 0.0)
            B[tx, 1] = T.Select(False, 1.0, A[tx, 1])
            B[tx, 2] = T.Select(T.cast(A[tx, 3], T.bfloat16) == tmp, A[tx, 2], T.cast(tmp, T.float32))
            B[tx, 3] = T.Select(B[tx, 0] != 0.0, T.if_then_else(B[tx, 1] != 0.0, A[tx, 3], 0.0), 0.0)

            for i in T.serial(4):
                B[tx, i + 4] = T.Select(
                    A[tx, 0] == 0, T.if_then_else(T.Select(True, False, True), 1.0, 2.0), T.Select(True, A[tx, i + 4], 3.0)
                )

    return main


def test_select_correctness():
    A = torch.randn((128, 8), dtype=torch.float32, device="cuda")
    B = torch.empty((128, 8), dtype=torch.float32, device="cuda")
    kernel = get_select_kernel_1()

    A = torch.clamp(A, min=1e-4)
    kernel(A, B)
    assert torch.allclose(A, B)


@tilelang.jit
def get_select_kernel_2():
    @T.prim_func
    def main(
        A: T.Tensor[(128, 8), T.float32],
        B: T.Tensor[(128, 8), T.float32],
    ):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding(0)
            tmp = T.alloc_var(T.bfloat16)
            tmp = A[tx, 3]

            B[tx, 0] = T.Select(True, A[tx, 0], 0.0)
            B[tx, 1] = T.Select(False, 1.0, A[tx, 1])
            B[tx, 2] = T.Select(T.cast(A[tx, 3], T.bfloat16) == tmp, A[tx, 2], T.cast(tmp, T.float32))
            B[tx, 3] = T.Select(B[tx, 0] != 0.0, T.Select(B[tx, 1] != 0.0, A[tx, 3], 0.0), 0.0)

            for i in T.serial(4):
                B[tx, i + 4] = T.Select(
                    A[tx, 0] == 0,
                    T.Select(T.Select(True, False, True), 1.0, 2.0),
                    T.Select(T.sin(A[tx, 2]) == 1.0, T.sin(A[tx, 0]), T.cos(A[tx, 1])),
                )

    return main


def test_select_codegen_no_if():
    kernel = get_select_kernel_2()
    source = kernel.get_kernel_source()
    assert "if (" not in source


if __name__ == "__main__":
    tilelang.testing.main()
