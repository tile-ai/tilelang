import tilelang
import tilelang.language as T
import torch
import tilelang.testing


@tilelang.jit(
    out_idx=-1,
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def get_test_barrier_gpu_kernel(num_blocks: int, threads: int):
    @T.prim_func
    def main(
        A: T.Tensor([threads], T.int32),
        bar: T.Tensor([1], T.uint32),  # TODO(wt): auto alloc global bar
        B: T.Tensor([num_blocks, threads], T.int32),
    ):
        with T.Kernel(num_blocks, threads=threads) as bid:
            tid = T.get_thread_binding()
            T.init_barrier_gpu(bar, num_blocks)

            b = T.alloc_shared([threads], T.int32)
            val = T.alloc_local([1], T.int32)
            val[0] = 1
            T.atomic_add(A[tid], val[0])

            T.sync_barrier_gpu(bar)

            T.copy(A, b)
            T.copy(b, B[bid, :])

    return main


@tilelang.testing.requires_cuda
def test_barrier_gpu(num_blocks: int = 64, threads: int = 128, print_source: bool = False):
    kernel = get_test_barrier_gpu_kernel(num_blocks, threads)
    input = torch.zeros(threads, dtype=torch.int32, device="cuda")
    bar = torch.zeros(1, dtype=torch.uint32, device="cuda")
    if print_source:
        print(kernel.get_kernel_source())
    print("Compilation done, start running...")

    output = kernel(input, bar)

    assert torch.all(output == num_blocks)
    print("Check passed✅")


if __name__ == "__main__":
    test_barrier_gpu()
