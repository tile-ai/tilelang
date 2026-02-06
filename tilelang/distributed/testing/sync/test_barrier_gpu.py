import tilelang
import tilelang.language as T
from tilelang.carver.arch import driver
import torch
import argparse

tilelang.disable_cache()


@tilelang.jit(out_idx=-1, pass_configs={"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True})
def get_test_barrier_gpu_kernel(num_blocks: int, threads: int):
    @T.prim_func
    def main(
        A: T.Tensor([threads], "int32"),
        bar: T.Tensor([1], "uint32"),  # TODO(wt): auto alloc global bar
        B: T.Tensor([num_blocks, threads], "int32"),
    ):
        with T.Kernel(num_blocks, threads=threads) as bid:
            tid = T.get_thread_binding()
            T.init_barrier_gpu(bar, num_blocks)

            b = T.alloc_shared([threads], "int32")
            val = T.alloc_local([1], "int32")
            val[0] = 1
            T.atomic_add(A[tid], val[0])

            T.sync_barrier_gpu(bar)

            T.copy(A, b)
            T.copy(b, B[bid, :])

    return main


@tilelang.jit(out_idx=-1, pass_configs={"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True})
def test_sync_grid_kernel(num_blocks: int, threads: int):
    @T.prim_func
    def main(
        A: T.Tensor([threads], "int32"),
        bar: T.Tensor([1], "uint32"),  # TODO(wt): auto alloc global bar
        B: T.Tensor([num_blocks, threads], "int32"),
    ):
        with T.Kernel(num_blocks, threads=threads) as bid:
            tid = T.get_thread_binding()

            b = T.alloc_shared([threads], "int32")
            val = T.alloc_local([1], "int32")
            val[0] = 1
            T.atomic_add(A[tid], val[0])

            T.sync_grid(bar)

            T.copy(A, b)
            T.copy(b, B[bid, :])

    return main


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


def test_sync_grid_gpu(num_blocks: int = 64, threads: int = 128, print_source: bool = False):
    kernel = test_sync_grid_kernel(num_blocks, threads)
    input = torch.zeros(threads, dtype=torch.int32, device="cuda")
    bar = torch.zeros(1, dtype=torch.uint32, device="cuda")
    if print_source:
        print(kernel.get_kernel_source())
    print("Compilation done, start running...")

    output = kernel(input, bar)

    assert torch.all(output == num_blocks)
    print("Check passed✅")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--print-source", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.blocks <= driver.get_num_sms(), (
        f"Launched {args.blocks} blocks, which is larger than the number of SM ({driver.get_num_sms()}) on the current device and may cause deadlock!"
    )
    test_barrier_gpu(args.blocks, args.threads, args.print_source)
    test_sync_grid_gpu(args.blocks, args.threads, args.print_source)
