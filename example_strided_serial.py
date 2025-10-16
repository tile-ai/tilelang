import argparse
import os
import sys
sys.path.append("/weka-hg/prod/deepseek/permanent/wanglei/tilelang")
import tilelang
import tilelang.language as T
import tvm


def build_cuda_kernel(numel: int, step: int):
    if tvm.get_global_func("target.build.cuda", allow_missing=True) is None:
        raise RuntimeError("Current TVM build does not contain CUDA codegen (USE_CUDA=ON).")

    threads = 128
    blocks = (numel + threads - 1) // threads
    print("numel", numel, "step", step, "threads", threads, "blocks", blocks)
    @T.prim_func
    def strided_fill(out: T.Tensor((numel,), "float32"), value: T.float32):
        with T.Kernel(blocks, threads=threads) as bx:
            tx = T.get_thread_binding(0)
            start = bx * threads + tx
            for idx in T.serial(start, numel, step=step):
                out[idx] = value

    print(strided_fill)
    kernel = tilelang.compile(strided_fill)
    return kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1024)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--value", type=float, default=1.0)
    args, _ = parser.parse_known_args()

    kernel = build_cuda_kernel(args.numel, args.step)
    print("Generated CUDA kernel:\n", kernel.get_kernel_source())


if __name__ == "__main__":
    main()
