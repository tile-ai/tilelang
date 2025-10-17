import argparse
import numpy as np

import tilelang.language as T


def build_cuda_kernel(numel: int, step: int):
    if tvm.get_global_func("target.build.cuda", allow_missing=True) is None:
        raise RuntimeError("Current TVM build does not contain CUDA codegen (USE_CUDA=ON).")

    threads = 128
    blocks = (numel + threads - 1) // threads
    print("numel", numel, "step", step, "threads", threads, "blocks", blocks)

    @T.prim_func
    def strided_fill(out: T.Buffer((numel,), "float32"), value: T.float32):
        tx = T.thread_binding(0, threads, thread="threadIdx.x")
        start = bx * threads + tx
        for idx in T.serial(start, numel, step=step):
            out[idx] = value

    print(strided_fill)
    return strided_fill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1024)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--value", type=float, default=1.0)
    args, _ = parser.parse_known_args()

    rt_mod = build_cuda_kernel(args.numel, args.step)
    print("Generated CUDA kernel:\n", rt_mod.imported_modules[0].get_source())

    if not tvm.cuda(0).exist:
        print(
            "CUDA device 0 not visible; skipping kernel launch. Set CUDA_VISIBLE_DEVICES or run on a GPU node."
        )
        return

    dev = tvm.cuda(0)
    out = tvm.nd.array(np.full(args.numel, -1.0, dtype="float32"), device=dev)
    rt_mod(out, np.float32(args.value))

    result = out.numpy()
    print("Result:", result)
    stride_idx = np.arange(0, args.numel, args.step)
    assert np.allclose(result[stride_idx], args.value)
    untouched = np.setdiff1d(np.arange(args.numel), stride_idx)
    assert np.allclose(result[untouched], -1.0)


if __name__ == "__main__":
    main()
