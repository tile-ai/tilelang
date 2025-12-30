import tilelang
import tilelang.language as T
import torch
import itertools
import argparse


def get_configs():
    iter_params = dict(
        blk_m=[64, 128, 256],
        threads=[128, 256, 512],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


@tilelang.autotune(configs=get_configs())
@tilelang.jit(out_idx=[1, 2])
def tl_topk(
    M,
    N,
    topk,
    blk_m,
    threads=128,
):
    dtype = T.float32

    @T.prim_func
    def topk_kernel(
        logits: T.Tensor([M, N], dtype),
        topk_gates: T.Tensor([M, topk], dtype),
        topk_indices: T.Tensor([M, topk], T.int32),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
            logits_frag = T.alloc_fragment([blk_m, N], dtype=dtype)
            max_val = T.alloc_fragment([blk_m], dtype=dtype)
            expand_max_idx = T.alloc_fragment([blk_m, N], T.int32)
            max_idx = T.alloc_fragment([blk_m], T.int32)

            T.copy(logits[bx * blk_m, 0], logits_frag)

            for k in T.serial(topk):
                T.fill(expand_max_idx, -1)
                T.reduce_max(logits_frag, max_val, dim=1, clear=True)

                for i, j in T.Parallel(blk_m, N):
                    expand_max_idx[i, j] = T.if_then_else(max_val[i] == logits_frag[i, j], j, expand_max_idx[i, j])

                T.reduce_max(expand_max_idx, max_idx, dim=1, clear=True)

                for i, j in T.Parallel(blk_m, N):
                    logits_frag[i, j] = T.if_then_else(max_val[i] == logits_frag[i, j], -10000.0, logits_frag[i, j])

                for i in T.Parallel(blk_m):
                    topk_gates[bx * blk_m + i, k] = max_val[i]
                    topk_indices[bx * blk_m + i, k] = max_idx[i]

    return topk_kernel


def ref_program(logits, top_k):
    top_k_gates, top_k_indices = logits.topk(top_k, dim=1)

    return top_k_gates, top_k_indices.to(torch.int32)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=320, help="num_tokens")
    parser.add_argument("--N", type=int, default=128, help="num_experts")
    parser.add_argument("--topk", type=int, default=6, help="topk")
    parser.add_argument("--blk_m", type=int, default=64, help="blk_m")
    args = parser.parse_args(argv)
    M, N, topk, blk_m = args.M, args.N, args.topk, args.blk_m

    logits = torch.rand((M, N), device="cuda", dtype=torch.float32)

    kernel = tl_topk(M=M, N=N, topk=topk, blk_m=blk_m)
    tl_gates, tl_indices = kernel(logits)

    torch_gates, torch_indices = ref_program(logits, topk)

    # test accuracy
    torch.testing.assert_close(tl_gates, torch_gates)
    torch.testing.assert_close(tl_indices, torch_indices)

    # profile
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
    tilelang_latency = profiler.do_bench()
    print(f"Tilelang latency: {tilelang_latency}")


def run_regression_perf(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=320, help="num_tokens")
    parser.add_argument("--N", type=int, default=128, help="num_experts")
    parser.add_argument("--topk", type=int, default=6, help="topk")
    parser.add_argument("--blk_m", type=int, default=64, help="blk_m")
    # In benchmark mode, ignore process-wide sys.argv unless an explicit argv is provided.
    args = parser.parse_args(argv or [])
    M, N, topk, blk_m = args.M, args.N, args.topk, args.blk_m

    logits = torch.rand((M, N), device="cuda", dtype=torch.float32)

    kernel = tl_topk(M=M, N=N, topk=topk, blk_m=blk_m)
    tl_gates, tl_indices = kernel(logits)

    torch_gates, torch_indices = ref_program(logits, topk)

    torch.testing.assert_close(tl_gates, torch_gates)
    torch.testing.assert_close(tl_indices, torch_indices)

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Auto)
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    main()
