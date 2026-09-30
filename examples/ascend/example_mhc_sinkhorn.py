"""MHC Sinkhorn forward/backward on Ascend NPU.

Row/column normalization of the combination matrix over a serial repeat loop.
Written in plain TileLang; the AutoSimtVF pass promotes the `T.Parallel` loops.
"""

import torch
import tilelang
import tilelang.ascend.language as T


def mhc_sinkhorn_fwd(hidden_size, token_block_size, repeat, eps):
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def main(
        comb_res_mix: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32],
        comb_res_mix_out: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid_x:
            comb_frag = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)
            row_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)
            col_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)

            T.copy(comb_res_mix[pid_x * token_block_size, 0, 0], comb_frag)

            row_max = T.alloc_fragment((token_block_size, hidden_size), T.float32)
            T.reduce_max(comb_frag, row_max, dim=2)
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                comb_frag[i, j, k] = T.exp(comb_frag[i, j, k] - row_max[i, j])
            T.reduce_sum(comb_frag, row_sum, dim=2)
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                comb_frag[i, j, k] = comb_frag[i, j, k] / row_sum[i, j] + eps

            T.reduce_sum(comb_frag, col_sum, dim=1)
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

            for _ in T.serial(repeat - 1):
                T.reduce_sum(comb_frag, row_sum, dim=2)
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    comb_frag[i, j, k] = comb_frag[i, j, k] / (row_sum[i, j] + eps)

                T.reduce_sum(comb_frag, col_sum, dim=1)
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

            T.copy(comb_frag, comb_res_mix_out[pid_x * token_block_size, 0, 0])

    return main


def mhc_sinkhorn_bwd(hidden_size, token_block_size, repeat, eps):
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def main(
        grad_output: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32],
        x: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32],
        grad_input: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid_x:
            grad_frag = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)
            x_frag = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)

            T.copy(grad_output[pid_x * token_block_size, 0, 0], grad_frag)
            T.copy(x[pid_x * token_block_size, 0, 0], x_frag)

            row_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)
            row_sum2 = T.alloc_fragment((token_block_size, hidden_size), T.float32)
            col_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)
            col_sum2 = T.alloc_fragment((token_block_size, hidden_size), T.float32)
            temp = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)

            xs = T.alloc_shared((repeat * 2, token_block_size, hidden_size, hidden_size), T.float32)
            sums = T.alloc_shared((repeat * 2, token_block_size, hidden_size), T.float32)

            row_max = T.alloc_fragment((token_block_size, hidden_size), T.float32)
            T.reduce_max(x_frag, row_max, dim=2)
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                x_frag[i, j, k] = T.exp(x_frag[i, j, k] - row_max[i, j])
            T.reduce_sum(x_frag, row_sum, dim=2)
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                x_frag[i, j, k] = x_frag[i, j, k] / row_sum[i, j]
            T.copy(x_frag, xs[0, 0, 0, 0])
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                x_frag[i, j, k] = x_frag[i, j, k] + eps
            T.copy(x_frag, xs[1, 0, 0, 0])

            T.reduce_sum(x_frag, col_sum, dim=1)
            T.copy(col_sum, sums[1, 0, 0])
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                x_frag[i, j, k] = x_frag[i, j, k] / (col_sum[i, k] + eps)

            for step in T.serial(repeat - 1):
                T.reduce_sum(x_frag, row_sum, dim=2)
                T.copy(row_sum, sums[step * 2 + 2, 0, 0])
                T.copy(x_frag, xs[step * 2 + 2, 0, 0, 0])
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    x_frag[i, j, k] = x_frag[i, j, k] / (row_sum[i, j] + eps)

                T.reduce_sum(x_frag, col_sum, dim=1)
                T.copy(col_sum, sums[step * 2 + 3, 0, 0])
                T.copy(x_frag, xs[step * 2 + 3, 0, 0, 0])
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    x_frag[i, j, k] = x_frag[i, j, k] / (col_sum[i, k] + eps)

            x_inter = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)
            for inv_step in T.serial(2 * repeat - 1):
                T.copy(xs[2 * repeat - 1 - inv_step, 0, 0, 0], x_inter)
                if inv_step % 2 == 0:
                    T.copy(sums[2 * repeat - 1 - inv_step, 0, 0], col_sum)
                    for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                        temp[i, j, k] = grad_frag[i, j, k] * x_inter[i, j, k]
                    T.reduce_sum(temp, col_sum2, dim=1)
                    for i, k in T.Parallel(token_block_size, hidden_size):
                        col_sum2[i, k] /= col_sum[i, k] + eps
                    for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                        grad_frag[i, j, k] = (grad_frag[i, j, k] - col_sum2[i, k]) / (col_sum[i, k] + eps)
                else:
                    T.copy(sums[2 * repeat - 1 - inv_step, 0, 0], row_sum)
                    for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                        temp[i, j, k] = grad_frag[i, j, k] * x_inter[i, j, k]
                    T.reduce_sum(temp, row_sum2, dim=2)
                    for i, j in T.Parallel(token_block_size, hidden_size):
                        row_sum2[i, j] /= row_sum[i, j] + eps
                    for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                        grad_frag[i, j, k] = (grad_frag[i, j, k] - row_sum2[i, j]) / (row_sum[i, j] + eps)

            T.copy(xs[0, 0, 0, 0], x_inter)
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                temp[i, j, k] = grad_frag[i, j, k] * x_inter[i, j, k]
            T.reduce_sum(temp, row_sum, dim=2)
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                grad_frag[i, j, k] = (grad_frag[i, j, k] - row_sum[i, j]) * x_inter[i, j, k]

            T.copy(grad_frag, grad_input[pid_x * token_block_size, 0, 0])

    return main


def ref_program_sinkhorn_fwd(x, repeat=10, eps=1e-6):
    output = torch.softmax(x, dim=-1) + eps
    output = output / (output.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        output = output / (output.sum(dim=-1, keepdim=True) + eps)
        output = output / (output.sum(dim=-2, keepdim=True) + eps)
    return output


def ref_program_sinkhorn_bwd(grad_output, x, repeat, eps):
    x = x.detach().requires_grad_()
    y = torch.softmax(x, dim=-1) + eps
    y = y / (y.sum(dim=1, keepdim=True) + eps)
    for _ in range(repeat - 1):
        y = y / (y.sum(dim=-1, keepdim=True) + eps)
        y = y / (y.sum(dim=1, keepdim=True) + eps)
    return torch.autograd.grad(y, x, grad_output)[0]


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("npu")

    num_tokens, hidden_size = 64, 4
    token_block_size, repeat, eps = 1, 10, 1e-6
    comb_res_mix = torch.randn((num_tokens, hidden_size, hidden_size), device=device, dtype=torch.float32)
    kernel = tilelang.compile(mhc_sinkhorn_fwd(hidden_size, token_block_size, repeat, eps), target="ascend", out_idx=-1)
    actual = kernel(comb_res_mix)
    torch.npu.synchronize()
    expected = ref_program_sinkhorn_fwd(comb_res_mix, repeat, eps)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    print("PASS: mhc_sinkhorn_fwd")

    num_tokens, hidden_size = 128, 4
    token_block_size, repeat, eps = 32, 2, 1e-6
    grad_output = torch.randn((num_tokens, hidden_size, hidden_size), device=device, dtype=torch.float32)
    x = torch.randn((num_tokens, hidden_size, hidden_size), device=device, dtype=torch.float32)
    kernel = tilelang.compile(mhc_sinkhorn_bwd(hidden_size, token_block_size, repeat, eps), target="ascend", out_idx=-1)
    grad_input = kernel(grad_output, x)
    torch.npu.synchronize()
    expected = ref_program_sinkhorn_bwd(grad_output, x, repeat, eps)
    torch.testing.assert_close(grad_input, expected, rtol=1e-5, atol=2e-5)
    print("PASS: mhc_sinkhorn_bwd")
