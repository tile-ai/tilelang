"""End-to-end correctness coverage for dynamic-shape UB allocations."""

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
import torch


@tilelang.jit(
    out_idx=None,
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def _rank_tl_ascend(nextn: int):
    num_seqs = T.dynamic("num_seqs")
    num_cores = 64
    vector_length = 64
    num_stages = 2
    block_rows = 4

    @T.prim_func
    def _rank(
        cum_rates: T.Tensor((num_seqs, nextn), "float32"),
        seqlens_q: T.Tensor((num_seqs,), "int32"),
        total_q: T.int32,
    ):
        num_rates = num_seqs * nextn
        num_rates_aligned = T.ceildiv(num_rates, vector_length) * vector_length
        cum_flat = T.reshape(cum_rates, (num_rates,))

        with T.Kernel(num_cores) as core_id:
            out_ub = T.alloc_shared((vector_length,), "int32")
            rates_ub = T.alloc_shared((num_rates_aligned,), "float32")
            T.copy(cum_flat[:num_rates], rates_ub[:num_rates])

            for block in T.Persistent(
                [T.ceildiv(num_seqs, block_rows)],
                num_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                block_start = block * block_rows
                block_length = T.min(block_rows, num_seqs - block_start)
                for row in T.serial(block_rows):
                    if row < block_length:
                        rate = rates_ub[(block_start + row) * nextn]
                        out_ub[row] = T.cast(rate, T.int32) + total_q
                T.copy(out_ub[:block_length], seqlens_q[block_start : block_start + block_length])

    return _rank


def test_rank_dynamic_ub():
    nextn = 5
    total_q = 4
    device = torch.device("npu")
    kernel = _rank_tl_ascend(nextn=nextn)

    for num_seqs in (1, 257):
        cum_rates = torch.arange(num_seqs * nextn, dtype=torch.float32, device=device).reshape(num_seqs, nextn)
        seqlens_q = torch.empty(num_seqs, dtype=torch.int32, device=device)

        kernel(cum_rates, seqlens_q, total_q)
        torch.npu.synchronize()

        expected = cum_rates[:, 0].to(torch.int32) + total_q
        torch.testing.assert_close(seqlens_q, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
