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


@tilelang.jit(
    out_idx=None,
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True},
)
def _tail_fill_tl_ascend():
    """Zero the invalid tail of a partial UB tile, then reduce it in a SimtVF.

    The fill region is dynamically shaped, so it lowers to an element-wise
    scalar store loop on PIPE_S. It used to be reported as a PIPE_V task, which
    dropped the S->V handshake with the vector consumer and let the reduction
    read stale UB contents.
    """
    tile, e, ntile = 64, 384, 3
    seq_len = T.dynamic("seq_len")

    @T.prim_func
    def _tail_fill(
        x: T.Tensor((seq_len, e), "float32"),
        out: T.Tensor((ntile, e), "float32"),
    ):
        with T.Kernel(1):
            buf = T.alloc_shared((tile, e), "float32")
            acc = T.alloc_shared((ntile, e), "float32")
            for t in T.serial(T.ceildiv(seq_len, tile)):
                valid = T.min(tile, seq_len - t * tile)
                T.copy(x[t * tile : t * tile + valid, :], buf[:valid, :])
                if valid < tile:
                    T.fill(buf[valid:tile, :], T.float32(0.0))
                with T.SimtVF(threads=e):
                    for col in T.Parallel(e):
                        total = T.alloc_var(T.float32, init=T.float32(0.0))
                        for row in T.unroll(tile):
                            total = total + buf[row, col]
                        acc[t, col] = total
            T.copy(acc, out)

    return _tail_fill


def test_tail_fill_with_dynamic_ub_region():
    e, ntile, seq_len = 384, 3, 129
    device = torch.device("npu")
    kernel = _tail_fill_tl_ascend()

    x = torch.ones((seq_len, e), dtype=torch.float32, device=device)
    out = torch.empty((ntile, e), dtype=torch.float32, device=device)
    kernel(x, out)
    torch.npu.synchronize()

    expected = torch.empty((ntile, e), dtype=torch.float32)
    expected[0].fill_(64.0)
    expected[1].fill_(64.0)
    expected[2].fill_(1.0)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


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
