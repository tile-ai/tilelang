"""End-to-end correctness coverage for dynamic-shape UB allocations."""

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
import torch


@tilelang.jit(target="ascend", out_idx=-1)
def _dynamic_allocation():
    n = T.dynamic("n")

    @T.prim_func
    def copy(src: T.Tensor((n,), "float32"), dst: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((T.ceildiv(n, 64) * 64,), "float32")
            T.copy(src, ub[:n])
            T.copy(ub[:n], dst)

    return copy


@tilelang.jit(
    target="ascend",
    out_idx=None,
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True},
)
def _tail_fill():
    """Zero the invalid tail of a partial UB tile, then reduce it in a SimtVF.

    The fill region is dynamically shaped, so it lowers to an element-wise
    scalar store loop on PIPE_S. It used to be reported as a PIPE_V task, which
    dropped the S->V handshake with the vector consumer and let the reduction
    read stale UB contents.
    """
    tile, e, ntile = 16, 32, 2
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
    e, ntile, seq_len = 32, 2, 17
    device = torch.device("npu")
    kernel = _tail_fill()

    x = torch.ones((seq_len, e), dtype=torch.float32, device=device)
    out = torch.empty((ntile, e), dtype=torch.float32, device=device)
    kernel(x, out)
    torch.npu.synchronize()

    expected = torch.empty((ntile, e), dtype=torch.float32)
    expected[0].fill_(16.0)
    expected[1].fill_(1.0)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_dynamic_ub_allocation():
    kernel = _dynamic_allocation()
    for size in (1, 65):
        src = torch.arange(size, dtype=torch.float32, device="npu")
        torch.testing.assert_close(kernel(src), src, rtol=0, atol=0)
