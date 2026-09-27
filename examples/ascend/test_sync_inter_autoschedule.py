import re

import pytest
import torch

import tilelang
import tilelang.ascend.language as T


NUM_VECTOR_CORES = 64
TILE_ELEMS = 64
INTER_CORE_FLAG = 4


def cross_core_mte3_to_mte2():
    @T.prim_func
    def main(
        values: T.Tensor((NUM_VECTOR_CORES, TILE_ELEMS), T.float32),
        exchange: T.Tensor((NUM_VECTOR_CORES, TILE_ELEMS), T.float32),
        output: T.Tensor((NUM_VECTOR_CORES, TILE_ELEMS), T.float32),
    ):
        with T.Kernel(NUM_VECTOR_CORES) as bx:
            write_ub = T.alloc_shared((TILE_ELEMS,), T.float32)
            read_ub = T.alloc_shared((TILE_ELEMS,), T.float32)

            T.copy(values[bx, :], write_ub)
            T.copy(write_ub, exchange[bx, :])

            # Every Vector core must finish its MTE3 write before any Vector
            # core issues the following MTE2 read from its neighbour's slot.
            T.ascend_sync_inter_arrive("PIPE_MTE3", INTER_CORE_FLAG)
            T.ascend_sync_inter_wait("PIPE_MTE2", INTER_CORE_FLAG)

            peer = (bx + 1) % NUM_VECTOR_CORES
            T.copy(exchange[peer, :], read_ub)
            T.copy(read_ub, output[bx, :])

    return main


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_cross_core_mte3_write_visible_to_mte2_read(target):
    kernel = tilelang.compile(cross_core_mte3_to_mte2(), target=target, out_idx=-1)
    source = kernel.get_kernel_source()

    if target == "ascend":
        mte3_store = source.index("asc_copy_ub2gm")
        arrive = source.index("asc_sync_inter_arrive(PIPE_MTE3, 4);")
        wait = source.index("asc_sync_inter_wait(PIPE_MTE2, 4);")
        mte2_loads = [match.start() for match in re.finditer("asc_copy_gm2ub", source)]
        assert len(mte2_loads) == 2
        assert mte3_store < arrive
        assert wait < mte2_loads[1]
    else:
        assert "tl.ascend_cross_core_set_flag" in source
        assert "tl.ascend_cross_core_wait_flag" in source

    values = torch.arange(
        NUM_VECTOR_CORES * TILE_ELEMS,
        dtype=torch.float32,
        device="npu",
    ).reshape(NUM_VECTOR_CORES, TILE_ELEMS)
    exchange = torch.full_like(values, -1)
    output = kernel(values, exchange)
    torch.npu.synchronize()

    expected = torch.roll(values, shifts=-1, dims=0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    for target in ("ascend", "pto"):
        test_cross_core_mte3_write_visible_to_mte2_read(target)
        print(f"PASS: test_cross_core_mte3_write_visible_to_mte2_read ({target})")
