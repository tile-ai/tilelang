"""Ascend vector addition using frontend-provided AutoSchedule stages."""

import torch
import tilelang
import tilelang.language as T


NUM_BLOCKS = 1
NUM_THREADS = 128
TILE_ELEMS = 1024


def manual_schedule_vector_add(num_tiles=4):
    n = NUM_BLOCKS * TILE_ELEMS * num_tiles

    @T.prim_func
    def main(
        A: T.Tensor((n,), "float32"),
        B: T.Tensor((n,), "float32"),
        C: T.Tensor((n,), "float32"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            a_ub = T.alloc_shared((TILE_ELEMS,), "float32")
            b_ub = T.alloc_shared((TILE_ELEMS,), "float32")
            c_ub = T.alloc_shared((TILE_ELEMS,), "float32")

            for tile in T.Pipelined(
                num_tiles,
                num_stages=2,
                annotations={"enable_offset": True},
            ):
                begin = (tile * NUM_BLOCKS + bx) * TILE_ELEMS
                end = begin + TILE_ELEMS

                # A Stage scope may live under control flow. The two branches
                # intentionally reverse their MTE2 issue order while producing
                # the same values.
                if tile % 2 == 0:
                    with T.Stage(0):
                        T.copy(A[begin:end], a_ub)
                        T.copy(B[begin:end], b_ub)
                else:
                    with T.Stage(0):
                        T.copy(B[begin:end], b_ub)
                        T.copy(A[begin:end], a_ub)
                # T.Stage is listed first so it wraps the complete SimtVF task.
                with T.Stage(1), T.SimtVF(threads=NUM_THREADS):
                    for i in T.Parallel(TILE_ELEMS):
                        c_ub[i] = a_ub[i] + b_ub[i]
                with T.Stage(2):
                    T.copy(c_ub, C[begin:end])

    return main


def ref_program(a, b):
    return a + b


if __name__ == "__main__":
    num_tiles = 4
    n = NUM_BLOCKS * TILE_ELEMS * num_tiles
    kernel = tilelang.compile(
        manual_schedule_vector_add(num_tiles),
        target="ascend",
        out_idx=-1,
    )

    device = torch.device("npu")
    a = torch.randn(n, dtype=torch.float32, device=device)
    b = torch.randn(n, dtype=torch.float32, device=device)
    result = kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(result, ref_program(a, b), rtol=0, atol=0)
    print("PASS: manual schedule vector add")
