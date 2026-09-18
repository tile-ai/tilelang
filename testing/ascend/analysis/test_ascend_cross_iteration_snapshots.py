"""Cross-iteration WAR dependencies through normalized condition snapshots."""

import re

import pytest
import tilelang.ascend.language as T
from tilelang.engine.lower import lower
from tilelang.ascend.language import simd as S


def _branch_switch_program(total_tasks, split, tile, num_cores):
    @T.prim_func
    def main(
        A: T.Tensor((total_tasks, tile), "float32"),
        B: T.Tensor((2, tile), "float32"),
        C: T.Tensor((total_tasks, tile), "float32"),
        D: T.Tensor((total_tasks, 2, tile), "float32"),
    ):
        with T.Kernel(num_cores) as bx:
            ub = T.alloc_shared((tile,), "float32")
            out = T.alloc_shared((tile,), "float32")
            for i in T.Persistent([total_tasks], num_cores, bx, group_size=1):
                # Preserve this pipe order: the later execution phase appears
                # first in the source. A wait guarding the else-branch load
                # cannot protect the first load in the nested loop above it.
                with T.Stage(0):
                    if i >= split:
                        for j in T.serial(2):
                            T.copy(B[j, :], ub)
                            with T.SimdVF():
                                for k in range(tile // 64):
                                    value = S.vld(ub[k * 64])
                                    S.vsts(out[k * 64], S.vmuls(value, 2.0))
                            T.copy(out, D[i, j, :])
                    else:
                        T.copy(A[i, :], ub)
                        with T.SimdVF():
                            for k in range(tile // 64):
                                value = S.vld(ub[k * 64])
                                S.vsts(out[k * 64], S.vmuls(value, 2.0))
                        T.copy(out, C[i, :])

    return main


@pytest.mark.parametrize("split", [1, 4])
def test_branch_switch_lower_preserves_cross_iteration_wait(split):
    source = lower(_branch_switch_program(7, split, 8192, 3), target="ascend").kernel_source
    # Ignore SIMD function definitions and initialization notifications.
    kernel = source[source.index('extern "C"') :]
    body = kernel[kernel.index("for (") :]
    nested_loop = body.index("for (", body.index("for (") + 1)
    first_load = body.index("asc_copy_gm2ub_align")
    second_load = body.index("asc_copy_gm2ub_align", first_load + 1)
    last_compute = re.search(r"\b\w+_simd_vf_\d+\([^;]*\);", body[second_load:])
    assert last_compute is not None
    read_done = second_load + last_compute.end()
    store = body.index("asc_copy_ub2gm_align", read_done)
    # Match the event released after the earlier phase reads UB, rather than
    # accepting the nested loop's unrelated V->MTE2 event or a final drain.
    notify = re.search(
        r"asc_sync_notify\(PIPE_V, PIPE_MTE2, ([^;]+)\);",
        body[read_done:store],
    )
    assert notify is not None
    wait = f"asc_sync_wait(PIPE_V, PIPE_MTE2, {notify.group(1)});"
    assert wait in body[:nested_loop], "The previous iteration's UB read must complete before entering the nested loop that overwrites UB"
