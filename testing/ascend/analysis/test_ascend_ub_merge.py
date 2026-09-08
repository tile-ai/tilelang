"""UB-merge (MergeUBAllocations) offset regression tests.

Both tests inspect the generated Ascend source and assert where the pass places
each T.alloc_shared staging buffer inside the merged `buf_dyn_shmem` arena.
No NPU execution is required — the decision is a compile-time property.

1. test_ub_merge_gemm_shared_offset:
   The Cube+Vector gemm example (examples/ascend/example_gemm_ub_merge.py) uses
   two staging buffers buf_a / buf_b whose lifetimes are serialized by a
   cross-core handshake (AIV signals "buf_a consumed" before AIC writes buf_b).
   Because both buffers are loop-free (accessed only at block top level), the
   cross-core set->wait ordering is trusted and the pass reuses one UB slot for
   both -> identical offset.

2. test_ub_merge_wait_alias_distinct_offset:
   Regression guard for the wait_flag-as-release bug. Two buffers used in
   disjoint phases (z_ub stored via MTE3, then val_ub loaded via MTE2 into the
   "freed" slot) are separated only by a wait_flag that the fill store never
   actually released to. The pass must NOT alias them; they must land at
   distinct offsets. If aliased, the fill loop's MTE3 read of z_ub races the
   val loop's MTE2 overwrite and corrupts fill_out at runtime.
"""

import os
import re
import sys

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.engine.lower import lower

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "ascend")),
)


def _shared_offsets(src: str, elem_ctype: str):
    """Distinct element offsets at which `buf_dyn_shmem` is indexed as elem_ctype*."""
    pat = re.escape(elem_ctype) + r"\*\)buf_dyn_shmem\)\[(\d+)\]"
    return sorted({int(m) for m in re.findall(pat, src)})


def test_ub_merge_gemm_shared_offset():
    from example_gemm_ub_merge import gemm_ub_merge

    with tilelang.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: False}):
        artifact = lower(gemm_ub_merge(256, 256, 128), target="ascend")
    offsets = _shared_offsets(artifact.kernel_source, "float")
    # Serialized lifetimes -> the two staging buffers share one UB slot.
    assert offsets == [0], f"buf_a / buf_b not merged; offsets={offsets}"


def _wait_alias_program(num_cores: int = 64, BB: int = 256):
    N = T.dynamic("n")
    M = T.dynamic("m")

    @T.prim_func
    def main(
        val: T.Tensor((N,), "int32"),
        out: T.Tensor((N, M), "int32"),
        fill_out: T.Tensor((N, M), "int32"),
    ):
        total_rows = N * M
        fill_1d = T.reshape(fill_out, (total_rows,))

        with T.Kernel(num_cores) as pid:
            for block in T.Persistent([N], num_cores, pid, group_size=1):
                val_ub = T.alloc_shared((BB,), "int32")
                T.copy(val[block : block + 1], val_ub[:1])

                with T.SimtVF(threads=32):
                    for j in T.Parallel(BB):
                        val_ub[j] = 1
                T.copy(val_ub[:M], out[block, :M])

            z_ub = T.alloc_shared((BB,), "int32")

            with T.SimtVF(threads=BB):
                for j in T.Parallel(BB):
                    z_ub[j] = 0

            for bid in T.Persistent([T.ceildiv(total_rows, BB)], num_cores, pid, group_size=1):
                row_start = bid * BB
                fill_len = T.min(BB, total_rows - row_start)
                T.copy(z_ub[:fill_len], fill_1d[row_start : row_start + fill_len])

    return main


def test_ub_merge_wait_alias_distinct_offset():
    with tilelang.transform.PassContext(config={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE.value: True}):
        artifact = lower(_wait_alias_program(), target="ascend")
    offsets = _shared_offsets(artifact.kernel_source, "int32_t")
    # z_ub and val_ub must NOT be aliased: a wait_flag is an acquire, not a
    # release, so nothing orders the fill-loop read against the val-loop write.
    assert len(offsets) >= 2, f"z_ub / val_ub wrongly aliased by MergeUBAllocations; offsets={offsets}"


def _unused_ub_program():
    @T.prim_func
    def main(A: T.Tensor((1,), "int32"), O: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            _dead_ub = T.alloc_shared((128,), "float32")
            O[0] = A[0]

    return main


def test_ub_merge_removes_single_unused_allocation():
    artifact = lower(_unused_ub_program(), target="ascend")
    assert "dead_ub" not in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
