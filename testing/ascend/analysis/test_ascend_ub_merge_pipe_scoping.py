"""
The happens-before model must not wire events across hardware pipes (e.g. a
fixpipe wait does not order an MTE1 store). Such fabricated ordering lets
Merge reuse two buffers whose new read/write conflict has no hardware
synchronization, corrupting kernels.

These tests assert the reuse decisions on the MergeUBAllocations-after IR.
"""

import pytest

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend import transform as ascend_transform
from tilelang import tvm
from tilelang.engine.lower import lower
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


def _make_two_gemm_program(M=512, K=128, N=128, dtype="bfloat16"):
    """O1 = X @ W1^T, O2 = X @ W2^T.

    W2 is staged through a GM bounce by the AIV core (mirrors the DSA KV_gm
    gather pattern), pinning the second MTE2 fill after gemm1 so w1/w2
    lifetimes are disjoint and become reuse candidates.
    """

    @T.prim_func
    def main(
        X: T.Buffer((M, K), dtype),
        W1: T.Buffer((N, K), dtype),
        W2: T.Buffer((N, K), dtype),
        W2_gm: T.Buffer((N, K), dtype),
        O1: T.Buffer((M, N), "float32"),
        O2: T.Buffer((M, N), "float32"),
    ):
        with T.MixedKernel(1) as (bx, sid):
            x_shared = T.alloc_l1((M, K), dtype)
            w1_shared = T.alloc_l1((N, K), dtype)
            w2_shared = T.alloc_l1((N, K), dtype)

            acc = T.alloc_l0c((M, N), "float32")
            temp = T.alloc_shared((M // 2, N), "float32")
            w2_ub = T.alloc_shared((N, K), dtype)

            # AIV: stage W2 through the GM bounce buffer.
            T.copy(W2[0:N, 0:K], w2_ub)
            T.copy(w2_ub, W2_gm[0:N, 0:K])

            # AIC: first GEMM + fixpipe drain of its accumulator.
            T.copy(X[0:M, 0:K], x_shared)
            T.copy(W1[0:N, 0:K], w1_shared)
            T.gemm(
                x_shared,
                w1_shared,
                acc,
                transpose_B=True,
                clear_accum=True,
            )
            T.dual_copy(acc, temp)
            T.dual_copy(temp, O1[0:M, 0:N])

            # AIC: fill w2_shared via MTE2, pinned after gemm1 by the
            # cross-core wait for the AIV's W2_gm store.
            T.copy(W2_gm[0:N, 0:K], w2_shared)
            T.gemm(
                x_shared,
                w2_shared,
                acc,
                transpose_B=True,
                clear_accum=True,
            )
            T.dual_copy(acc, temp)
            T.dual_copy(temp, O2[0:M, 0:N])

    return main


def _merge_after_module():
    """Run manual alias inference on the final lowered two-gemm program.

    The program requires AutoSchedule to lower ``T.MixedKernel``. Capture the
    unified merge pass's input, replace AutoSchedule's contract with the manual
    inference result, then run the common allocator.
    """

    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class CaptureMerge:
        def run_before_pass(self, mod, info):
            if info.name == "tl.MergeUBAllocations":
                snapshots["mod"] = mod

    with tvm.transform.PassContext(
        opt_level=3,
        instruments=[CaptureMerge()],
    ):
        lower(
            _make_two_gemm_program(),
            target="ascend",
        )

    assert "mod" in snapshots, "tl.MergeUBAllocations pass not observed"

    mod = ascend_transform.InferBufferAliases()(snapshots["mod"])
    return ascend_transform.MergeUBAllocations(align_bytes=32)(mod)


def _var_name(x):
    """Name of the buffer a var-like expression refers to (Var or .data load)."""

    if isinstance(x, tirx.Var):
        return x.name

    if isinstance(x, tirx.BufferLoad):
        return x.buffer.data.name

    return None


def _dst_byte_offset(access_ptr):
    """Byte offset encoded by the access_ptr offset expression.

    MergeUBAllocations rewrites accesses as element offsets of the form
    ``FloorDiv(<bytes * 8>, <elem_bits>)``; the byte offset is the FloorDiv
    numerator divided by 8. A plain IntImm offset is interpreted as an
    element offset for a 16-bit dtype (bf16), i.e. multiplied by 2.
    """

    off = access_ptr.args[2]

    if isinstance(off, tirx.IntImm):
        return int(off) * 2

    numerator = getattr(off, "a", None)
    if numerator is not None and isinstance(numerator, tirx.IntImm):
        return int(numerator) // 8

    return None


def _extract_layout(mod):
    """Walk every PrimFunc body in the Merge-after module and extract
    (buf_dyn_l1 arena bytes, {GM source param -> dst byte offset}).
    """

    result = {
        "arena": None,
        "fills": {},
    }

    def scan_expr(e):
        if not isinstance(e, tirx.Call):
            return

        name = getattr(e.op, "name", "")

        if "ascend_copy_gm_to_cbuf" in name and len(e.args) >= 2:
            dst, src = e.args[0], e.args[1]

            if (
                isinstance(dst, tirx.Call)
                and "tvm_access_ptr" in getattr(dst.op, "name", "")
                and len(dst.args) >= 3
                and isinstance(src, tirx.Call)
                and "tvm_access_ptr" in getattr(src.op, "name", "")
                and len(src.args) >= 2
            ):
                src_name = _var_name(src.args[1])

                if src_name in ("W1", "W2_gm"):
                    byte_off = _dst_byte_offset(dst)

                    if byte_off is not None:
                        result["fills"][src_name] = byte_off

        for arg in e.args:
            scan_expr(arg)

    def scan_stmt(node):
        if isinstance(node, tirx.AllocBuffer):
            if node.buffer.data.name == "buf_dyn_l1":
                result["arena"] = int(node.buffer.shape[0])
            return

        # Evaluate / BufferStore / ... carry an expression payload.
        value = getattr(node, "value", None)

        if isinstance(value, tirx.Call):
            scan_expr(value)

    for func in mod.functions.values():
        if isinstance(func, tirx.PrimFunc):
            post_order_visit(func.body, scan_stmt)

    return result


def test_two_gemm_no_cross_pipe_reuse():
    """w1_shared / w2_shared must not alias after the pipe-scoping fix."""

    layout = _extract_layout(_merge_after_module())

    # Direct check on the reuse decision itself: the two fills must land at
    # distinct byte offsets inside buf_dyn_l1.
    fills = layout["fills"]

    assert "W1" in fills, f"W1 fill not found in Merge-after IR (fills={fills})"

    assert "W2_gm" in fills, f"W2_gm fill not found in Merge-after IR (fills={fills})"

    assert fills["W1"] != fills["W2_gm"], (
        f"w1_shared/w2_shared wrongly aliased by MergeUBAllocations: both fills write buf_dyn_l1 at byte offset {fills['W1']}"
    )

    # Indirect check: the L1 arena must be x + w1 + w2 (no slot shared).
    arena = layout["arena"]

    assert arena is not None, "buf_dyn_l1 arena not found in Merge-after IR"

    x_bytes = 512 * 128 * 2  # bf16
    w_bytes = 128 * 128 * 2
    expected = x_bytes + 2 * w_bytes

    assert arena == expected, (
        f"w1_shared/w2_shared wrongly reused by MergeUBAllocations: L1 arena={arena}B, expected {expected}B (x + w1 + w2, no aliasing)"
    )


def _pipe_all_program(N=64):
    """Two UB staging buffers separated by an explicit PIPE_ALL fence.

    The fence is a full-core barrier, so the pass must model it with all-pipe
    edges and reuse one slot.
    """

    @T.prim_func
    def main(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        O: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(1):
            u1 = T.alloc_shared((N,), "float32")
            u2 = T.alloc_shared((N,), "float32")

            T.copy(A, u1)
            T.copy(u1, O)

            T.ascend_pipe_barrier("PIPE_ALL")

            T.copy(B, u2)
            T.copy(u2, O)

    return main


def _set_pad_value_scalar_pipe_program(N=64):
    """A Scalar UB read followed by an unrelated Vector -> MTE2 handshake.

    The SetPadValue operand must be tagged Scalar rather than inheriting the
    enclosing Vector block. Otherwise the V_MTE2 handshake fabricates an
    ordering from the read of u1 to the later write of u2 and permits reuse.
    """

    @T.prim_func
    def main(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(1), T.Vector():
            u1 = T.alloc_shared((N,), "float32")
            u2 = T.alloc_shared((N,), "float32")

            T.copy(A, u1)
            T.ascend_set_flag("MTE2_S", 0)
            T.ascend_wait_flag("MTE2_S", 0)
            T.ascend_set_copy_pad_value(u1[0])

            T.ascend_set_flag("V_MTE2", 1)
            T.ascend_wait_flag("V_MTE2", 1)
            T.copy(B, u2)

    return main


def _dyn_shmem_arena_bytes(mod):
    """Size of the merged buf_dyn_shmem arena in the given IRModule."""

    def find_arena(func):
        arena = None

        def scan_stmt(node):
            nonlocal arena

            if isinstance(node, tirx.AllocBuffer) and node.buffer.data.name == "buf_dyn_shmem":
                arena = int(node.buffer.shape[0])

        post_order_visit(func.body, scan_stmt)
        return arena

    for func in mod.functions.values():
        if not isinstance(func, tirx.PrimFunc):
            continue

        arena = find_arena(func)

        if arena is not None:
            return arena

    return None


def test_set_copy_pad_value_buffer_load_uses_scalar_pipe():
    """SetPadValue's UB operand must execute on the Scalar pipe."""

    n = 64
    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class CaptureMerge:
        def run_after_pass(self, mod, info):
            if info.name == "tl.MergeUBAllocations":
                snapshots["mod"] = mod

    with tvm.transform.PassContext(
        opt_level=3,
        instruments=[CaptureMerge()],
        config={
            tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: False,
        },
    ):
        lower(
            _set_pad_value_scalar_pipe_program(n),
            target="ascend",
        )

    assert "mod" in snapshots, "tl.MergeUBAllocations pass not observed"

    arena = _dyn_shmem_arena_bytes(snapshots["mod"])
    two_slots = 2 * n * 4  # two float32 buffers

    assert arena == two_slots, (
        f"SetPadValue's scalar UB read inherited the enclosing Vector pipe: arena={arena}B, expected {two_slots}B (u1/u2 must not alias)"
    )


@pytest.mark.parametrize("disable_reuse", [False, True])
def test_pipe_all_barrier_is_full_core_fence(disable_reuse):
    """PIPE_ALL must be modeled as a full-core fence (kAll).

    The fence serializes the lifetimes it separates.

    Regression: PIPE_ALL used to parse as kUnknown and the fence was dropped
    entirely, wrongly rejecting this reuse.
    """

    snapshots = {}
    scripts = {}

    @tvm.ir.instrument.pass_instrument
    class CaptureMerge:
        def run_after_pass(self, mod, info):
            if info.name == "tl.InferBufferAliases":
                scripts[info.name] = mod.script()
            if info.name == "tl.MergeUBAllocations":
                snapshots["mod"] = mod
                scripts[info.name] = mod.script()

    with tvm.transform.PassContext(
        opt_level=3,
        instruments=[CaptureMerge()],
        config={
            tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: False,
            tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE.value: disable_reuse,
        },
    ):
        lower(
            _pipe_all_program(),
            target="ascend",
        )

    assert "mod" in snapshots, "tl.MergeUBAllocations pass not observed"
    assert "tl.buffer_alias_map" in scripts["tl.InferBufferAliases"]
    assert "tl.buffer_alias_map" not in scripts["tl.MergeUBAllocations"]

    arena = _dyn_shmem_arena_bytes(snapshots["mod"])
    expected_arena = 64 * 4 * (2 if disable_reuse else 1)

    assert arena == expected_arena, (
        f"Incorrect PIPE_ALL reuse layout with disable_reuse={disable_reuse}: UB arena={arena}B, expected {expected_arena}B"
    )


if __name__ == "__main__":
    tilelang.testing.main()
