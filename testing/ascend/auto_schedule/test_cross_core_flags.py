"""Generated cross-core handshakes must avoid user-reserved flag slots."""

import pytest
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import calls, schedule


def _program(reservation):
    @T.prim_func
    def main(C: T.Tensor((16, 16), "float32"), flag: T.int32):
        with T.Kernel(1):
            accum = T.alloc_l0c((16, 16), "float32")
            temp = T.alloc_shared((8, 16), "float32")
            # This is an InsertSync input: Fixpipe produces UB and MTE3 reads
            # it. GEMM, layout inference and device execution are irrelevant.
            T.dual_copy(accum, temp)
            if reservation == "sparse":
                T.ascend_sync_inter_arrive("PIPE_FIX", 15)
                T.ascend_sync_inter_wait("PIPE_FIX", 15)
            elif reservation == "bound":
                for outer in T.serial(2):
                    for inner in T.serial(2):
                        flag_id = T.bind(outer * 2 + inner)
                        T.ascend_sync_inter_arrive("PIPE_FIX", flag_id)
                        T.ascend_sync_inter_wait("PIPE_FIX", flag_id)
            elif reservation == "guard":
                for flag_id in T.serial(32):
                    if flag_id < 4:
                        T.ascend_sync_inter_arrive("PIPE_FIX", flag_id)
                        T.ascend_sync_inter_wait("PIPE_FIX", flag_id)
            else:
                if reservation == "assume":
                    T.assume(flag >= 0)
                    T.assume(flag < 4)
                T.ascend_sync_inter_arrive("PIPE_FIX", flag)
                T.ascend_sync_inter_wait("PIPE_FIX", flag)
            T.dual_copy(temp, C)

    return main


def _generated_slots(mod, operation):
    generated = [call for call in calls(mod, "tl.ascend_cross_core_" + operation + "_flag") if int(call.args[0]) == 4]
    assert generated, "Missing the Fixpipe/MTE3 cross-core handshake"
    # Both AIV subcores use the same 16-slot namespace, offset by subcore id.
    slots = set()
    for call in generated:
        if isinstance(call.args[2], tirx.IntImm):
            slots.add(int(call.args[2]) % 16)
    assert slots
    return slots


@pytest.mark.parametrize(
    "reservation, reserved",
    [
        pytest.param("sparse", {15}, id="sparse-explicit"),
        pytest.param("bound", set(range(4)), id="nested-bind"),
        pytest.param("guard", set(range(4)), id="guard-bounded"),
        pytest.param("assume", set(range(4)), id="assumption-bounded"),
    ],
)
def test_generated_flags_do_not_collide_with_explicit_flags(reservation, reserved):
    with tvm.transform.PassContext(config={"tl.Simplify": {"enable_simplify_let_inline": False}}):
        before = schedule(_program(reservation), through="ResolveCore", auto_schedule=False)
        mod = transform.InsertSync()(before)
    sets = _generated_slots(mod, "set")
    waits = _generated_slots(mod, "wait")
    assert sets == waits
    assert not sets & reserved
    assert all(0 <= slot < 16 for slot in sets)


def test_unbounded_explicit_flag_is_rejected():
    with pytest.raises(tvm.error.InternalError, match="integer range cannot be bounded"):
        before = schedule(_program("unbounded"), through="ResolveCore", auto_schedule=False)
        transform.InsertSync()(before)


if __name__ == "__main__":
    tilelang.testing.main()
