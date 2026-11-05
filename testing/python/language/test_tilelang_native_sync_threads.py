"""Keep public thread-barrier defaults and named participant arguments native."""

import pytest
from tvm_tirx import ir, tirx
import tilelang.language as T
from tilelang.testing.ir import collect_calls


@pytest.mark.parametrize(
    "barrier_id,arrive_count,op", [(None, None, "tirx.gpu_storage_sync"), (0, None, "tl.storage_sync"), (1, 64, "tl.storage_sync")]
)
def test_sync_threads_preserves_native_barrier_contract(barrier_id, arrive_count, op):
    @T.prim_func
    def kernel(output: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            T.sync_threads(barrier_id, arrive_count)
            output[T.get_thread_binding()] = 7

    assert tirx.analysis.check_well_formed(kernel)
    (call,) = collect_calls(kernel, op=op)
    assert call.ty == ir.PrimType("void")
    assert call.args[0].value == "shared"
    expected = [value for value in (barrier_id, arrive_count) if value is not None]
    assert [int(value) for value in call.args[1:]] == expected
