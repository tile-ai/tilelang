import tilelang.language as T
from tvm import tirx
from tvm.tirx import op
from tilelang.engine.lower import lower_to_host_device_ir


def _atomic_load_dynamic_block_index(num_tiles=4, threads=128):
    @T.prim_func
    def kernel(
        status: T.Tensor((num_tiles,), T.int32),
        out: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(num_tiles, threads=threads) as tile:
            look = T.alloc_var(T.int32)
            state = T.alloc_var(T.int32)
            done = T.alloc_var(T.bool)
            tx = T.get_thread_binding()

            if tx == 0:
                look = tile - 1
                done = look < 0
                state = 0
                while not done:
                    state = T.atomic_load(status[look], memory_order="acquire")
                    if state != 0:
                        done = True
                    else:
                        look -= 1
                        done = look < 0

                if tile == num_tiles - 1:
                    out[0] = state

    return kernel


def test_issue_2123_atomic_load_block_derived_index_lowers():
    func = _atomic_load_dynamic_block_index()
    _, device_mod, *_ = lower_to_host_device_ir(func, target="cuda")

    calls: list[tirx.Call] = []
    for prim_func in device_mod.functions.values():
        tirx.stmt_functor.post_order_visit(
            prim_func.body,
            lambda expr: calls.append(expr) if isinstance(expr, tirx.Call) else None,
        )

    assert not any(call.op.same_as(op.Op.get("tl.access_ptr")) for call in calls)
    assert any(
        call.op.same_as(op.Op.get("tl.atomic_load_elem_op"))
        and isinstance(call.args[0], tirx.Call)
        and call.args[0].op.same_as(op.Op.get("tirx.tvm_access_ptr"))
        for call in calls
    )
