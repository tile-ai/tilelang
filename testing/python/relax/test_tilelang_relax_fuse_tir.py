import tilelang
import tilelang.relax
from tilelang import tvm
from tvm import relax, tirx, topi
from tvm.script import relax as R


def _make_fused_module():
    builder = relax.BlockBuilder()

    x = relax.Var("x", R.Tensor([10, 20], "float32"))
    scalar = relax.Var("scalar", R.Tensor([], "float32"))
    with builder.function("fused_add_exp", [x, scalar], attrs={"Primitive": True}, private=True):
        with builder.dataflow():
            added = builder.emit_te(topi.add, x, scalar)
            output = builder.emit_output(builder.call_te(topi.exp, added))
        builder.emit_func_output(output)

    fused_add_exp = builder.get().get_global_var("fused_add_exp")
    x = relax.Var("x", R.Tensor([10, 20], "float32"))
    scalar = relax.Var("scalar", R.Tensor([], "float32"))
    with builder.function("main", [x, scalar]):
        with builder.dataflow():
            output = builder.emit_output(relax.Call(fused_add_exp, [x, scalar]))
        builder.emit_func_output(output)

    return builder.get()


def test_fuse_tir_normalizes_static_buffer_shapes():
    mod = tilelang.relax.FuseTIR()(_make_fused_module())
    prim_func = next(func for func in mod.functions.values() if isinstance(func, tirx.PrimFunc))

    buffers = list(prim_func.buffer_map.values())

    def collect_alloc_buffers(stmt):
        if isinstance(stmt, tirx.SBlock):
            buffers.extend(stmt.alloc_buffers)

    tirx.stmt_functor.post_order_visit(prim_func.body, collect_alloc_buffers)

    static_dims = [dim for buffer in buffers for dim in buffer.shape if isinstance(dim, tirx.IntImm)]
    assert static_dims
    assert all(dim.dtype == "int32" for dim in static_dims)


if __name__ == "__main__":
    tvm.testing.main()
