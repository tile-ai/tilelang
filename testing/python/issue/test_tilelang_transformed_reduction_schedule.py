from tilelang import tvm
from tvm import te, tirx

from tilelang.schedule.gpu.reduction import Reduction


def test_transformed_single_source_reduction_uses_single_thread_launch():
    a = te.placeholder((8, 32), name="a", dtype="float32")
    rk = te.reduce_axis((0, 32), name="rk")
    c = te.compute(
        (8,),
        lambda i: te.sum(a[i, rk] + tirx.const(1.0, "float32"), axis=rk),
        name="c",
    )

    func = te.create_prim_func([a, c])
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    sch = Reduction().apply(func, target, False)

    assert sch is not None
    assert 'thread_binding(1, thread="threadIdx.x")' in sch.mod.script()


def test_square_single_source_reduction_lowers_to_sumsq():
    a = te.placeholder((8, 32), name="a", dtype="float32")
    rk = te.reduce_axis((0, 32), name="rk")
    c = te.compute(
        (8,),
        lambda i: te.sum(a[i, rk] * a[i, rk], axis=rk),
        name="c",
    )

    func = te.create_prim_func([a, c])
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    sch = Reduction().apply(func, target, False)

    assert sch is not None
    script = sch.mod.script()
    assert '"sumsq"' in script
    assert 'thread_binding(2, thread="threadIdx.x")' in script
