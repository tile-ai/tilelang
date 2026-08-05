import pytest

import numpy as np

import tilelang as tl
from tilelang import tvm


def test_lower_opaque_block_preserves_non_unit_loop_step():
    output_buffer = tvm.tirx.decl_buffer((6,), "int32", name="output")
    i = tvm.tirx.Var("i", "int32")
    loop = tvm.tirx.For(
        i,
        1,
        5,
        tvm.tirx.ForKind.SERIAL,
        tvm.tirx.BufferStore(output_buffer, 1, [i]),
        step=tvm.tirx.IntImm("int32", 2),
    )
    before = tvm.tirx.PrimFunc(
        [output_buffer.data],
        loop,
        buffer_map={output_buffer.data: output_buffer},
    ).with_attr("global_symbol", "main")

    mod = tl.transform.LowerOpaqueBlock()(tvm.IRModule.from_expr(before))
    executable = tvm.compile(mod["main"], target="c").jit(options=["-std=c++17"])

    output = tvm.runtime.tensor(np.zeros(6, dtype="int32"))
    executable["main"](output)

    np.testing.assert_array_equal(
        output.numpy(),
        np.array([0, 1, 0, 1, 0, 1], dtype="int32"),
    )


@pytest.mark.parametrize("rng", [(0, 6, 2), (0, 5, 2)], ids=lambda v: f"rng=({v[0]},{v[1]},{v[2]})")
@pytest.mark.parametrize("explicit", [False, True], ids=lambda v: f"explicit={v}")
def test_unroll_loop_preserves_non_unit_loop_step(rng, explicit):
    output_buffer = tvm.tirx.decl_buffer((8,), "int32", name="output")
    i = tvm.tirx.Var("i", "int32")
    start, stop, step = rng
    loop = tvm.tirx.For(
        i,
        start,
        stop,
        tvm.tirx.ForKind.UNROLLED,
        tvm.tirx.BufferStore(output_buffer, i, [i]),
        step=tvm.tirx.IntImm("int32", step),
        annotations={"pragma_unroll_explicit": explicit},
    )
    before = tvm.tirx.PrimFunc(
        [output_buffer.data],
        loop,
        buffer_map={output_buffer.data: output_buffer},
    ).with_attr("global_symbol", "main")

    mod = tl.transform.UnrollLoop()(tvm.IRModule.from_expr(before))
    executable = tvm.compile(mod["main"], target="c").jit(options=["-std=c++17"])

    output = tvm.runtime.tensor(np.zeros(8, dtype="int32"))
    executable["main"](output)

    np.testing.assert_array_equal(
        output.numpy(),
        np.array([0, 0, 2, 0, 4, 0, 0, 0], dtype="int32"),
    )


def test_unroll_loop_preserves_empty_non_unit_loop():
    output_buffer = tvm.tirx.decl_buffer((1,), "int32", name="output")
    i = tvm.tirx.Var("i", "int32")
    loop = tvm.tirx.For(
        i,
        0,
        0,
        tvm.tirx.ForKind.UNROLLED,
        tvm.tirx.BufferStore(output_buffer, 1, [0]),
        step=tvm.tirx.IntImm("int32", 2),
        annotations={"pragma_unroll_explicit": True},
    )
    before = tvm.tirx.PrimFunc(
        [output_buffer.data],
        loop,
        buffer_map={output_buffer.data: output_buffer},
    ).with_attr("global_symbol", "main")

    mod = tl.transform.UnrollLoop()(tvm.IRModule.from_expr(before))
    executable = tvm.compile(mod["main"], target="c").jit(options=["-std=c++17"])

    output = tvm.runtime.tensor(np.zeros(1, dtype="int32"))
    executable["main"](output)

    np.testing.assert_array_equal(output.numpy(), np.zeros(1, dtype="int32"))
