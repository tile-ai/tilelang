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
