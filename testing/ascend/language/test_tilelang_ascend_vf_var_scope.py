from unittest.mock import patch

import tilelang.ascend.language as T
from tilelang.language.eager import builder as eager_builder
from tilelang.testing.ir import collect_nodes
from tvm_tirx import ir


def _execution_blocks(function):
    blocks = [node for node in collect_nodes(function, ir.RegionStmt) if node.op.name == "tl.ascend_block"]
    for block in blocks:
        assert not block.args
        assert set(block.attrs.keys()) <= {"name", "vector_count", "tl.vf_source_index", "tl.vf_latency"}
        assert "name" in block.attrs
    return blocks


def test_sibling_simtvf_blocks_allow_same_immutable_name():
    with patch.object(eager_builder.logger, "warning") as warning:

        @T.prim_func
        def main(A: T.Tensor[(256,), T.int32], B: T.Tensor[(256,), T.int32]):
            with T.Kernel(1):
                with T.SimtVF(threads=128):
                    tx = T.get_thread_binding()
                    value = A[tx] + 1
                    B[tx] = value
                with T.SimtVF(threads=128):
                    tx = T.get_thread_binding()
                    value = A[tx + 128] + 1
                    B[tx + 128] = value

    assert main is not None
    warning.assert_not_called()
    blocks = _execution_blocks(main)
    assert [block.attrs["name"] for block in blocks] == ["SIMT_VF", "SIMT_VF"]
    assert [int(block.attrs["tl.vf_source_index"]) for block in blocks] == [0, 1]


def test_cube_and_vector_blocks_allow_same_immutable_name():
    with patch.object(eager_builder.logger, "warning") as warning:

        @T.prim_func
        def main():
            with T.Kernel(1) as bx:
                with T.Cube():
                    value = bx + 1
                    alias = value
                    T.evaluate(alias)
                with T.Vector() as sid:
                    value = bx + sid
                    T.evaluate(value)
                alias = bx + 2
                T.evaluate(alias)

    assert main is not None
    warning.assert_not_called()
    blocks = _execution_blocks(main)
    assert [block.attrs["name"] for block in blocks] == ["CUBE", "VECTOR"]
    assert int(blocks[1].attrs["vector_count"]) == 2
