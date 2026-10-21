import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir

tilelang.env.disable_cache()


def apply_sunmmio_passes(mod, target):
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.ResolveSunmmioMeshSymbols()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.LegalizeSunmmioDataPath()(mod)
    mod = tilelang.transform.SunmmioLayoutInference()(mod)
    mod = tilelang.transform.LegalizeSunmmioGemm()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    return mod


def allreduce_kernel(direction="all", clear=True, dtype="float32"):
    shape = (32, 32)
    out_shape = shape

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out: T.Tensor(out_shape, dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.copy(A, A_shared)
            if not clear:
                T.copy(Out, Out_shared)
            T.comm.all_reduce(A_shared, Out_shared, "sum", direction, clear=clear)
            T.copy(Out_shared, Out)

    return tvm.IRModule({"main": main})


@tir.functor.visitor
class AllreduceIRChecker(tir.PyStmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.broadcast_calls = []
        self.dma_copy_calls = []
        self.in_tile_reduce_calls = []
        self.rsram_alloc_names = []

    def visit_block_(self, op):
        for buf in op.alloc_buffers:
            if buf.scope() == "shared.rsram":
                self.rsram_alloc_names.append(buf.name)
        super().visit_block_(op)

    def visit_call_(self, op):
        if hasattr(op, "op") and hasattr(op.op, "name"):
            if op.op.name == "tl.broadcast_":
                self.broadcast_calls.append(op)
            elif op.op.name == "tl.dma_copy":
                self.dma_copy_calls.append(op)
            elif op.op.name == "tl.vector_core_in_tile_reduce":
                self.in_tile_reduce_calls.append(op)
        super().visit_call_(op)


def _region_access_mask(region_call):
    assert isinstance(region_call, tir.Call)
    assert region_call.op.name == "tl.tileop.region"
    return int(region_call.args[1])


def _region_buffer_name(region_call):
    assert isinstance(region_call, tir.Call)
    assert region_call.op.name == "tl.tileop.region"
    load = region_call.args[0]
    assert isinstance(load, tir.BufferLoad)
    return load.buffer.name


@pytest.mark.parametrize(
    "direction, clear, expected_directions",
    [
        ("h", True, [0]),
        ("v", True, [1]),
        ("all", True, [1, 0]),
        ("h", False, [0]),
        ("v", False, [1]),
        ("all", False, [1, 0]),
    ],
)
def test_tilelang_allreduce_sunmmio_lowers_to_broadcast_and_tile_reduce(direction, clear, expected_directions):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = allreduce_kernel(direction=direction, clear=clear)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    checker = AllreduceIRChecker()
    checker.visit_stmt(mod["main"].body)

    assert len(checker.broadcast_calls) == len(expected_directions)
    assert [int(call.args[2]) for call in checker.broadcast_calls] == expected_directions
    assert all(len(call.args) == 5 for call in checker.broadcast_calls)
    assert all(_region_access_mask(call.args[0]) == 1 for call in checker.broadcast_calls)
    assert all(_region_access_mask(call.args[1]) == 2 for call in checker.broadcast_calls)
    assert all(int(call.args[3]) == 15 for call in checker.broadcast_calls)
    assert all(int(call.args[4]) == 0 for call in checker.broadcast_calls)
    assert all(int(call.annotations["tl.allow_rank3_allreduce_multicast"]) == 1 for call in checker.broadcast_calls)
    # Cross-core-only all_reduce lowers each carrier reduction along its
    # leading participant axis; it must not insert a local payload reduction.
    assert len(checker.in_tile_reduce_calls) == 0
    dma_buffer_pairs = [(_region_buffer_name(call.args[0]), _region_buffer_name(call.args[1])) for call in checker.dma_copy_calls]
    if clear:
        assert ("Out", "Out_shared") not in dma_buffer_pairs
    else:
        assert ("Out", "Out_shared") in dma_buffer_pairs
