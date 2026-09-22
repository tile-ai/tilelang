"""Fractal storage integrates with version expansion and numerical copy tails."""

import pytest
import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx
from testing.ascend._ir import allocated_buffer, calls, nodes, statements


def test_padded_alias_storage_sets_the_physical_version_pitch():
    @T.prim_func
    def main(A: T.Tensor((60, 15), "bfloat16")):
        with T.Kernel(1):
            storage = T.alloc_l1((225,), "uint16")
            tile = T.view(storage, (15, 15), dtype="bfloat16")
            l0 = T.alloc_l0a((15, 15), "bfloat16")
            T.annotate_buffer_versions({tile: (2, "iteration")})
            for i in T.Pipelined(4, num_stages=2):
                T.copy(A[i * 15 : (i + 1) * 15, :], tile)
                T.copy(tile, l0)

    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in {"tl.MaterializeMultiBuffer", "tl.FlattenBuffer"}:
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        tilelang.lower(main, target="ascend")
    assert int(allocated_buffer(snapshots["tl.MaterializeMultiBuffer"], "storage").strides[0]) == 256
    flat = snapshots["tl.FlattenBuffer"]
    (allocation,) = [node.buffer for node in nodes(flat, tirx.AllocBuffer) if node.buffer.scope() == "shared.l1"]
    assert int(allocation.shape[0]) == 512
    pointers = [call for call in calls(flat, "tirx.tvm_access_ptr") if call.args[1].type_annotation.storage_scope == "shared.l1"]
    assert len(pointers) == 2
    tvm.ir.assert_structural_equal(pointers[0].args[2], pointers[1].args[2])


def test_dynamic_fixpipe_padding_survives_full_lowering():
    @T.prim_func
    def main(A: T.Tensor((16, 16), "bfloat16"), B: T.Tensor((16, 16), "bfloat16"), C: T.Tensor((16, 4), "float32"), n: T.int32):
        with T.Kernel(1):
            T.assume(n >= 0)
            T.assume(n <= 4)
            a = T.alloc_l1((16, 16), "bfloat16")
            b = T.alloc_l1((16, 16), "bfloat16")
            accum = T.alloc_l0c((16, 16), "float32")
            ub = T.alloc_shared((16, 16), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, accum, transpose_B=True, clear_accum=True)
            T.copy(accum[:, :n], ub[:, :n])
            T.copy(ub[:, :n], C[:, :n])

    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AscendLowerTileOp":
                snapshots[info.name] = mod

    with tvm.transform.PassContext(instruments=[Capture()]):
        tilelang.lower(main, target="ascend")
    lowered = snapshots["tl.AscendLowerTileOp"]
    (copy,) = calls(lowered, "tl.ascend_copy_matrix_cc_to_ub")
    (parents,) = [ancestors for stmt, ancestors in statements(lowered) if isinstance(stmt, tirx.Evaluate) and stmt.value.same_as(copy)]
    guard = next(parent.condition for parent in reversed(parents) if isinstance(parent, tirx.IfThenElse))
    n = lowered["main"].params[-1]
    analyzer = tvm.arith.Analyzer()
    for value in range(5):
        substitute = {n: tirx.IntImm("int32", value)}
        assert int(analyzer.simplify(tirx.stmt_functor.substitute(copy.args[3], substitute))) == (8 if value else 0)
        assert bool(analyzer.simplify(tirx.stmt_functor.substitute(guard, substitute))) == (value > 0)


@pytest.mark.parametrize("use_alias", [False, True])
def test_full_logical_tile_padding_npu(use_alias):
    import torch

    @T.prim_func
    def main(A: T.Tensor((31, 128), "bfloat16"), B: T.Tensor((16, 64), "bfloat16"), C: T.Tensor((31, 16), "float32")):
        with T.Kernel(1):
            if use_alias:
                a_storage = T.alloc_l1((31 * 64,), "bfloat16")
                a_l1 = T.reshape(a_storage, (31, 64))
            else:
                a_l1 = T.alloc_l1((31, 64), "bfloat16")
            b_l1 = T.alloc_l1((16, 64), "bfloat16")
            a_l0 = T.alloc_l0a((31, 64), "bfloat16")
            b_l0 = T.alloc_l0b((16, 64), "bfloat16")
            if use_alias:
                c_storage = T.alloc_l0c((31, 16), "float32")
                accum = T.view(c_storage, (31, 16))
            else:
                accum = T.alloc_l0c((31, 16), "float32")
            T.copy(A[:, 112:176], a_l1)
            T.copy(B, b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, accum, transpose_B=True, clear_accum=True)
            T.copy(accum, C)

    kernel = tilelang.compile(main, out_idx=-1)
    torch.manual_seed(42)
    a = torch.randn((31, 128), device="npu", dtype=torch.bfloat16)
    b = torch.randn((16, 64), device="npu", dtype=torch.bfloat16)
    actual = kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, a[:, 112:128].float() @ b[:, :16].float().T, atol=1e-3, rtol=1e-3)


def test_short_k_slice_does_not_consume_neighboring_gm_values_npu():
    import torch

    @T.prim_func
    def main(A: T.Tensor((16, 32), "bfloat16"), B: T.Tensor((16, 32), "bfloat16"), C: T.Tensor((16, 16), "float32")):
        with T.Kernel(1):
            a_storage = T.alloc_l1((16 * 15,), "uint16")
            a_l1 = T.view(a_storage, (16, 15), dtype="bfloat16")
            b_l1 = T.alloc_l1((16, 15), "bfloat16")
            a_l0 = T.alloc_l0a((16, 15), "bfloat16")
            b_l0 = T.alloc_l0b((16, 15), "bfloat16")
            accum = T.alloc_l0c((16, 16), "float32")
            T.copy(A[:, :15], a_l1)
            T.copy(B[:, :15], b_l1)
            T.copy(a_l1, a_l0)
            T.copy(b_l1, b_l0)
            T.gemm(a_l0, b_l0, accum, transpose_B=True, clear_accum=True)
            T.copy(accum, C)

    torch.manual_seed(42)
    a = torch.randn((16, 32), device="npu", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a[:, 15:] = 128
    b[:, 15:] = 256
    result = tilelang.compile(main, out_idx=-1)(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(result, a[:, :15].float() @ b[:, :15].float().T, atol=1e-3, rtol=1e-3)
