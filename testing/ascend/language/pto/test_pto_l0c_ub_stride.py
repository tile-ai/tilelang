"""L0C NZ source pitches cover M; ND UB destination pitches cover N."""

import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.ascend import language as T
from tilelang.backend import create_backend_context
from tilelang.instrumentation import PassInstrumentationTool, compile_pass_instrumentation
from tvm import tirx


pytestmark = pytest.mark.pto
GEOMETRIES = [(32, 64), (64, 32), (16, 64), (64, 16), (64, 64)]


def program(m, n):
    @T.prim_func
    def main(A: T.Tensor((m, 16), "bfloat16"), B: T.Tensor((n, 16), "bfloat16"), C: T.Tensor((m, n), "float32")):
        with T.MixedKernel(1, sids=1) as (_, sid):
            a = T.alloc_l1((m, 16), "bfloat16")
            b = T.alloc_l1((n, 16), "bfloat16")
            acc = T.alloc_l0c((m, n), "float32")
            ub = T.alloc_shared((m, n), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, ub, sub_blockid=0)
            if sid == 0:
                T.copy(ub, C)

    return main


def lower(m, n):
    target = create_backend_context("pto", execution_backend="pto").target
    with target:
        return tilelang.lower(program(m, n), target=target), target


@pytest.mark.parametrize("m,n", GEOMETRIES)
def test_rectangular_source_and_destination_pitches(m, n):
    artifact, _ = lower(m, n)
    line = next(line for line in artifact.kernel_source.splitlines() if "pto.mte_l0c_ub(" in line)
    assert f', {m}, {n}, {m}, {n}, 0, layout="nz2nd")' in line


@pytest.mark.parametrize(
    "m,n,index,pitch,diagnostic",
    [(64, 32, 6, 32, "source stride must be at least m_size"), (32, 64, 5, 32, "destination stride must be at least n_size")],
)
def test_undersized_pitch_is_still_rejected(m, n, index, pitch, diagnostic):
    target = create_backend_context("pto", execution_backend="pto").target
    replacements = []

    def replace(node):
        if isinstance(node, tirx.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.ascend_copy_matrix_cc_to_ub":
            args = list(node.args)
            args[index] = tirx.IntImm(args[index].dtype, pitch)
            replacements.append(node)
            return tirx.Call(node.dtype, node.op, args, annotations=node.annotations, span=node.span)

    class CorruptPitch(PassInstrumentationTool):
        def run_codegen(self, event, next_call):
            for gv, func in list(event.mod.functions.items()):
                event.mod.update_func(gv, func.with_body(tirx.stmt_functor.ir_transform(func.body, None, replace)))
            assert len(replacements) == 1
            return next_call()

    with (
        target,
        compile_pass_instrumentation(tools=[CorruptPitch()], include_default_tools=False),
        pytest.raises(tvm.error.InternalError, match=diagnostic),
    ):
        tilelang.lower(program(m, n), target=target)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("m,n", GEOMETRIES)
def test_rectangular_copy_numerics(m, n):
    target = create_backend_context("pto", execution_backend="pto").target
    with target:
        kernel = tilelang.compile(program(m, n), target=target, execution_backend="pto", out_idx=-1)
    # Small asymmetric integers make address/stride errors visible and keep
    # the fp32 GEMM reference exact, independently of reduction order.
    a = ((torch.arange(m * 16).reshape(m, 16) % 7) - 3).to(torch.bfloat16)
    b = ((torch.arange(n * 16).reshape(n, 16) % 5) - 2).to(torch.bfloat16)
    previous = None
    for factor in (1, 2):
        actual = kernel((a * factor).npu(), b.npu())
        torch.npu.synchronize()
        torch.testing.assert_close(actual.cpu(), (a.float() * factor) @ b.float().T, rtol=0, atol=0)
        if previous is not None:
            torch.testing.assert_close(previous[0].cpu(), previous[1], rtol=0, atol=0)
        previous = (actual, actual.cpu().clone())
