"""Instruction selection consumes native boolean annotation representations."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.language as T


def make_op(value, *, truth, scope):
    def tensor(name, shape, dtype, allocation_scope):
        return ir.Var(name, tirx.TensorType(shape, dtype, scope=allocation_scope, layout="default"))

    tilelang.cuda.register_backend()
    ab_dtype = "float8_e4m3fn" if scope == "shared.tmem" else "float4_e2m1fn"
    sf_scope = "shared.tmem" if scope == "shared.tmem" else "shared"
    granularity = 128 if scope == "shared.tmem" else 16
    a = tensor("A", (128, 128), ab_dtype, "shared")
    b = tensor("B", (128, 128), ab_dtype, "shared")
    c = tensor("C", (128, 256 if truth else 128), "float32", scope)
    sfa = tensor("SFA", (128, 4), "uint32", sf_scope)
    sfb = tensor("SFB", (128, 8 if truth else 4), "uint32", sf_scope)
    done = tensor("done", (1,), "uint64", "shared")
    call = T.gemm_blockscaled(
        a,
        b,
        c,
        sfa,
        sfb,
        transpose_B=True,
        k_start=0,
        sf_a_granularity_k=granularity,
        sf_b_granularity_k=granularity,
        mbar=done[0],
        use_2cta=truth,
    )
    count = int(call.attrs["operand_count"])
    annotations = dict(tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[count], call.attrs["annotation_schema"]))
    annotations["use_2cta"] = value
    return call.op.get_attr("TLOpBuilder")(call.args[:count], annotations)


FALSE_VALUES = [False, 0, ir.const(0, "int32"), ir.const(0, "uint64"), ir.const(False, "bool")]
TRUE_VALUES = [True, 1, -2, ir.const(1, "int32"), ir.const(-2, "int64"), ir.const(True, "bool")]


@pytest.mark.parametrize("value", FALSE_VALUES)
def test_false_representations_allow_sm120_mma(value):
    op = make_op(value, truth=False, scope="local.fragment")
    assert op._select_gemm_instruction(128, Target({"kind": "cuda", "arch": "sm_120"})) == "cuda.mma.blockscaled"


@pytest.mark.parametrize("value", TRUE_VALUES)
def test_true_representations_do_not_fall_back_to_sm120(value):
    op = make_op(value, truth=True, scope="local.fragment")
    with pytest.raises(Exception, match="Block-scaled GEMM requires Blackwell SM100 TCGEN5MMA"):
        op._select_gemm_instruction(128, Target({"kind": "cuda", "arch": "sm_120"}))


@pytest.mark.parametrize("value,truth", [(value, False) for value in FALSE_VALUES] + [(value, True) for value in TRUE_VALUES])
def test_valid_representations_keep_sm100_tcgen05_selection(value, truth):
    op = make_op(value, truth=truth, scope="shared.tmem")
    assert op._select_gemm_instruction(128, Target({"kind": "cuda", "arch": "sm_100"})) == "cuda.tcgen05.blockscaled"


@pytest.mark.parametrize("value", [0.5, "true", [], ir.Var("symbolic", "int32")])
def test_invalid_annotation_representations_raise_at_selection(value):
    op = make_op(value, truth=False, scope="local.fragment")
    with pytest.raises(TypeError):
        op._select_gemm_instruction(128, Target({"kind": "cuda", "arch": "sm_120"}))


def test_uint64_intimm_preserves_existing_nonzero_truth_semantics():
    value = ir.const((1 << 64) - 1, "uint64")
    op = make_op(value, truth=True, scope="shared.tmem")
    assert op._select_gemm_instruction(128, Target({"kind": "cuda", "arch": "sm_100"})) == "cuda.tcgen05.blockscaled"
