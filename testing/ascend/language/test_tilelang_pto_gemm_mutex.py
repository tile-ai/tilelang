"""L1 GEMM must retain the mutex window reserved by RewriteFlagToBuf."""

import ast

import pytest

from tilelang import tvm


def _gemm_source(buf_offset, blockscaled):
    tir = tvm.tirx
    dtype = "float8_e4m3fn" if blockscaled else "float16"
    a = tir.decl_buffer((2048,), dtype, name="a", scope="shared.l1")
    b = tir.decl_buffer((2048,), dtype, name="b", scope="shared.l1")
    c = tir.decl_buffer((256,), "float32", name="c", scope="shared.l0c")
    buffers = [a, b, c]
    pointers = [tir.address_of(buf[0]) for buf in (c, a, b)]
    if blockscaled:
        sa = tir.decl_buffer((32,), "uint16", name="sa", scope="shared.l1")
        sb = tir.decl_buffer((32,), "uint16", name="sb", scope="shared.l1")
        buffers += [sa, sb]
        arguments = pointers + [
            tir.address_of(sa[0]),
            tir.address_of(sb[0]),
            16,
            128,
            16,
            64,
            1,
            True,
            "float8_e4m3_t",
            "uint16_t",
            "float",
            buf_offset,
            0,
            2,
            0,
        ]
        name = "tl.ascend_blockscaled_gemm_l1"
    else:
        arguments = pointers + [16, 128, 16, 64, 1, True, "half", buf_offset, 0]
        name = "tl.ascend_gemm_l1"
    call = tir.call_intrin("void", tvm.ir.Op.get(name), *arguments)
    body = tir.SeqStmt([tir.AllocBuffer(buf) for buf in buffers] + [tir.Evaluate(call)])
    target = tvm.target.Target({"kind": "ascend", "keys": ["pto"]})
    func = tir.PrimFunc([], body).with_attr("global_symbol", "main").with_attr("target", target)
    source = tvm.get_global_func("target.build.tilelang_pto")(tvm.IRModule({"main": func}), target).inspect_source()
    ast.parse(source)
    return source


@pytest.mark.parametrize("blockscaled", [False, True], ids=["mad", "mx"])
@pytest.mark.parametrize("buf_offset", [0, 6, 30])
def test_gemm_codegen_preserves_mutex_offset(blockscaled, buf_offset):
    source = _gemm_source(buf_offset, blockscaled)
    call = next(line for line in source.splitlines() if ".run_l1_tile(" in line)
    assert f"buf_offset={buf_offset}" in call


@pytest.mark.parametrize("blockscaled", [False, True], ids=["mad", "mx"])
@pytest.mark.parametrize("buf_offset", [-1, 31])
def test_gemm_codegen_rejects_invalid_mutex_window(blockscaled, buf_offset):
    with pytest.raises(tvm.error.InternalError, match="buf_offset"):
        _gemm_source(buf_offset, blockscaled)
