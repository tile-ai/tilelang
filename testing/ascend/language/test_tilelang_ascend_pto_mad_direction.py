import tvm_ffi

"""PTO MAD direction propagation and valid Python control-flow emission."""

from tvm_tirx import ir
from testing.ascend._ir import allocate, tensor

import ast

import pytest

import tilelang
import tvm_tirx
from tilelang.ascend import language as T


tirx = tvm_tirx.tirx


def _codegen(body_builder):
    """Build minimal device IR directly, retaining loops at the codegen boundary."""
    flag = ir.Var("flag", "int32")
    a = tensor((256,), "bfloat16", name="a", scope="shared.l0a")
    b = tensor((256,), "bfloat16", name="b", scope="shared.l0b")
    c = tensor((256,), "float32", name="c", scope="shared.l0c")
    value = tensor((1,), "int32", name="value", scope="local.var")

    def mad():
        return ir.Evaluate(
            ir.Call(
                tvm_tirx.ir.Op.get("tl.ascend_mad"),
                [tirx.address_of(c[0]), tirx.address_of(a[0]), tirx.address_of(b[0]), 16, 16, 16, 0, 1, 0, True],
                ty="handle",
            )
        )

    body = ir.SeqStmt([allocate(buf) for buf in (a, b, c, value)] + body_builder(flag, value, mad))
    func = (
        tirx.Function([flag], body)
        .with_attr("global_symbol", "main")
        .with_attr("target", tvm_tirx.target.Target({"kind": "ascend", "keys": ["pto"]}))
    )
    build = tvm_ffi.get_global_func("target.build.tilelang_pto")
    source = build(tvm_tirx.IRModule({"main": func}), tvm_tirx.target.Target({"kind": "ascend", "keys": ["pto"]})).inspect_source()
    ast.parse(source)
    return source


def _if(condition, then_case, else_case=None):
    return ir.If(condition, then_case, else_case)


def _set(direction):
    return ir.Evaluate(ir.Call(tvm_tirx.ir.Op.get("tl.ascend_set_mmad_direction"), [direction], ty="handle"))


def _mad_directions(source):
    return ["n_dir=True" in line for line in source.splitlines() if line.strip().startswith("pto.mad(")]


def test_nested_guards_preserve_direction():
    def body(flag, value, mad):
        condition = ir.prim.And(flag > 0, flag < 10)
        return [
            _if(condition, _set("n")),
            _if(flag > 0, _if(flag < 10, ir.SeqStmt([mad(), _set("m")]))),
            mad(),
        ]

    assert _mad_directions(_codegen(body)) == [True, False]


def test_ambiguous_direction_is_rejected():
    def body(flag, value, mad):
        return [_if(flag > 0, _set("n")), mad()]

    with pytest.raises(Exception, match="MAD direction depends on control flow"):
        _codegen(body)


def test_mutable_guards_are_not_correlated():
    def body(flag, value, mad):
        return [
            ir.TensorStore(value, [0], flag),
            _if(value[0] > 0, _set("n")),
            ir.TensorStore(value, [0], 1),
            _if(value[0] > 0, mad()),
        ]

    with pytest.raises(Exception, match="MAD direction depends on control flow"):
        _codegen(body)


@pytest.mark.parametrize("extent", [0, 1])
def test_loop_reaches_codegen(extent):
    def body(flag, value, mad):
        loop = ir.For(
            ir.Var("i", "int32"), ir.const(0), ir.const(0) + extent, ir.const(1), ir.ForKind.DEFAULT, ir.SeqStmt([mad(), _set("n")])
        )
        return [loop, mad()]

    source = _codegen(body)
    assert "range(" in source
    directions = _mad_directions(source)
    if extent == 0:
        # Only the direction after the loop matters; its body is unreachable.
        assert directions[-1] is False
    else:
        assert directions == [False, True]


def test_loop_backedge_conflict_is_rejected():
    def body(flag, value, mad):
        return [
            ir.For(
                ir.Var("i", "int32"),
                ir.const(0),
                ir.const(0) + ir.const(2),
                ir.const(1),
                ir.ForKind.DEFAULT,
                ir.SeqStmt([mad(), _set("n")]),
            )
        ]

    with pytest.raises(Exception, match="MAD direction depends on control flow"):
        _codegen(body)


def test_break_does_not_restore_unreachable_direction():
    def body(flag, value, mad):
        loop_body = ir.SeqStmt([_set("n"), ir.Evaluate(ir.Call(tvm_tirx.ir.Op.get("tl.loop_break"), [], ty="handle")), _set("m")])
        return [ir.While(ir.const(True, "bool"), loop_body), mad()]

    assert _mad_directions(_codegen(body)) == [True]


@pytest.mark.parametrize("kind", ["then", "else", "while", "for"])
def test_setter_only_control_flow_is_valid_python(kind):
    def body(flag, value, mad):
        if kind == "then":
            stmt = _if(flag > 0, _set("m"))
        elif kind == "else":
            stmt = _if(flag > 0, mad(), _set("m"))
        elif kind == "while":
            stmt = ir.While(flag > 0, _set("m"))
        else:
            stmt = ir.For(
                ir.Var("i", "int32"),
                ir.const(0),
                ir.const(0) + ir.const(1),
                ir.const(1),
                ir.ForKind.UNROLLED,
                _set("m"),
                annotations={"pragma_unroll_explicit": True},
            )
        return [stmt, mad()]

    _codegen(body)  # The shared verifier parses the generated Python source.


@pytest.mark.parametrize("direction", ["m", "n"])
def test_l1_gemm_direction(direction):
    @T.prim_func
    def main(A: T.Tensor((16, 16), "bfloat16"), B: T.Tensor((16, 16), "bfloat16"), C: T.Tensor((16, 16), "float32")):
        with T.Kernel(1):
            a = T.alloc_l1((16, 16), "bfloat16")
            b = T.alloc_l1((16, 16), "bfloat16")
            c = T.alloc_l0c((16, 16), "float32")
            ub = T.alloc_shared((16, 16), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.set_mmad_direction(direction)
            T.gemm(a, b, c, transpose_B=True, clear_accum=True)
            T.copy(c, ub)
            T.copy(ub, C)

    if direction == "n":
        with pytest.raises(Exception, match="PTO L1 GEMM does not support N MAD direction"):
            tilelang.lower(main, target="pto")
    else:
        source = tilelang.lower(main, target="pto").kernel_source
        ast.parse(source)
        call = next(line for line in source.splitlines() if ".run_l1_tile(" in line)
        assert "n_dir" not in call


@pytest.mark.parametrize("direction", ["m", "n"])
def test_blockscaled_l1_direction_is_not_silently_ignored(direction):
    @T.prim_func
    def main(
        A: T.Tensor((16, 128), "float8_e4m3fn"),
        B: T.Tensor((16, 128), "float8_e4m3fn"),
        SA: T.Tensor((2, 16), "uint16"),
        SB: T.Tensor((2, 16), "uint16"),
        C: T.Tensor((16, 16), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_l1((16, 128), "float8_e4m3fn")
            b = T.alloc_l1((16, 128), "float8_e4m3fn")
            sa = T.alloc_l1((16, 2), "uint16")
            sb = T.alloc_l1((16, 2), "uint16")
            c = T.alloc_l0c((16, 16), "float32")
            ub = T.alloc_shared((16, 16), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(SA, sa, transpose=True)
            T.copy(SB, sb, transpose=True)
            T.set_mmad_direction(direction)
            T.gemm_blockscaled(a, b, c, sa, sb, transpose_B=True, clear_accum=True)
            T.copy(c, ub)
            T.copy(ub, C)

    if direction == "n":
        with pytest.raises(Exception, match="blockscaled L1 GEMM does not support N MAD direction"):
            tilelang.lower(main, target="pto")
    else:
        source = tilelang.lower(main, target="pto").kernel_source
        ast.parse(source)
        assert "PTOBlockscaledGemmL1Template" in source


def test_scheduled_l0_branches_preserve_direction():
    @T.prim_func
    def main(
        A: T.Tensor((16, 16), "bfloat16"),
        B: T.Tensor((16, 16), "bfloat16"),
        C: T.Tensor((2, 16, 16), "float32"),
        flag: T.int32,
    ):
        with T.Kernel(1):
            a = T.alloc_l1((16, 16), "bfloat16")
            b = T.alloc_l1((16, 16), "bfloat16")
            a0 = T.alloc_l0a((16, 16), "bfloat16")
            b0 = T.alloc_l0b((16, 16), "bfloat16")
            c0 = T.alloc_l0c((16, 16), "float32")
            c1 = T.alloc_l0c((16, 16), "float32")
            ub = T.alloc_shared((16, 16), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.copy(a, a0)
            T.copy(b, b0)
            if flag > 0:
                T.set_mmad_direction("n")
                T.gemm(a0, b0, c0, transpose_B=True, clear_accum=True)
                T.set_mmad_direction("m")
            else:
                T.gemm(a0, b0, c0, transpose_B=True, clear_accum=True)
            T.gemm(a0, b0, c1, transpose_B=True, clear_accum=True)
            T.copy(c0, ub)
            T.copy(ub, C[0, :, :])
            T.copy(c1, ub)
            T.copy(ub, C[1, :, :])

    source = tilelang.lower(main, target="pto").kernel_source
    ast.parse(source)
    assert _mad_directions(source) == [True, False, False]


def test_nz_post_copy_uses_independent_storage_pitches():
    src = tensor((4096,), "bfloat16", name="src", scope="shared.dyn")
    dst = tensor((4096,), "bfloat16", name="dst", scope="shared.l1")
    call = ir.Evaluate(
        ir.Call(
            tvm_tirx.ir.Op.get("tl.ascend_nd2nz_post_copy"),
            [tirx.address_of(dst[512]), tirx.address_of(src[128]), 8, 64, 64, 64, "bfloat16_t"],
            ty="void",
        )
    )
    body = ir.SeqStmt([allocate(src), allocate(dst), call])
    target = tvm_tirx.target.Target({"kind": "ascend", "keys": ["pto"]})
    func = tirx.Function([], body).with_attr({"global_symbol": "main", "target": target})
    source = tvm_ffi.get_global_func("target.build.tilelang_pto")(tvm_tirx.IRModule({"main": func}), target).inspect_source()
    ast.parse(source)
    assert ", 8, nburst=(4, 56, 56))" in source


@pytest.mark.parametrize("num_guards", [16, 32, 64])
@pytest.mark.parametrize("with_mad", [False, True])
def test_repeated_mutable_guards_do_not_expand_direction_state(num_guards, with_mad):
    def body(flag, value, mad):
        stores = [_if(value[0] > 0, ir.TensorStore(value, [0], i)) for i in range(num_guards)]
        # A pure outer guard makes the direction predicate nonconstant. The
        # mutable inner guards preserve it on both paths, requiring idempotent
        # merges even when MAD analysis cannot be skipped.
        inner = [_set("n"), *stores]
        if with_mad:
            inner.append(mad())
        inner.append(_set("m"))
        result = [_if(flag > 0, ir.SeqStmt(inner))]
        if with_mad:
            result.append(mad())
        return result

    assert _mad_directions(_codegen(body)) == ([True, False] if with_mad else [])
