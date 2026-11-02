import pytest

import tilelang
import tilelang.ppu.language as T
from tilelang import tvm
from tilelang.contrib.hgcc import have_pdl
from tvm import tirx


PPU_TARGET = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})


def _build_expr_source(expr, params):
    build = tvm.get_global_func("target.build.tilelang_ppu_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the PPU code generator")
    func = tirx.PrimFunc(params, tirx.Evaluate(expr))
    func = func.with_attr("global_symbol", "main")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    return build(tvm.IRModule({"main": func}), PPU_TARGET).inspect_source()


def _kernel_body(source):
    return source[source.rindex("__global__") :]


@pytest.mark.parametrize("arch", ["ppu_10", "ppu_15"])
def test_ppu_pdl_capability(arch):
    target = tvm.target.Target({"kind": "ppu", "arch": arch})
    assert not have_pdl(target)


@pytest.mark.parametrize(
    ("pdl_op", "expected"),
    [
        ("pdl_trigger", "hggcTriggerProgrammaticLaunchCompletion"),
        ("pdl_sync", "hggcGridDependencySynchronize"),
    ],
)
def test_ppu_pdl_codegen_follows_have_pdl(pdl_op, expected, monkeypatch):
    with pytest.raises(ValueError, match="PDL is not supported on PPU backend"):
        _build_expr_source(getattr(T, pdl_op)(), [])

    monkeypatch.setattr("tilelang.ppu.backend.hgcc._PPU_PDL_SUPPORTED_ARCHS", frozenset({15}))
    assert expected in _build_expr_source(getattr(T, pdl_op)(), [])


def test_ppu_clamp_uses_fallback_lowering():
    value = T.Var("value", "float32x4")
    func = tirx.PrimFunc([value], tirx.Evaluate(T.clamp(value, -1.0, 1.0)))
    func = func.with_attr("target", PPU_TARGET)
    lowered = tilelang.transform.LowerIntrin()(tvm.IRModule({"main": func}))

    calls = []
    tirx.stmt_functor.post_order_visit(
        lowered["main"].body,
        lambda node: calls.append(node) if isinstance(node, tirx.Call) else None,
    )
    clamp_op = tirx.op.Op.get("tl.clamp")
    isnan_op = tirx.op.Op.get("tirx.isnan")
    assert not any(call.op.same_as(clamp_op) for call in calls)
    assert any(call.op.same_as(isnan_op) for call in calls)


def test_ppu_python_assert_uses_device_helper():
    if tvm.get_global_func("target.build.tilelang_ppu_without_compile", allow_missing=True) is None:
        pytest.skip("TileLang was built without the PPU code generator")

    @T.prim_func
    def main(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            tid = T.get_thread_binding()
            assert tid < 32, 'bad "thread"'
            B[tid] = A[tid]

    with PPU_TARGET:
        source = tilelang.lower(main, target=PPU_TARGET, enable_device_compile=False).kernel_source
    body = _kernel_body(source)
    assert "device_assert_with_msg" in body
    assert 'bad \\"thread\\"' in body
    assert "TVMFFIErrorSetRaisedFromCStrParts" not in body
    assert "return -1" not in body


@pytest.mark.parametrize(
    ("dtype", "expected", "count"),
    [("bool", "!", 1), ("boolx4", "!bool(", 4), ("int32x4", "^", 4)],
)
def test_ppu_bitwise_not_codegen(dtype, expected, count):
    value = tirx.Var("value", dtype)
    body = _kernel_body(_build_expr_source(~value, [value]))
    assert body.count(expected) >= count
    assert "~value" not in body


@pytest.mark.parametrize("dtype", ["int4", "uint4"])
@pytest.mark.parametrize(
    ("lanes", "carrier"),
    [(4, "int16_t)"), (8, "int)"), (16, "int2("), (32, "int4(")],
)
def test_ppu_nonconstant_int4_broadcast(dtype, lanes, carrier):
    value = tirx.Var("value", "int32")
    scalar = tirx.Cast(dtype, value)
    body = _kernel_body(_build_expr_source(tirx.Broadcast(scalar, lanes), [value]))
    if dtype == "uint4":
        carrier = "u" + carrier
    assert carrier in body
    assert "static_cast<unsigned int>" in body
    assert "make_int16_t" not in body
