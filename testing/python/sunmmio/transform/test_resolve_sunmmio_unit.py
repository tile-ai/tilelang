import pytest

import tilelang
from tilelang import tvm
from tilelang.utils.target import SUNMMIO_TARGET_DESC


TRANSFER_OPS = {
    "tl.dma_copy",
    "tl.sunmmio_layout_transform",
    "tl.sunmmio_transpose",
    "tl.broadcast_",
}


def _region(buffer, access):
    load = tvm.tir.BufferLoad(
        buffer,
        [tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 0)],
    )
    return tvm.tir.call_intrin(
        "handle",
        tvm.ir.Op.get("tl.tileop.region"),
        load,
        tvm.tir.IntImm("int32", access),
        tvm.tir.IntImm("int32", 32),
        tvm.tir.IntImm("int32", 32),
    )


def _make_transfer_module(op_name, src_scope, dst_scope, *extra_args):
    elem_type = tvm.ir.PrimType("bfloat16")
    src_data = tvm.tir.Var("src", tvm.ir.PointerType(elem_type, src_scope))
    dst_data = tvm.tir.Var("dst", tvm.ir.PointerType(elem_type, dst_scope))
    src = tvm.tir.decl_buffer((32, 32), "bfloat16", name="Src", data=src_data, scope=src_scope)
    dst = tvm.tir.decl_buffer((32, 32), "bfloat16", name="Dst", data=dst_data, scope=dst_scope)
    call = tvm.tir.call_intrin(
        "handle",
        tvm.ir.Op.get(op_name),
        _region(src, 1),
        _region(dst, 2),
        *extra_args,
    )
    body = tvm.tir.DeclBuffer(
        src,
        tvm.tir.DeclBuffer(dst, tvm.tir.Evaluate(call)),
    )
    func = tvm.tir.PrimFunc([src_data, dst_data], body).with_attr("global_symbol", "main")
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    return tvm.tir.transform.BindTarget(target)(tvm.IRModule({"main": func}))


def _dma_offset():
    return tvm.tir.IntImm("int32", 0)


def _odma_unit(name):
    return tvm.tir.call_intrin(
        "handle",
        tvm.ir.Op.get("tl.odma_unit"),
        tvm.tir.StringImm(name),
    )


def _broadcast_args(direction):
    return (
        tvm.tir.IntImm("int32", direction),
        tvm.tir.IntImm("int64", 15),
        tvm.tir.IntImm("int32", 0),
    )


def _find_transfer_call(mod):
    calls = []

    def visit(node):
        if isinstance(node, tvm.tir.Call) and node.op.name in TRANSFER_OPS:
            calls.append(node)

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert len(calls) == 1
    return calls[0]


def _resolved_unit(mod):
    call = _find_transfer_call(mod)
    unit_args = [arg for arg in call.args if isinstance(arg, tvm.tir.Call) and arg.op.name == "tl.odma_unit"]
    assert len(unit_args) == 1
    assert call.args[-1].same_as(unit_args[0])
    return unit_args[0].args[0].value


@pytest.mark.parametrize(
    "op_name,src_scope,dst_scope,extra_args,expected_unit",
    [
        pytest.param(
            "tl.dma_copy",
            "global",
            "shared.rsram",
            (_dma_offset(),),
            "odma0",
            id="global-to-rsram",
        ),
        pytest.param(
            "tl.dma_copy",
            "shared.rsram",
            "global",
            (_dma_offset(),),
            "odma0",
            id="rsram-to-global",
        ),
        pytest.param(
            "tl.dma_copy",
            "shared.rsram",
            "shared.wsram",
            (_dma_offset(),),
            "odma0",
            id="rsram-to-wsram",
        ),
        pytest.param(
            "tl.dma_copy",
            "shared.rsram",
            "shared.asram",
            (_dma_offset(),),
            "odma1",
            id="rsram-to-asram",
        ),
        pytest.param(
            "tl.dma_copy",
            "shared.rsram",
            "shared.rsram",
            (_dma_offset(),),
            "odma1",
            id="rsram-to-rsram",
        ),
        pytest.param(
            "tl.sunmmio_layout_transform",
            "shared.rsram",
            "shared.rsram",
            (),
            "odma1",
            id="layout-transform",
        ),
        pytest.param(
            "tl.sunmmio_transpose",
            "shared.rsram",
            "shared.rsram",
            (),
            "odma1",
            id="transpose",
        ),
        pytest.param(
            "tl.broadcast_",
            "shared.rsram",
            "shared.asram",
            _broadcast_args(0),
            "odma1",
            id="row-multicast",
        ),
        pytest.param(
            "tl.broadcast_",
            "global",
            "shared.wsram",
            _broadcast_args(1),
            "odma0",
            id="col-multicast",
        ),
    ],
)
def test_resolve_sunmmio_unit_routes(
    op_name,
    src_scope,
    dst_scope,
    extra_args,
    expected_unit,
):
    mod = _make_transfer_module(op_name, src_scope, dst_scope, *extra_args)
    resolved = tilelang.transform.ResolveSunmmioUnit()(mod)
    assert _resolved_unit(resolved) == expected_unit


def test_resolve_sunmmio_unit_is_idempotent():
    mod = _make_transfer_module(
        "tl.dma_copy",
        "global",
        "shared.rsram",
        _dma_offset(),
    )
    once = tilelang.transform.ResolveSunmmioUnit()(mod)
    twice = tilelang.transform.ResolveSunmmioUnit()(once)
    assert tvm.ir.structural_equal(once, twice)
    assert _resolved_unit(twice) == "odma0"


def test_resolve_sunmmio_unit_rejects_mismatched_existing_unit():
    mod = _make_transfer_module(
        "tl.dma_copy",
        "global",
        "shared.rsram",
        _dma_offset(),
        _odma_unit("odma1"),
    )
    with pytest.raises(tvm.error.InternalError, match="A4E routing selects odma0"):
        tilelang.transform.ResolveSunmmioUnit()(mod)


def test_resolve_sunmmio_unit_rejects_unsupported_route():
    mod = _make_transfer_module(
        "tl.dma_copy",
        "global",
        "shared.asram",
        _dma_offset(),
    )
    with pytest.raises(tvm.error.InternalError, match="No A4E ODMA route"):
        tilelang.transform.ResolveSunmmioUnit()(mod)
