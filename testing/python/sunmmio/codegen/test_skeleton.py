import json
import os

import pytest
import tilelang.testing
import tilelang.language as T
from tilelang.layout import make_zz_layout
from tilelang import tvm as tvm
from tilelang.utils.target import determine_target

from testing.python.sunmmio.common.compile_pipeline import target
from testing.python.sunmmio.common.codegen_validation import print_sunmmio_codegen_debug

# os.environ["SUNMMIO_TEST_LOG_IR"] = "1"
os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")


def _to_device_kernel_func(func):
    return func.with_attr("global_symbol", "main").with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))


def _primfunc_from_stmt(stmt):
    return _to_device_kernel_func(tvm.tir.PrimFunc([], stmt))


def _resolve_transfer_units(mod, target):
    mod = tvm.tir.transform.BindTarget(target)(mod)
    return tilelang.transform.ResolveSunmmioUnit()(mod)


def build_sunmmio_module_without_compile(func):
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": _to_device_kernel_func(func)})
    mod = _resolve_transfer_units(mod, target)
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    return builder(mod, target, "suvm")


def build_sunmmio_source_without_compile(func):
    src = build_sunmmio_module_without_compile(func).inspect_source()
    print_sunmmio_codegen_debug(label="TVM Kernel", ir_obj=func, mlir_src=src)
    return src


def build_sunmmio_source_from_stmt(stmt):
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": _primfunc_from_stmt(stmt)})
    mod = _resolve_transfer_units(mod, target)
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    src = builder(mod, target, "suvm").inspect_source()
    print_sunmmio_codegen_debug(label="TVM Kernel", ir_obj=mod["main"], mlir_src=src)
    return src


@target("Sunmmio")
def make_scalar_control_kernel():
    i = tvm.tir.Var("i", "int32")
    j = tvm.tir.Var("j", "int32")
    v = tvm.tir.Var("v", "int32")

    expr_then = tvm.tir.Select(
        tvm.tir.LT(j, tvm.tir.IntImm("int32", 8)),
        tvm.tir.Add(i, j),
        tvm.tir.Sub(i, j),
    )
    let_stmt = tvm.tir.LetStmt(
        v,
        tvm.tir.Add(i, j),
        tvm.tir.Evaluate(tvm.tir.Max(v, tvm.tir.IntImm("int32", 0))),
    )
    then_stmt = tvm.tir.SeqStmt([tvm.tir.Evaluate(expr_then), let_stmt])
    else_stmt = tvm.tir.Evaluate(tvm.tir.Min(i, j))
    inner_if = tvm.tir.IfThenElse(tvm.tir.LE(j, tvm.tir.IntImm("int32", 12)), then_stmt, else_stmt)
    inner_for = tvm.tir.For(j, 0, 16, tvm.tir.ForKind.SERIAL, inner_if)
    outer_for = tvm.tir.For(i, 0, 8, tvm.tir.ForKind.SERIAL, inner_for)
    return _primfunc_from_stmt(outer_for)


@target("Sunmmio")
def make_explicit_step_for_kernel():
    i = tvm.tir.Var("i", "int32")
    loop = tvm.tir.For(
        i,
        0,
        8,
        tvm.tir.ForKind.SERIAL,
        tvm.tir.Evaluate(i),
        step=tvm.tir.IntImm("int32", 2),
    )
    return _primfunc_from_stmt(loop)


@target("Sunmmio")
def make_alloc_scope_kernel():
    bf16 = tvm.ir.PrimType("bfloat16")
    one = tvm.tir.IntImm("bool", 1)
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))

    rsram = tvm.tir.Var("rsram_buf", tvm.ir.PointerType(bf16, "shared.rsram"))
    wsram = tvm.tir.Var("wsram_buf", tvm.ir.PointerType(bf16, "shared.wsram"))
    asram = tvm.tir.Var("asram_buf", tvm.ir.PointerType(bf16, "shared.asram"))
    rsram_buf = tvm.tir.decl_buffer((16, 16), "bfloat16", name="Rsram", data=rsram, scope="shared.rsram")
    wsram_buf = tvm.tir.decl_buffer((16, 16), "bfloat16", name="Wsram", data=wsram, scope="shared.wsram")
    asram_buf = tvm.tir.decl_buffer((16, 16), "bfloat16", name="Asram", data=asram, scope="shared.asram")

    stmt = tvm.tir.Allocate(rsram, "bfloat16", [16, 16], one, body)
    stmt = tvm.tir.Allocate(wsram, "bfloat16", [16, 16], one, stmt)
    stmt = tvm.tir.Allocate(asram, "bfloat16", [16, 16], one, stmt)
    stmt = tvm.tir.DeclBuffer(rsram_buf, stmt)
    stmt = tvm.tir.DeclBuffer(wsram_buf, stmt)
    stmt = tvm.tir.DeclBuffer(asram_buf, stmt)
    return _primfunc_from_stmt(stmt)


@target("Sunmmio")
def make_helper_consumed_expr_root_kernel():
    data = tvm.tir.Var(
        "local_data",
        tvm.ir.PointerType(tvm.ir.PrimType("int32"), "local.var"),
    )
    buffer = tvm.tir.decl_buffer((1,), "int32", name="Local", data=data, scope="local.var")

    # EmitLocalVarLoad consumes and simplifies this root expression directly,
    # without dispatching it through EvalExpr.
    helper_consumed_index = tvm.tir.Mul(tvm.tir.IntImm("int32", 1), tvm.tir.IntImm("int32", 0))
    body = tvm.tir.Evaluate(tvm.tir.BufferLoad(buffer, [helper_consumed_index]))
    stmt = tvm.tir.Allocate(
        data,
        "int32",
        [tvm.tir.IntImm("int32", 1)],
        tvm.tir.IntImm("bool", 1),
        body,
    )
    stmt = tvm.tir.DeclBuffer(buffer, stmt)
    return _primfunc_from_stmt(stmt)


@target("Sunmmio")
def make_ret_evaluate_kernel():
    return _primfunc_from_stmt(tvm.tir.Evaluate(tvm.tir.ret(0)))


@target("Sunmmio")
def make_allocate_without_decl_buffer_kernel():
    bf16 = tvm.ir.PrimType("bfloat16")
    one = tvm.tir.IntImm("bool", 1)
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    asram = tvm.tir.Var("asram_buf", tvm.ir.PointerType(bf16, "shared.asram"))
    stmt = tvm.tir.Allocate(asram, "bfloat16", [16, 16], one, body)
    return _primfunc_from_stmt(stmt)


@target("Sunmmio")
def make_invalid_dma_shape_kernel():
    bf16 = tvm.ir.PrimType("bfloat16")
    src_data = tvm.tir.Var("src_data", tvm.ir.PointerType(bf16, "shared.rsram"))
    dst_data = tvm.tir.Var("dst_data", tvm.ir.PointerType(bf16, "shared.rsram"))
    src_buf = tvm.tir.decl_buffer((32, 32), "bfloat16", name="Src", data=src_data, scope="shared.rsram")
    dst_buf = tvm.tir.decl_buffer((16, 32), "bfloat16", name="Dst", data=dst_data, scope="shared.rsram")

    def region(buf, access, m, n):
        return tvm.tir.call_intrin(
            "handle",
            tvm.ir.Op.get("tl.tileop.region"),
            tvm.tir.BufferLoad(
                buf,
                [tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 0)],
            ),
            tvm.tir.IntImm("int32", access),
            tvm.tir.IntImm("int32", m),
            tvm.tir.IntImm("int32", n),
        )

    dma = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.dma_copy"),
        [region(src_buf, 1, 32, 32), region(dst_buf, 2, 16, 32), tvm.tir.IntImm("int32", 0)],
    )
    stmt = tvm.tir.DeclBuffer(src_buf, tvm.tir.DeclBuffer(dst_buf, tvm.tir.Evaluate(dma)))
    return _to_device_kernel_func(tvm.tir.PrimFunc([src_data, dst_data], stmt))


@target("Sunmmio")
def make_layout_transform_kernel():
    bf16 = tvm.ir.PrimType("bfloat16")
    src_data = tvm.tir.Var("src_data", tvm.ir.PointerType(bf16, "shared.rsram"))
    dst_data = tvm.tir.Var("dst_data", tvm.ir.PointerType(bf16, "shared.rsram"))
    src_buf = tvm.tir.decl_buffer((32, 32), "bfloat16", name="Src", data=src_data, scope="shared.rsram")
    dst_buf = tvm.tir.decl_buffer((32, 32), "bfloat16", name="Dst", data=dst_data, scope="shared.rsram")

    def region(buf, access):
        return tvm.tir.call_intrin(
            "handle",
            tvm.ir.Op.get("tl.tileop.region"),
            tvm.tir.BufferLoad(
                buf,
                [tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 0)],
            ),
            tvm.tir.IntImm("int32", access),
            tvm.tir.IntImm("int32", 32),
            tvm.tir.IntImm("int32", 32),
        )

    transform = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.sunmmio_layout_transform"),
        [region(src_buf, 1), region(dst_buf, 2)],
    )
    stmt = tvm.tir.DeclBuffer(src_buf, tvm.tir.DeclBuffer(dst_buf, tvm.tir.Evaluate(transform)))
    return _to_device_kernel_func(tvm.tir.PrimFunc([src_data, dst_data], stmt)).with_attr(
        "layout_map",
        {src_buf: make_zz_layout(src_buf, axes=[0, 1], block_shape=(32, 32))},
    )


@target("Sunmmio")
def make_dynamic_broadcast_mask_kernel():
    bf16 = tvm.ir.PrimType("bfloat16")
    src_data = tvm.tir.Var("src_data", tvm.ir.PointerType(bf16, "shared.rsram"))
    dst_data = tvm.tir.Var("dst_data", tvm.ir.PointerType(bf16, "shared.asram"))
    src_buf = tvm.tir.decl_buffer((32, 32), "bfloat16", name="Src", data=src_data, scope="shared.rsram")
    dst_buf = tvm.tir.decl_buffer((32, 32), "bfloat16", name="Dst", data=dst_data, scope="shared.asram")

    def region(buf, access):
        return tvm.tir.call_intrin(
            "handle",
            tvm.ir.Op.get("tl.tileop.region"),
            tvm.tir.BufferLoad(
                buf,
                [tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 0)],
            ),
            tvm.tir.IntImm("int32", access),
            tvm.tir.IntImm("int32", 32),
            tvm.tir.IntImm("int32", 32),
        )

    bx = tvm.tir.Var("bx", "int32")
    bx_i64 = tvm.tir.Cast("int64", bx)
    one = tvm.tir.IntImm("int64", 1)
    mask = tvm.tir.bitwise_or(
        tvm.tir.shift_left(one, bx_i64),
        tvm.tir.shift_left(one, bx_i64 + tvm.tir.IntImm("int64", 1)),
    )
    broadcast = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.broadcast_"),
        [
            region(src_buf, 1),
            region(dst_buf, 2),
            tvm.tir.IntImm("int32", 0),
            mask,
            tvm.tir.IntImm("int32", 0),
            bx,
        ],
    )
    stmt = tvm.tir.For(
        bx,
        tvm.tir.IntImm("int32", 0),
        tvm.tir.IntImm("int32", 3),
        tvm.tir.ForKind.SERIAL,
        tvm.tir.Evaluate(broadcast),
    )
    stmt = tvm.tir.DeclBuffer(src_buf, tvm.tir.DeclBuffer(dst_buf, stmt))
    return _to_device_kernel_func(tvm.tir.PrimFunc([src_data, dst_data], stmt))


@target("Sunmmio")
def make_reusable_barrier_kernel():
    mask = tvm.tir.IntImm("int64", 15)
    barrier_init = tvm.tir.Call("handle", tvm.ir.Op.get("tl.barrier_init"), [mask])
    barrier_wait = tvm.tir.Call("handle", tvm.ir.Op.get("tl.barrier_arrive_and_wait"), [mask])
    stmt = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(barrier_init),
            tvm.tir.Evaluate(barrier_wait),
            tvm.tir.Evaluate(barrier_wait),
        ]
    )
    return _primfunc_from_stmt(stmt)


@target("Sunmmio")
def make_dynamic_barrier_kernel():
    mask = tvm.tir.Var("mask", "int64")
    barrier_init = tvm.tir.Call("handle", tvm.ir.Op.get("tl.barrier_init"), [mask])
    barrier_wait = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.barrier_arrive_and_wait"),
        [mask],
    )
    stmt = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(barrier_init),
            tvm.tir.Evaluate(barrier_wait),
        ]
    )
    return _to_device_kernel_func(tvm.tir.PrimFunc([mask], stmt))


@target("Sunmmio")
def make_dynamic_barrier_candidates_kernel():
    bx = tvm.tir.Var("bx", "int32")
    bx_i64 = tvm.tir.Cast("int64", bx)
    mask = tvm.tir.shift_left(
        tvm.tir.IntImm("int64", 15),
        bx_i64 * tvm.tir.IntImm("int64", 4),
    )
    candidates = [15, 240, 3840, 61440]
    barrier_init = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.barrier_init"),
        [tvm.tir.IntImm("int64", -1)] + [tvm.tir.IntImm("int64", candidate) for candidate in candidates],
    )
    barrier_wait = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.barrier_arrive_and_wait"),
        [mask] + [tvm.tir.IntImm("int64", candidate) for candidate in candidates],
    )
    stmt = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(barrier_init),
            tvm.tir.For(
                bx,
                tvm.tir.IntImm("int32", 0),
                tvm.tir.IntImm("int32", 4),
                tvm.tir.ForKind.SERIAL,
                tvm.tir.Evaluate(barrier_wait),
            ),
        ]
    )
    return _primfunc_from_stmt(stmt)


@target("Sunmmio")
def make_block_realize_kernel():
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    block = tvm.tir.Block([], [], [], "B", body)
    stmt = tvm.tir.BlockRealize([], tvm.tir.IntImm("bool", 1), block)
    return _primfunc_from_stmt(stmt)


@target("Sunmmio")
def make_decl_buffer_kernel():
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    buf = tvm.tir.decl_buffer((16, 16), "bfloat16", name="A")
    stmt = tvm.tir.DeclBuffer(buf, body)
    return _primfunc_from_stmt(stmt)


@target("Sunmmio")
def make_buffer_realize_kernel():
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    buf = tvm.tir.decl_buffer((16, 16), "bfloat16", name="A")
    bounds = [
        tvm.ir.Range.from_min_extent(0, 16),
        tvm.ir.Range.from_min_extent(0, 16),
    ]
    stmt = tvm.tir.BufferRealize(buf, bounds, tvm.tir.IntImm("bool", 1), body)
    return _primfunc_from_stmt(stmt)


@target("Sunmmio")
def make_buffer_load_kernel():
    buf = tvm.tir.decl_buffer((16, 16), "bfloat16", name="A")
    stmt = tvm.tir.Evaluate(
        tvm.tir.BufferLoad(
            buf,
            [tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 0)],
        )
    )
    return _primfunc_from_stmt(stmt)


@target("Sunmmio")
def make_buffer_store_kernel():
    buf = tvm.tir.decl_buffer((16, 16), "bfloat16", name="A")
    stmt = tvm.tir.BufferStore(
        buf,
        tvm.tir.FloatImm("bfloat16", 1.0),
        [tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 0)],
    )
    return _primfunc_from_stmt(stmt)


@target("Sunmmio")
def make_real_tilelang_frontend_kernel():
    @T.prim_func
    def main():
        with T.Kernel(1, 1, threads=1) as (bx, by):
            for i in T.serial(0, 8):
                T.evaluate(i + 1)

    return main


def _assert_coverage_report_complete(report_path):
    assert report_path.exists(), f"coverage report not generated: {report_path}"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required_keys = [
        "expected_node_types",
        "visited_node_types",
        "missing_node_types",
        "expected_call_ops",
        "visited_call_ops",
        "missing_call_ops",
    ]
    for domain in ("main", "tiles"):
        assert domain in report, f"missing coverage domain: {domain}"
        for key in required_keys:
            assert key in report[domain], f"missing {domain} coverage key: {key}"
        assert report[domain]["missing_node_types"] == []
        assert report[domain]["missing_call_ops"] == []


def test_sunmmio_codegen_without_compile_emits_nonempty_suvm_source():
    src = build_sunmmio_source_without_compile(make_scalar_control_kernel())
    assert src.strip()
    assert "module" in src
    assert "func.func @main" in src


def test_sunmmio_codegen_preserves_explicit_for_step():
    src = build_sunmmio_source_without_compile(make_explicit_step_for_kernel())
    for_line = next(line.strip() for line in src.splitlines() if "scf.for" in line)
    step_value = for_line.split(" step ", 1)[1].split(" ", 1)[0]
    step_definition = next(line.strip() for line in src.splitlines() if line.strip().startswith(f"{step_value} ="))
    if "arith.index_cast" in step_definition:
        step_source = step_definition.split("arith.index_cast ", 1)[1].split(" ", 1)[0]
        assert f"{step_source} = arith.constant 2 : i32" in src
    else:
        assert "arith.constant 2 : index" in step_definition


def test_sunmmio_codegen_while_emits_scf_while():
    cond = tvm.tir.LT(tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 1))
    body = tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0))
    stmt = tvm.tir.While(cond, body)
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": _primfunc_from_stmt(stmt)})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    src = builder(mod, target, "suvm").inspect_source()
    assert "scf.while" in src
    assert "scf.condition" in src
    assert "scf.yield" in src


def test_sunmmio_codegen_lowers_reusable_barrier():
    src = build_sunmmio_source_without_compile(make_reusable_barrier_kernel())
    assert "suvm.barrier.init mask = 15 : !suvm.barrier" in src
    assert src.count("suvm.barrier.init") == 1
    assert src.count("suvm.barrier.arrive_and_wait") == 2
    assert "sunmmio.fake" not in src


def test_sunmmio_codegen_lowers_dynamic_barrier_mask():
    src = build_sunmmio_source_without_compile(make_dynamic_barrier_kernel())
    assert "suvm.barrier.init mask = %" in src
    assert " : i64 -> !suvm.barrier" in src
    assert src.count("suvm.barrier.init") == 1
    assert src.count("suvm.barrier.arrive_and_wait") == 1


def test_sunmmio_codegen_lowers_dynamic_barrier_candidates():
    src = build_sunmmio_source_without_compile(make_dynamic_barrier_candidates_kernel())
    for mask in [15, 240, 3840, 61440]:
        assert f"suvm.barrier.init mask = {mask} : !suvm.barrier" in src
    assert src.count("suvm.barrier.init") == 4
    assert src.count("suvm.barrier.arrive_and_wait") == 4
    assert "arith.shli" in src
    assert "arith.cmpi eq" in src
    assert "scf.if" in src
    assert "sunmmio.fake" not in src


def test_sunmmio_codegen_lowers_dynamic_broadcast_mask():
    src = build_sunmmio_source_without_compile(make_dynamic_broadcast_mask_kernel())
    assert "arith.shli" in src
    assert "arith.ori" in src
    assert "suvm.mcast_tok" in src
    assert "suvm.sync  hlink" in src
    assert "sunmmio.fake" not in src


def test_sunmmio_codegen_lowers_layout_transform():
    src = build_sunmmio_source_without_compile(make_layout_transform_kernel())
    assert "suvm.transform_layout_async" in src
    assert "sunmmio.fake" not in src


def test_sunmmio_codegen_rejects_unresolved_odma_unit():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": make_layout_transform_kernel()})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="expects src region, dst region, and odma_unit"):
        builder(mod, target, "suvm")


def test_sunmmio_codegen_module_verification_failure_fails_loudly():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": make_invalid_dma_shape_kernel()})
    mod = _resolve_transfer_units(mod, target)
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="SunMMIO MLIR module verification failed"):
        builder(mod, target, "suvm")


def test_sunmmio_codegen_shuffle_fails_loudly():
    shuffle = tvm.tir.Shuffle(
        [tvm.tir.Broadcast(tvm.tir.IntImm("int32", 7), 4)],
        [tvm.tir.IntImm("int32", 0)],
    )
    stmt = tvm.tir.Evaluate(shuffle)
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": _primfunc_from_stmt(stmt)})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="CodeGenTileLangSunMMIO unsupported expr: tir.Shuffle"):
        builder(mod, target, "suvm")


def test_sunmmio_codegen_ramp_fails_loudly():
    ramp = tvm.tir.Ramp(
        tvm.tir.IntImm("int32", 0),
        tvm.tir.IntImm("int32", 1),
        4,
    )
    stmt = tvm.tir.Evaluate(ramp)
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": _primfunc_from_stmt(stmt)})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="Generic SunMMIO ramp expression lowering is unsupported"):
        builder(mod, target, "suvm")


def test_sunmmio_codegen_broadcast_fails_loudly():
    broadcast = tvm.tir.Broadcast(tvm.tir.IntImm("int32", 7), 4)
    stmt = tvm.tir.Evaluate(broadcast)
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": _primfunc_from_stmt(stmt)})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="Generic SunMMIO broadcast expression lowering is unsupported"):
        builder(mod, target, "suvm")


def test_sunmmio_codegen_unsupported_call_fails_loudly():
    call = tvm.tir.call_pure_extern(
        "int32",
        "unsupported_external_call",
        tvm.tir.IntImm("int32", 1),
    )
    stmt = tvm.tir.Evaluate(call)
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": _primfunc_from_stmt(stmt)})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="Unsupported SunMMIO call lowering.*tir.call_pure_extern"):
        builder(mod, target, "suvm")


def test_sunmmio_codegen_unbound_tir_var_fails_loudly():
    missing = tvm.tir.Var("missing_runtime_var", "int32")
    stmt = tvm.tir.Evaluate(tvm.tir.Add(missing, tvm.tir.IntImm("int32", 1)))
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": _primfunc_from_stmt(stmt)})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(Exception, match="unbound TIR var.*missing_runtime_var"):
        builder(mod, target, "suvm")


def test_sunmmio_codegen_compile_path_not_implemented():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": make_scalar_control_kernel()})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio")
    with pytest.raises(Exception, match="not implemented yet"):
        builder(mod, target)


def test_sunmmio_codegen_block_realize_fails_loudly():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": make_block_realize_kernel()})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(
        Exception,
        match="BlockRealizeNode should be eliminated by LowerOpaqueBlock before SunMMIO codegen",
    ):
        builder(mod, target, "suvm")


def test_sunmmio_codegen_decl_buffer_is_benign_wrapper():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": make_decl_buffer_kernel()})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    src = builder(mod, target, "suvm").inspect_source()
    assert src.strip()
    assert "module" in src
    assert "func.func @main" in src


def test_sunmmio_codegen_buffer_realize_fails_loudly():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": make_buffer_realize_kernel()})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(
        Exception,
        match="BufferRealizeNode should be lowered into a concrete view/alias representation before SunMMIO codegen",
    ):
        builder(mod, target, "suvm")


def test_sunmmio_codegen_buffer_load_fails_loudly():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": make_buffer_load_kernel()})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(
        Exception,
        match="Sunmmio scalar BufferLoad from DRAM/global must be legalized by staging through RSRAM before codegen",
    ):
        builder(mod, target, "suvm")


def test_sunmmio_codegen_buffer_store_fails_loudly():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": make_buffer_store_kernel()})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(
        Exception,
        match="Sunmmio scalar BufferStore to DRAM/global must be legalized by staging through RSRAM before codegen",
    ):
        builder(mod, target, "suvm")


def test_sunmmio_codegen_allocate_without_decl_buffer_fails_loudly():
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": make_allocate_without_decl_buffer_kernel()})
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    with pytest.raises(
        Exception,
        match="SunMMIO SUVM allocate cannot find buffer for variable asram_buf",
    ):
        builder(mod, target, "suvm")


@pytest.mark.parametrize(
    "kernel_name,kernel_factory",
    [
        ("scalar_control", make_scalar_control_kernel),
        ("alloc_scope", make_alloc_scope_kernel),
        pytest.param(
            "real_tilelang_frontend",
            make_real_tilelang_frontend_kernel,
            marks=pytest.mark.xfail(
                reason=(
                    "generic SunMMIO codegen now requires pre-lowered backend IR; "
                    "this real frontend case still bypasses the full SunMMIO "
                    "lowering pipeline. Revisit when the full pass pipeline lands."
                ),
                strict=True,
            ),
        ),
    ],
)
def test_sunmmio_codegen_coverage_report_has_no_missing_entries(tmp_path, kernel_name, kernel_factory):
    report_path = tmp_path / f"codegen_coverage_{kernel_name}.json"
    old_path = os.environ.get("TL_SUNMMIO_CODEGEN_COVERAGE_PATH")
    old_strict = os.environ.get("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT")
    os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_PATH"] = str(report_path)
    os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_STRICT"] = "1"
    try:
        _ = build_sunmmio_source_without_compile(kernel_factory())
    finally:
        if old_path is None:
            os.environ.pop("TL_SUNMMIO_CODEGEN_COVERAGE_PATH", None)
        else:
            os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_PATH"] = old_path
        if old_strict is None:
            os.environ.pop("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", None)
        else:
            os.environ["TL_SUNMMIO_CODEGEN_COVERAGE_STRICT"] = old_strict

    _assert_coverage_report_complete(report_path)


def test_sunmmio_codegen_coverage_tracks_helper_consumed_expr_root(tmp_path, monkeypatch):
    report_path = tmp_path / "codegen_coverage_helper_consumed_expr.json"
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_PATH", str(report_path))
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")

    build_sunmmio_source_without_compile(make_helper_consumed_expr_root_kernel())

    _assert_coverage_report_complete(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    main = report["main"]
    assert "tir.Mul" in main["expected_node_types"]
    assert "tir.Mul" in main["visited_node_types"]


def test_sunmmio_codegen_coverage_tracks_ret_call_node(tmp_path, monkeypatch):
    report_path = tmp_path / "codegen_coverage_ret.json"
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_PATH", str(report_path))
    monkeypatch.setenv("TL_SUNMMIO_CODEGEN_COVERAGE_STRICT", "1")

    build_sunmmio_source_without_compile(make_ret_evaluate_kernel())

    _assert_coverage_report_complete(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    main = report["main"]
    tiles = report["tiles"]
    assert "tir.Call" in main["expected_node_types"]
    assert "tir.Call" in main["visited_node_types"]
    assert "tir.ret" in main["expected_call_ops"]
    assert "tir.ret" in main["visited_call_ops"]
    assert tiles["expected_node_types"] == []
    assert tiles["visited_node_types"] == []
    assert tiles["expected_call_ops"] == []
    assert tiles["visited_call_ops"] == []


if __name__ == "__main__":
    tilelang.testing.main()
