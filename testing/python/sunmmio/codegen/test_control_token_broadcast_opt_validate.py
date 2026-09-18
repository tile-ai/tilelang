import os

import pytest
import tilelang.testing
from tilelang import tvm
from tilelang.utils.target import determine_target

from testing.python.sunmmio.common.codegen_validation import (
    assert_source_contains,
    validate_suvm_mlir_with_npuir_opt,
    write_sunmmio_codegen_logs,
)


os.environ.setdefault("SUNMMIO_TEST_PRINT", "0")
MCAST_VERIFY_ARGS = ("--verify-each", "--suvm-verify-mcast-sync")


def _to_device_kernel_func(func, name="main"):
    return (
        func.with_attr("global_symbol", name)
        .with_attr("calling_conv", int(tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH))
        .with_attr("tir.is_global_func", True)
    )


def _primfunc_from_stmt(stmt, params=None, name="main"):
    return _to_device_kernel_func(tvm.tir.PrimFunc(params or [], stmt), name)


def _resolve_transfer_units(mod, target, *, inject_sync=False):
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tvm.ffi.get_global_func("tl.transform.ResolveSunmmioUnit")()(mod)
    if inject_sync:
        mod = tvm.ffi.get_global_func("tl.transform.InjectSunmmioSync")()(mod)
    return mod


def _build_sunmmio_source_from_stmt(stmt, params=None, *, inject_sync=False):
    target = determine_target("Sunmmio", return_object=True)
    mod = tvm.IRModule({"main": _primfunc_from_stmt(stmt, params=params)})
    mod = _resolve_transfer_units(mod, target, inject_sync=inject_sync)
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    return mod, builder(mod, target, "suvm").inspect_source()


def _build_sunmmio_source_from_module(mod):
    target = determine_target("Sunmmio", return_object=True)
    mod = _resolve_transfer_units(mod, target)
    builder = tvm.ffi.get_global_func("target.build.tilelang_sunmmio_without_compile")
    return builder(mod, target, "suvm").inspect_source()


def _validate_stmt_codegen(
    stmt,
    tmp_path,
    *,
    mlir_filename,
    expected_fragments=(),
    params=None,
    opt_args=("-suvm-device-validate",),
):
    tir_mod, src = _build_sunmmio_source_from_stmt(stmt, params=params, inject_sync=True)
    assert_source_contains(src, ("module", "suvm.device_arch", "func.func @", *expected_fragments))
    assert "!suvm.token" not in src
    assert "suvm.wait_token" not in src
    write_sunmmio_codegen_logs(case_name=mlir_filename, tir_mod=tir_mod, mlir_src=src)
    validate_suvm_mlir_with_npuir_opt(
        src,
        tmp_path,
        mlir_filename=mlir_filename,
        opt_args=opt_args,
    )
    return src


def _shared_buffers(dtype="bfloat16", shape=(32, 32)):
    elem_ty = tvm.ir.PrimType(dtype)
    src_data = tvm.tir.Var("src_data", tvm.ir.PointerType(elem_ty, "shared.rsram"))
    dst_data = tvm.tir.Var("dst_data", tvm.ir.PointerType(elem_ty, "shared.asram"))
    src_buf = tvm.tir.decl_buffer(shape, dtype, name="Src", data=src_data, scope="shared.rsram")
    dst_buf = tvm.tir.decl_buffer(shape, dtype, name="Dst", data=dst_data, scope="shared.asram")
    return src_data, dst_data, src_buf, dst_buf


def _region(buf, access, extents=(32, 32)):
    return tvm.tir.call_intrin(
        "handle",
        tvm.ir.Op.get("tl.tileop.region"),
        tvm.tir.BufferLoad(buf, [tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 0)]),
        tvm.tir.IntImm("int32", access),
        *[tvm.tir.IntImm("int32", extent) for extent in extents],
    )


def _broadcast(src_buf, dst_buf, *, direction=0, mask=None, src_core=None, extents=(32, 32)):
    args = [
        _region(src_buf, 1, extents=extents),
        _region(dst_buf, 2, extents=extents),
        tvm.tir.IntImm("int32", direction),
        mask if mask is not None else tvm.tir.IntImm("int64", 15),
        tvm.tir.IntImm("int32", 0),
    ]
    if src_core is not None:
        args.append(src_core)
    return tvm.tir.Call("handle", tvm.ir.Op.get("tl.broadcast_"), args)


def _dma(src_buf, dst_buf):
    return tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.dma_copy"),
        [
            _region(src_buf, 1),
            _region(dst_buf, 2),
            tvm.tir.IntImm("int32", 0),
        ],
    )


def _unit_sync(mask):
    return tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.sunmmio_sync"),
        [tvm.tir.IntImm("int32", mask)],
    )


def _with_decl_buffers(stmt, buffers):
    for buf in reversed(buffers):
        stmt = tvm.tir.DeclBuffer(buf, stmt)
    return stmt


def _with_thread_extent(stmt):
    block = tvm.te.thread_axis("blockIdx.x")
    return tvm.tir.AttrStmt(
        block,
        "thread_extent",
        tvm.tir.IntImm("int32", 16),
        stmt,
    )


def _broadcast_stmt(*, direction=0, mask=None, src_core=None):
    src_data, dst_data, src_buf, dst_buf = _shared_buffers()
    body = tvm.tir.Evaluate(_broadcast(src_buf, dst_buf, direction=direction, mask=mask, src_core=src_core))
    body = _with_thread_extent(body)
    return _with_decl_buffers(body, [src_buf, dst_buf]), [src_data, dst_data]


def test_unit_sync_marker_codegen_validates_with_original_npuir(tmp_path):
    # TileLang's mask is independent from NPU-IR's enum values: bit 0 is
    # ODMA0 and bit 2 is TC.
    stmt = tvm.tir.Evaluate(
        tvm.tir.Call(
            "handle",
            tvm.ir.Op.get("tl.sunmmio_sync"),
            [tvm.tir.IntImm("int32", 0b101)],
        )
    )
    src = _validate_stmt_codegen(
        stmt,
        tmp_path,
        mlir_filename="unit_sync_suvm.mlir",
        expected_fragments=("suvm.sync",),
        opt_args=("--verify-each", "--suvm-to-llvm-pipeline"),
    )
    sync_line = next(line for line in src.splitlines() if "suvm.sync" in line)
    assert "odma0,tc" in sync_line
    assert "!suvm.token" not in src
    assert "suvm.wait_token" not in src


def test_broadcast_static_mask_codegen_validates_with_npuir_pipeline(tmp_path):
    stmt, params = _broadcast_stmt()
    src = _validate_stmt_codegen(
        stmt,
        tmp_path,
        params=params,
        mlir_filename="broadcast_static_mask_suvm.mlir",
        expected_fragments=("suvm.get_partitioned_tile_view", "suvm.mcast_tok", "suvm.sync"),
        opt_args=MCAST_VERIFY_ARGS,
    )
    assert src.count("suvm.mcast_tok") == 1
    barrier_indices = [i for i, line in enumerate(src.splitlines()) if "suvm.barrier.arrive_and_wait" in line]
    mcast_index = next(i for i, line in enumerate(src.splitlines()) if "suvm.mcast_tok" in line)
    sync_index = next(i for i, line in enumerate(src.splitlines()) if "suvm.sync" in line)
    assert len(barrier_indices) == 8
    assert max(barrier_indices[:4]) < mcast_index < sync_index < min(barrier_indices[4:])
    assert any("suvm.sync" in line and "hlink" in line for line in src.splitlines())


def test_independent_row_col_broadcasts_complete_separately(tmp_path):
    def make_rsram_buffer(name):
        elem_ty = tvm.ir.PrimType("bfloat16")
        data = tvm.tir.Var(
            f"{name}_data",
            tvm.ir.PointerType(elem_ty, "shared.rsram"),
        )
        buffer = tvm.tir.decl_buffer(
            (32, 32),
            "bfloat16",
            name=name,
            data=data,
            scope="shared.rsram",
        )
        return data, buffer

    src0_data, src0 = make_rsram_buffer("src0")
    dst0_data, dst0 = make_rsram_buffer("dst0")
    src1_data, src1 = make_rsram_buffer("src1")
    dst1_data, dst1 = make_rsram_buffer("dst1")
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(_broadcast(src0, dst0, direction=0)),
            tvm.tir.Evaluate(_broadcast(src1, dst1, direction=1)),
        ]
    )
    stmt = _with_decl_buffers(
        _with_thread_extent(body),
        [src0, dst0, src1, dst1],
    )
    src = _validate_stmt_codegen(
        stmt,
        tmp_path,
        params=[src0_data, dst0_data, src1_data, dst1_data],
        mlir_filename="independent_row_col_broadcast_suvm.mlir",
        expected_fragments=("suvm.mcast_tok", "suvm.sync"),
        opt_args=("--verify-each", "--suvm-to-llvm-pipeline"),
    )
    sync_lines = [line for line in src.splitlines() if "suvm.sync" in line]
    assert len(sync_lines) == 2
    assert "hlink" in sync_lines[0]
    assert "vlink" in sync_lines[1]
    assert src.count("suvm.mcast_tok") == 2
    assert src.count("suvm.barrier.arrive_and_wait") == 16


def test_broadcast_dynamic_mask_codegen_validates_with_npuir_pipeline(tmp_path):
    src_data, dst_data, src_buf, dst_buf = _shared_buffers()
    bx = tvm.tir.Var("bx", "int32")
    bx_i64 = tvm.tir.Cast("int64", bx)
    one = tvm.tir.IntImm("int64", 1)
    mask = tvm.tir.bitwise_or(
        tvm.tir.shift_left(one, bx_i64),
        tvm.tir.shift_left(one, bx_i64 + tvm.tir.IntImm("int64", 1)),
    )
    body = tvm.tir.Evaluate(_broadcast(src_buf, dst_buf, mask=mask))
    stmt = _with_decl_buffers(
        _with_thread_extent(tvm.tir.For(bx, 0, 3, tvm.tir.ForKind.SERIAL, body)),
        [src_buf, dst_buf],
    )
    _validate_stmt_codegen(
        stmt,
        tmp_path,
        params=[src_data, dst_data],
        mlir_filename="broadcast_dynamic_mask_suvm.mlir",
        expected_fragments=("scf.for", "arith.shli", "arith.ori", "suvm.mcast_tok", "suvm.sync"),
        opt_args=MCAST_VERIFY_ARGS,
    )


def test_broadcast_missing_dynamic_mask_fails_loudly():
    src_data, dst_data, src_buf, dst_buf = _shared_buffers()
    body = tvm.tir.Evaluate(_broadcast(src_buf, dst_buf, mask=tvm.tir.Var("missing_mask", "int64")))
    stmt = _with_decl_buffers(body, [src_buf, dst_buf])
    with pytest.raises(Exception, match="unbound TIR var.*missing_mask"):
        _build_sunmmio_source_from_stmt(stmt, params=[src_data, dst_data])


def test_broadcast_a4e_alignment_fails_during_codegen():
    shape = (16, 16)
    src_data, dst_data, src_buf, dst_buf = _shared_buffers(shape=shape)
    body = tvm.tir.Evaluate(_broadcast(src_buf, dst_buf, extents=shape))
    stmt = _with_decl_buffers(body, [src_buf, dst_buf])
    with pytest.raises(Exception, match="violates A4E multicast data-path constraints"):
        _build_sunmmio_source_from_stmt(stmt, params=[src_data, dst_data])


def test_broadcast_with_src_core_keeps_sync_in_guarded_branch(tmp_path):
    stmt, params = _broadcast_stmt(src_core=tvm.tir.IntImm("int32", 0))
    src = _validate_stmt_codegen(
        stmt,
        tmp_path,
        params=params,
        mlir_filename="broadcast_src_core_guard_suvm.mlir",
        expected_fragments=("suvm.get_core_id", "arith.cmpi eq", "scf.if", "suvm.mcast_tok", "suvm.sync"),
        opt_args=MCAST_VERIFY_ARGS,
    )
    lines = src.splitlines()
    barrier_indices = [i for i, line in enumerate(lines) if "suvm.barrier.arrive_and_wait" in line]
    if_index = next(i for i, line in enumerate(lines) if "scf.if" in line)
    mcast_index = next(i for i, line in enumerate(lines) if "suvm.mcast_tok" in line)
    sync_index = next(i for i, line in enumerate(lines) if "suvm.sync" in line)
    assert len(barrier_indices) == 2
    assert barrier_indices[0] < if_index < mcast_index < sync_index < barrier_indices[1]


def test_explicit_unit_sync_consumes_pending_codegen_state():
    src_data, dst_data, src_buf, dst_buf = _shared_buffers()
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(_dma(src_buf, dst_buf)),
            tvm.tir.Evaluate(_unit_sync(0b11)),
        ]
    )
    stmt = _with_decl_buffers(body, [src_buf, dst_buf])
    _, src = _build_sunmmio_source_from_stmt(
        stmt,
        params=[src_data, dst_data],
    )
    assert src.count("suvm.copy_async") == 1
    assert src.count("suvm.sync") == 1


def test_one_branch_sync_preserves_pending_state_on_other_branch():
    src_data, dst_data, src_buf, dst_buf = _shared_buffers()
    cond = tvm.tir.Var("cond", "bool")
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(_dma(src_buf, dst_buf)),
            tvm.tir.IfThenElse(
                cond,
                tvm.tir.Evaluate(_unit_sync(0b11)),
                tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0)),
            ),
        ]
    )
    stmt = _with_decl_buffers(body, [src_buf, dst_buf])
    _, src = _build_sunmmio_source_from_stmt(
        stmt,
        params=[src_data, dst_data, cond],
    )
    assert "scf.if" in src
    assert src.count("suvm.sync") == 2


def test_zero_trip_loop_preserves_entry_pending_state():
    src_data, dst_data, src_buf, dst_buf = _shared_buffers()
    i = tvm.tir.Var("i", "int32")
    extent = tvm.tir.Var("extent", "int32")
    body = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(_dma(src_buf, dst_buf)),
            tvm.tir.For(
                i,
                0,
                extent,
                tvm.tir.ForKind.SERIAL,
                tvm.tir.Evaluate(_unit_sync(0b11)),
            ),
        ]
    )
    stmt = _with_decl_buffers(body, [src_buf, dst_buf])
    _, src = _build_sunmmio_source_from_stmt(
        stmt,
        params=[src_data, dst_data, extent],
    )
    assert "scf.for" in src
    assert src.count("suvm.sync") == 2


def test_while_condition_pending_state_survives_initial_false_path():
    elem_ty = tvm.ir.PrimType("int32")
    state_data = tvm.tir.Var(
        "state_data",
        tvm.ir.PointerType(elem_ty, "shared.rsram"),
    )
    state = tvm.tir.decl_buffer(
        (32, 32),
        "int32",
        name="State",
        data=state_data,
        scope="shared.rsram",
    )
    zero = tvm.tir.IntImm("int32", 0)
    condition = tvm.tir.BufferLoad(state, [zero, zero]) > zero
    body = tvm.tir.Evaluate(_unit_sync(0b1100000))
    stmt = _with_decl_buffers(tvm.tir.While(condition, body), [state])

    _, src = _build_sunmmio_source_from_stmt(stmt, params=[state_data])
    lines = src.splitlines()
    sync_indices = [i for i, line in enumerate(lines) if "suvm.sync" in line]
    return_index = next(i for i, line in enumerate(lines) if line.strip() == "return")
    assert "scf.while" in src
    assert "suvm.tile.load" in src
    assert len(sync_indices) == 2
    assert sync_indices[0] < sync_indices[1] < return_index


def test_static_barrier_reuses_single_init(tmp_path):
    mask = tvm.tir.IntImm("int64", 15)
    barrier_init = tvm.tir.Call("handle", tvm.ir.Op.get("tl.barrier_init"), [mask])
    barrier_wait = tvm.tir.Call("handle", tvm.ir.Op.get("tl.barrier_arrive_and_wait"), [mask])
    stmt = tvm.tir.SeqStmt([tvm.tir.Evaluate(barrier_init), tvm.tir.Evaluate(barrier_wait), tvm.tir.Evaluate(barrier_wait)])
    src = _validate_stmt_codegen(
        stmt,
        tmp_path,
        mlir_filename="static_barrier_reuse_suvm.mlir",
        expected_fragments=("suvm.barrier.init", "suvm.barrier.arrive_and_wait"),
    )
    assert src.count("suvm.barrier.init") == 1
    assert src.count("suvm.barrier.arrive_and_wait") == 2


def test_barrier_state_does_not_leak_between_functions():
    mask = tvm.tir.IntImm("int64", 15)
    initializer = _primfunc_from_stmt(
        tvm.tir.Evaluate(tvm.tir.Call("handle", tvm.ir.Op.get("tl.barrier_init"), [mask])),
        name="a_initializer",
    )
    waiter = _primfunc_from_stmt(
        tvm.tir.Evaluate(tvm.tir.Call("handle", tvm.ir.Op.get("tl.barrier_arrive_and_wait"), [mask])),
        name="z_waiter",
    )
    with pytest.raises(Exception, match="has no corresponding tl.barrier_init"):
        _build_sunmmio_source_from_module(tvm.IRModule({"a_initializer": initializer, "z_waiter": waiter}))


def test_dynamic_barrier_candidates(tmp_path):
    bx = tvm.tir.Var("bx", "int32")
    bx_i64 = tvm.tir.Cast("int64", bx)
    mask = tvm.tir.shift_left(tvm.tir.IntImm("int64", 15), bx_i64 * tvm.tir.IntImm("int64", 4))
    candidates = [15, 240, 3840, 61440]
    barrier_init = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.barrier_init"),
        [tvm.tir.IntImm("int64", -1)] + [tvm.tir.IntImm("int64", value) for value in candidates],
    )
    barrier_wait = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.barrier_arrive_and_wait"),
        [mask] + [tvm.tir.IntImm("int64", value) for value in candidates],
    )
    stmt = tvm.tir.SeqStmt(
        [
            tvm.tir.Evaluate(barrier_init),
            tvm.tir.For(bx, 0, 4, tvm.tir.ForKind.SERIAL, tvm.tir.Evaluate(barrier_wait)),
        ]
    )
    src = _validate_stmt_codegen(
        stmt,
        tmp_path,
        mlir_filename="dynamic_barrier_candidates_suvm.mlir",
        expected_fragments=("scf.for", "scf.if", "arith.shli", "arith.cmpi eq"),
    )
    assert src.count("suvm.barrier.init") == len(candidates)
    assert src.count("suvm.barrier.arrive_and_wait") == len(candidates)


def test_nested_for_if_while_barrier_is_tokenless(tmp_path):
    i = tvm.tir.Var("i", "int32")
    cond = tvm.tir.LT(i, tvm.tir.IntImm("int32", 1))
    while_cond = tvm.tir.LT(tvm.tir.IntImm("int32", 0), tvm.tir.IntImm("int32", 1))
    mask = tvm.tir.IntImm("int64", 15)
    barrier_init = tvm.tir.Evaluate(tvm.tir.Call("handle", tvm.ir.Op.get("tl.barrier_init"), [mask]))
    barrier_wait = tvm.tir.Evaluate(tvm.tir.Call("handle", tvm.ir.Op.get("tl.barrier_arrive_and_wait"), [mask]))
    loop_body = tvm.tir.IfThenElse(
        cond,
        tvm.tir.While(while_cond, barrier_wait),
        tvm.tir.Evaluate(tvm.tir.IntImm("int32", 0)),
    )
    stmt = tvm.tir.SeqStmt([barrier_init, tvm.tir.For(i, 0, 1, tvm.tir.ForKind.SERIAL, loop_body)])
    _validate_stmt_codegen(
        stmt,
        tmp_path,
        mlir_filename="nested_for_if_while_barrier_suvm.mlir",
        expected_fragments=("scf.for", "scf.if", "scf.while", "suvm.barrier.arrive_and_wait"),
    )


if __name__ == "__main__":
    tilelang.testing.main()
