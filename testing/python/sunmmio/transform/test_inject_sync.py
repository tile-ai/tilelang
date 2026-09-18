import tilelang
from tilelang import tvm
from tilelang.utils.target import determine_target


def _target():
    target = determine_target("Sunmmio", return_object=True)
    target_host = "llvm" if tvm.runtime.enabled("llvm") else "c"
    return tvm.target.Target(target, tvm.target.Target.canon_target(target_host))


def _pointer_var(name, dtype="bfloat16", scope="shared.rsram"):
    return tvm.tir.Var(
        name,
        tvm.ir.PointerType(tvm.ir.PrimType(dtype), scope),
    )


def _region(buffer, access):
    return _region_slice(buffer, access, 0, 32)


def _region_slice(buffer, access, row, rows):
    return tvm.tir.call_intrin(
        "handle",
        tvm.ir.Op.get("tl.tileop.region"),
        tvm.tir.BufferLoad(
            buffer,
            [tvm.tir.IntImm("int32", row), tvm.tir.IntImm("int32", 0)],
        ),
        tvm.tir.IntImm("int32", access),
        tvm.tir.IntImm("int32", rows),
        tvm.tir.IntImm("int32", 32),
    )


def _region_at(buffer, access, row, col, rows=32, cols=32):
    return tvm.tir.call_intrin(
        "handle",
        tvm.ir.Op.get("tl.tileop.region"),
        tvm.tir.BufferLoad(buffer, [row, col]),
        tvm.tir.IntImm("int32", access),
        tvm.tir.IntImm("int32", rows),
        tvm.tir.IntImm("int32", cols),
    )


def _broadcast(src, dst, *, direction=0, mask=15, src_core=None):
    return _broadcast_regions(
        _region(src, 1),
        _region(dst, 2),
        direction=direction,
        mask=mask,
        src_core=src_core,
    )


def _broadcast_regions(src_region, dst_region, *, direction=0, mask=15, src_core=None):
    args = [
        src_region,
        dst_region,
        tvm.tir.IntImm("int32", direction),
        mask if isinstance(mask, tvm.tir.PrimExpr) else tvm.tir.IntImm("int64", mask),
        tvm.tir.IntImm("int32", 0),
    ]
    if src_core is not None:
        args.append(src_core if isinstance(src_core, tvm.tir.PrimExpr) else tvm.tir.IntImm("int32", src_core))
    return tvm.tir.Evaluate(tvm.tir.Call("handle", tvm.ir.Op.get("tl.broadcast_"), args))


def _dma(src, dst, *, unit="odma0"):
    return _dma_regions(_region(src, 1), _region(dst, 2), unit=unit)


def _dma_regions(src_region, dst_region, *, unit="odma0"):
    unit_marker = tvm.tir.Call(
        "handle",
        tvm.ir.Op.get("tl.odma_unit"),
        [tvm.tir.StringImm(unit)],
    )
    return tvm.tir.Evaluate(
        tvm.tir.Call(
            "handle",
            tvm.ir.Op.get("tl.dma_copy"),
            [
                src_region,
                dst_region,
                tvm.tir.IntImm("int32", 0),
                unit_marker,
            ],
        )
    )


def _mx_compound(op_name, src, dst):
    if op_name == "tl.mx_pack":
        regions = [_region(src, 1), _region(src, 1), _region(dst, 2)]
    else:
        assert op_name == "tl.mx_unpack"
        regions = [_region(src, 1), _region(dst, 2), _region(dst, 2)]
    return tvm.tir.Evaluate(tvm.tir.Call("handle", tvm.ir.Op.get(op_name), regions))


def _make_module(statements, *, with_thread_extent=True, is_global=True):
    src_data = _pointer_var("src")
    dst_data = _pointer_var("dst")
    src = tvm.tir.decl_buffer((32, 32), "bfloat16", name="src", data=src_data, scope="shared.rsram")
    dst = tvm.tir.decl_buffer((32, 32), "bfloat16", name="dst", data=dst_data, scope="shared.rsram")
    body_stmts = statements(src, dst)
    if not isinstance(body_stmts, list):
        body_stmts = [body_stmts]
    body = body_stmts[0] if len(body_stmts) == 1 else tvm.tir.SeqStmt(body_stmts)
    body = tvm.tir.DeclBuffer(src, tvm.tir.DeclBuffer(dst, body))
    if with_thread_extent:
        block = tvm.te.thread_axis("blockIdx.x")
        body = tvm.tir.AttrStmt(block, "thread_extent", tvm.tir.IntImm("int32", 16), body)
    func = tvm.tir.PrimFunc([src_data, dst_data], body).with_attr("global_symbol", "main")
    if is_global:
        func = func.with_attr("tir.is_global_func", True)
    mod = tvm.IRModule({"main": func})
    return tvm.tir.transform.BindTarget(_target())(mod)


def _apply(mod):
    return tilelang.transform.InjectSunmmioSync()(mod)


def _call_lines(mod, name):
    return [line.strip() for line in mod.script().splitlines() if name in line]


def test_non_collective_ops_remain_tokenless():
    def body(src, dst):
        return tvm.tir.Evaluate(
            tvm.tir.Call(
                "handle",
                tvm.ir.Op.get("tl.dma_copy"),
                [_region(src, 1), _region(dst, 2), tvm.tir.IntImm("int32", 0)],
            )
        )

    script = _apply(_make_module(body)).script()
    assert "T.dma_copy" in script
    assert "sync_token_id" not in script
    assert "sync_null_token" not in script
    assert "wait_token" not in script


def test_raw_dependency_inserts_producer_unit_sync():
    mod = _apply(
        _make_module(
            lambda src, dst: [
                _dma(src, dst, unit="odma0"),
                _dma(dst, src, unit="odma1"),
            ]
        )
    )
    script = mod.script()
    sync_lines = _call_lines(mod, "T.sunmmio_sync")
    assert sync_lines == ["T.sunmmio_sync(1)"]
    first_dma = script.index("T.dma_copy")
    sync = script.index("T.sunmmio_sync")
    second_dma = script.index("T.dma_copy", first_dma + 1)
    assert first_dma < sync < second_dma
    assert "token" not in script


def test_repeated_dma_write_inserts_waw_sync():
    def body(src, dst):
        src_first = tvm.tir.decl_buffer(
            (32, 32),
            "bfloat16",
            name="src_first",
            data=src.data,
            scope="shared.rsram",
            elem_offset=0,
        )
        dst_first = tvm.tir.decl_buffer(
            (32, 32),
            "bfloat16",
            name="dst_first",
            data=dst.data,
            scope="shared.rsram",
            elem_offset=0,
        )
        return [_dma(src_first, dst_first), _dma(src_first, dst_first)]

    # Two writes to the same region are a WAW dependency even when the source
    # reads are identical.
    script = _apply(_make_module(body)).script()
    assert "T.sunmmio_sync(1)" in script


def test_distinct_buffer_views_of_same_storage_insert_sync():
    def body(src, dst):
        src_alias = tvm.tir.decl_buffer(
            (32, 32),
            "bfloat16",
            name="src_alias",
            data=src.data,
            scope="shared.rsram",
        )
        dst_alias = tvm.tir.decl_buffer(
            (32, 32),
            "bfloat16",
            name="dst_alias",
            data=dst.data,
            scope="shared.rsram",
        )
        return [
            _dma(src, dst, unit="odma0"),
            _dma(dst_alias, src_alias, unit="odma1"),
        ]

    mod = _apply(_make_module(body))
    assert _call_lines(mod, "T.sunmmio_sync") == ["T.sunmmio_sync(1)"]


def test_equivalent_alias_views_preserve_disjoint_region_overlap():
    def body(src, dst):
        src_alias = tvm.tir.decl_buffer(
            (32, 32),
            "bfloat16",
            name="src_alias",
            data=src.data,
            scope="shared.rsram",
        )
        dst_alias = tvm.tir.decl_buffer(
            (32, 32),
            "bfloat16",
            name="dst_alias",
            data=dst.data,
            scope="shared.rsram",
        )
        return [
            _dma_regions(
                _region_slice(src, 1, 0, 16),
                _region_slice(dst, 2, 0, 16),
                unit="odma0",
            ),
            _dma_regions(
                _region_slice(src_alias, 1, 16, 16),
                _region_slice(dst_alias, 2, 16, 16),
                unit="odma1",
            ),
        ]

    assert "T.sunmmio_sync" not in _apply(_make_module(body)).script()


def test_hidden_async_region_index_inserts_producer_sync():
    zero = tvm.tir.IntImm("int32", 0)
    idx_src_data = _pointer_var("idx_src", dtype="int32", scope="global")
    idx_data = _pointer_var("idx", dtype="int32")
    data_data = _pointer_var("data")
    out_data = _pointer_var("out", scope="global")
    idx_src = tvm.tir.decl_buffer((32, 32), "int32", name="idx_src", data=idx_src_data, scope="global")
    idx = tvm.tir.decl_buffer((32, 32), "int32", name="idx", data=idx_data, scope="shared.rsram")
    data = tvm.tir.decl_buffer((32, 32), "bfloat16", name="data", data=data_data, scope="shared.rsram")
    out = tvm.tir.decl_buffer((32, 32), "bfloat16", name="out", data=out_data, scope="global")
    dynamic_row = tvm.tir.BufferLoad(idx, [zero, zero])
    body = tvm.tir.SeqStmt(
        [
            _dma_regions(_region(idx_src, 1), _region(idx, 2), unit="odma0"),
            _dma_regions(
                _region_at(data, 1, dynamic_row, zero, 1, 32),
                _region_at(out, 2, zero, zero, 1, 32),
                unit="odma0",
            ),
        ]
    )
    for buffer in reversed([idx_src, idx, data, out]):
        body = tvm.tir.DeclBuffer(buffer, body)
    func = (
        tvm.tir.PrimFunc([idx_src_data, idx_data, data_data, out_data], body)
        .with_attr("global_symbol", "main")
        .with_attr("tir.is_global_func", True)
    )
    mod = tvm.tir.transform.BindTarget(_target())(tvm.IRModule({"main": func}))

    lines = _apply(mod).script().splitlines()
    dma_indices = [i for i, line in enumerate(lines) if "T.dma_copy" in line]
    sync_index = next(i for i, line in enumerate(lines) if "T.sunmmio_sync(1)" in line)
    assert dma_indices[0] < sync_index < dma_indices[1]


def test_condition_load_is_synchronized_before_branch_async_write():
    zero = tvm.tir.IntImm("int32", 0)

    def body(src, dst):
        condition = tvm.tir.BufferLoad(dst, [zero, zero]) > tvm.tir.FloatImm("bfloat16", 0.0)
        return tvm.tir.IfThenElse(
            condition,
            _dma(src, dst, unit="odma0"),
            None,
        )

    lines = _apply(_make_module(body)).script().splitlines()
    if_index = next(i for i, line in enumerate(lines) if line.strip().startswith("if "))
    sync_index = next(i for i, line in enumerate(lines) if "T.sunmmio_sync(96)" in line)
    dma_index = next(i for i, line in enumerate(lines) if "T.dma_copy" in line)
    assert if_index < sync_index < dma_index


def test_disjoint_dma_regions_do_not_insert_sync():
    def body(src, dst):
        return [
            _dma_regions(
                _region_slice(src, 1, 0, 16),
                _region_slice(dst, 2, 0, 16),
                unit="odma0",
            ),
            _dma_regions(
                _region_slice(src, 1, 16, 16),
                _region_slice(dst, 2, 16, 16),
                unit="odma1",
            ),
        ]

    assert "T.sunmmio_sync" not in _apply(_make_module(body)).script()


def test_unit_sync_clears_all_pending_accesses_on_that_unit():
    def body(src, dst):
        return [
            _dma_regions(
                _region_slice(src, 1, 0, 16),
                _region_slice(dst, 2, 0, 16),
                unit="odma0",
            ),
            _dma_regions(
                _region_slice(src, 1, 16, 16),
                _region_slice(dst, 2, 16, 16),
                unit="odma0",
            ),
            _dma_regions(
                _region_slice(dst, 1, 0, 16),
                _region_slice(src, 2, 0, 16),
                unit="odma1",
            ),
            _dma_regions(
                _region_slice(dst, 1, 16, 16),
                _region_slice(src, 2, 16, 16),
                unit="odma1",
            ),
        ]

    sync_lines = _call_lines(_apply(_make_module(body)), "T.sunmmio_sync")
    assert sync_lines == ["T.sunmmio_sync(1)"]


def test_mx_compound_write_is_synchronized_before_async_consumer():
    for op_name in ("tl.mx_pack", "tl.mx_unpack"):
        input_data = _pointer_var("input")
        mx_data = _pointer_var("mx")
        sink_data = _pointer_var("sink", scope="shared.asram")
        input_buffer = tvm.tir.decl_buffer((32, 32), "bfloat16", name="input", data=input_data, scope="shared.rsram")
        mx = tvm.tir.decl_buffer((32, 32), "bfloat16", name="mx", data=mx_data, scope="shared.rsram")
        sink = tvm.tir.decl_buffer((32, 32), "bfloat16", name="sink", data=sink_data, scope="shared.asram")
        body = tvm.tir.SeqStmt(
            [
                _mx_compound(op_name, input_buffer, mx),
                _dma_regions(_region(mx, 1), _region(sink, 2), unit="odma1"),
            ]
        )
        for buffer in reversed([input_buffer, mx, sink]):
            body = tvm.tir.DeclBuffer(buffer, body)
        func = (
            tvm.tir.PrimFunc([input_data, mx_data, sink_data], body)
            .with_attr("global_symbol", "main")
            .with_attr("tir.is_global_func", True)
        )
        input_mod = tvm.tir.transform.BindTarget(_target())(tvm.IRModule({"main": func}))

        result = _apply(input_mod)
        script = result.script()
        assert _call_lines(result, "T.sunmmio_sync") == ["T.sunmmio_sync(96)"]
        assert script.index(f"T.{op_name.removeprefix('tl.')}") < script.index("T.sunmmio_sync") < script.index("T.dma_copy")


def test_predicated_store_synchronizes_condition_load():
    input_data = _pointer_var("input", dtype="int32", scope="global")
    mask_data = _pointer_var("mask", dtype="int32")
    output_data = _pointer_var("output", dtype="int32")
    input_buffer = tvm.tir.decl_buffer((32, 32), "int32", name="input", data=input_data, scope="global")
    mask = tvm.tir.decl_buffer((32, 32), "int32", name="mask", data=mask_data, scope="shared.rsram")
    output = tvm.tir.decl_buffer((32, 32), "int32", name="output", data=output_data, scope="shared.rsram")
    zero = tvm.tir.IntImm("int32", 0)
    body = tvm.tir.SeqStmt(
        [
            _dma_regions(_region(input_buffer, 1), _region(mask, 2)),
            tvm.tir.BufferStore(
                output,
                tvm.tir.IntImm("int32", 1),
                [zero, zero],
                predicate=tvm.tir.BufferLoad(mask, [zero, zero]) != zero,
            ),
        ]
    )
    for buffer in reversed([input_buffer, mask, output]):
        body = tvm.tir.DeclBuffer(buffer, body)
    func = (
        tvm.tir.PrimFunc([input_data, mask_data, output_data], body)
        .with_attr("global_symbol", "main")
        .with_attr("tir.is_global_func", True)
    )
    mod = tvm.tir.transform.BindTarget(_target())(tvm.IRModule({"main": func}))

    result = _apply(mod)
    script = result.script()
    assert _call_lines(result, "T.sunmmio_sync") == ["T.sunmmio_sync(1)"]
    assert script.index("T.dma_copy") < script.index("T.sunmmio_sync") < script.index(".vstore(")
    assert "predicate=mask_1[0, 0] != 0" in script


def test_loop_drains_pending_units_at_iteration_boundary():
    def body(src, dst):
        i = tvm.tir.Var("i", "int32")
        return tvm.tir.For(i, 0, 2, tvm.tir.ForKind.SERIAL, _dma(src, dst))

    script = _apply(_make_module(body)).script()
    assert script.index("T.dma_copy") < script.index("T.sunmmio_sync(1)")
    assert "sync_token_id" not in script
    assert "wait_token" not in script


def test_loop_preserves_explicit_step():
    i = tvm.tir.Var("i", "int32")
    mod = _apply(
        _make_module(
            lambda src, dst: tvm.tir.For(
                i,
                0,
                8,
                tvm.tir.ForKind.SERIAL,
                tvm.tir.Evaluate(i),
                step=tvm.tir.IntImm("int32", 2),
            )
        )
    )
    steps = []

    def collect_steps(node):
        if isinstance(node, tvm.tir.For) and node.loop_var.name == "i":
            steps.append(node.step)

    tvm.tir.stmt_functor.post_order_visit(mod["main"].body, collect_steps)
    assert len(steps) == 1
    assert isinstance(steps[0], tvm.tir.IntImm)
    assert int(steps[0]) == 2


def test_source_guarded_broadcast_has_pre_and_post_barriers():
    mod = _apply(_make_module(lambda src, dst: _broadcast(src, dst, src_core=0)))
    lines = mod.script().splitlines()
    assert len(_call_lines(mod, "T.barrier_init")) == 1
    barrier_indices = [i for i, line in enumerate(lines) if "T.barrier_arrive_and_wait" in line]
    broadcast_index = next(i for i, line in enumerate(lines) if "T.broadcast_" in line)
    assert len(barrier_indices) == 2
    assert barrier_indices[0] < broadcast_index < barrier_indices[1]
    assert "T.sunmmio_sync" not in "\n".join(lines)
    assert "token" not in "\n".join(lines)


def test_unguarded_broadcast_completes_link_before_post_barrier():
    mod = _apply(_make_module(lambda src, dst: _broadcast(src, dst)))
    lines = mod.script().splitlines()
    assert _call_lines(mod, "T.sunmmio_sync") == ["T.sunmmio_sync(8)"]
    barrier_indices = [i for i, line in enumerate(lines) if "T.barrier_arrive_and_wait" in line]
    broadcast_index = next(i for i, line in enumerate(lines) if "T.broadcast_" in line)
    sync_index = next(i for i, line in enumerate(lines) if "T.sunmmio_sync(8)" in line)
    assert barrier_indices[0] < broadcast_index < sync_index < barrier_indices[1]


def test_repeated_unguarded_broadcasts_complete_each_collective():
    mod = _apply(
        _make_module(
            lambda src, dst: [
                _broadcast_regions(
                    _region_slice(src, 1, 0, 16),
                    _region_slice(dst, 2, 0, 16),
                    direction=0,
                ),
                _broadcast_regions(
                    _region_slice(src, 1, 16, 16),
                    _region_slice(dst, 2, 16, 16),
                    direction=0,
                ),
            ]
        )
    )
    lines = mod.script().splitlines()
    assert _call_lines(mod, "T.sunmmio_sync") == ["T.sunmmio_sync(8)", "T.sunmmio_sync(8)"]
    broadcast_indices = [i for i, line in enumerate(lines) if "T.broadcast_" in line]
    sync_indices = [i for i, line in enumerate(lines) if "T.sunmmio_sync(8)" in line]
    barrier_indices = [i for i, line in enumerate(lines) if "T.barrier_arrive_and_wait" in line]
    assert len(barrier_indices) == 4
    assert barrier_indices[0] < broadcast_indices[0] < sync_indices[0] < barrier_indices[1]
    assert barrier_indices[2] < broadcast_indices[1] < sync_indices[1] < barrier_indices[3]


def test_independent_links_complete_before_their_post_barriers():
    mod = _apply(
        _make_module(
            lambda src, dst: [
                _broadcast_regions(
                    _region_slice(src, 1, 0, 16),
                    _region_slice(dst, 2, 0, 16),
                    direction=0,
                ),
                _broadcast_regions(
                    _region_slice(src, 1, 16, 16),
                    _region_slice(dst, 2, 16, 16),
                    direction=1,
                ),
            ]
        )
    )
    lines = mod.script().splitlines()
    assert _call_lines(mod, "T.sunmmio_sync") == ["T.sunmmio_sync(8)", "T.sunmmio_sync(16)"]
    broadcast_indices = [i for i, line in enumerate(lines) if "T.broadcast_" in line]
    sync_indices = [i for i, line in enumerate(lines) if "T.sunmmio_sync" in line]
    barrier_indices = [i for i, line in enumerate(lines) if "T.barrier_arrive_and_wait" in line]
    assert barrier_indices[0] < broadcast_indices[0] < sync_indices[0] < barrier_indices[1]
    assert barrier_indices[2] < broadcast_indices[1] < sync_indices[1] < barrier_indices[3]


def test_broadcast_link_sync_is_placed_before_dependent_consumer():
    mod = _apply(
        _make_module(
            lambda src, dst: [
                _broadcast(src, dst, direction=0),
                _dma(dst, src, unit="odma1"),
            ]
        )
    )
    lines = mod.script().splitlines()
    assert _call_lines(mod, "T.sunmmio_sync") == ["T.sunmmio_sync(8)"]
    broadcast_index = next(i for i, line in enumerate(lines) if "T.broadcast_" in line)
    sync_index = next(i for i, line in enumerate(lines) if "T.sunmmio_sync(8)" in line)
    post_barrier_index = max(i for i, line in enumerate(lines) if "T.barrier_arrive_and_wait" in line)
    dma_index = next(i for i, line in enumerate(lines) if "T.dma_copy" in line)
    assert broadcast_index < sync_index < post_barrier_index < dma_index


def test_repeated_broadcasts_reuse_barrier_and_wait_each_launch():
    mod = _apply(
        _make_module(
            lambda src, dst: [
                _broadcast(src, dst, src_core=0),
                _broadcast(dst, src, src_core=0),
            ]
        )
    )
    script = mod.script()
    assert len(_call_lines(mod, "T.barrier_init")) == 1
    assert len(_call_lines(mod, "T.barrier_arrive_and_wait")) == 4
    assert script.count("T.broadcast_") == 2


def test_broadcast_without_src_core_uses_current_core_candidates():
    mod = _apply(_make_module(lambda src, dst: _broadcast(src, dst)))
    barrier_init = _call_lines(mod, "T.barrier_init")
    barrier_wait = _call_lines(mod, "T.barrier_arrive_and_wait")
    assert len(barrier_init) == 1
    assert len(barrier_wait) == 2
    for mask in (15, 240, 3840, 61440):
        assert str(mask) in barrier_init[0]
        assert str(mask) in barrier_wait[0]


def test_column_broadcast_expands_vertical_participants():
    mod = _apply(_make_module(lambda src, dst: _broadcast(src, dst, direction=1, src_core=0)))
    assert "4369" in _call_lines(mod, "T.barrier_init")[0]
    barrier_wait = _call_lines(mod, "T.barrier_arrive_and_wait")
    assert len(barrier_wait) == 2
    assert all("4369" in line for line in barrier_wait)


def test_non_global_function_is_unchanged():
    mod = _make_module(lambda src, dst: _broadcast(src, dst, src_core=0), is_global=False)
    assert tvm.ir.structural_equal(mod, _apply(mod))


if __name__ == "__main__":
    tilelang.testing.main()
