import re
from tvm import tir, IRModule, arith, ir


def verify_comm_lower(func: tir.PrimFunc):
    expected_broadcasts = []
    analyzer = arith.Analyzer()
    current_mesh_nrow = 4
    current_mesh_ncol = 4

    def split_top_level_args(arg_str):
        args = []
        start = 0
        depth = 0
        in_string = False
        escape = False
        for idx, ch in enumerate(arg_str):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 0:
                args.append(arg_str[start:idx].strip())
                start = idx + 1
        tail = arg_str[start:].strip()
        if tail:
            args.append(tail)
        return args

    def broadcast_call_args(line):
        prefix = "T.broadcast_("
        start = line.find(prefix)
        if start < 0:
            return None
        arg_start = start + len(prefix)
        depth = 1
        in_string = False
        escape = False
        for idx in range(arg_start, len(line)):
            ch = line[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return split_top_level_args(line[arg_start:idx])
        return None

    def full_local_mask(axis_len):
        return (1 << axis_len) - 1

    def horizontal_mask(src_core):
        return full_local_mask(current_mesh_ncol)

    def vertical_mask(src_core):
        return full_local_mask(current_mesh_nrow)

    def direction_from_arg(direction_val):
        if isinstance(direction_val, tir.StringImm):
            return 0 if direction_val.value == "h" else 1 if direction_val.value == "v" else 2
        return int(direction_val) if isinstance(direction_val, tir.IntImm) else 0

    def add_expected(src_core, direction, mask=None, has_src_core=True):
        expected_broadcasts.append((stringify_expr(src_core) if src_core is not None else None, direction, mask, has_src_core))

    def get_region_size(node):
        if isinstance(node, tir.Call) and node.op.name == "tl.tileop.region":
            size = 1
            for i in range(2, len(node.args)):
                size *= node.args[i]
            return analyzer.simplify(size)
        elif isinstance(node, tir.BufferRegion):
            size = 1
            for r in node.region:
                size *= r.extent
            return analyzer.simplify(size)
        elif isinstance(node, tir.Buffer):
            size = 1
            for s in node.shape:
                size *= s
            return analyzer.simplify(size)
        elif isinstance(node, tir.ProducerLoad):
            size = 1
            for s in node.buffer.shape:
                size *= s
            return analyzer.simplify(size)
        elif isinstance(node, tir.BufferLoad):
            # If it's a BufferLoad, we try to see if it looks like a slice.
            # In TileLang's debug representation, slices look like A[start:end, ...]
            # But in the actual object, it's a BufferLoad with indices.
            # If we can't determine the slice size, we return the buffer size
            # BUT we mark it as "possibly a full buffer" so the caller can prefer
            # a more specific size if available.
            # For now, let's just return the buffer size but try to handle the fallback better.
            size = 1
            for s in node.buffer.shape:
                size *= s
            return analyzer.simplify(size)
        elif hasattr(node, "buffer"):
            return get_region_size(node.buffer)
        return tir.IntImm("int32", 1)

    def stringify_expr(expr):
        if isinstance(expr, tir.IntImm):
            return str(int(expr))
        return str(analyzer.simplify(expr))

    def visitor(node):
        nonlocal current_mesh_nrow, current_mesh_ncol
        if isinstance(node, tir.Call) and node.op.name.startswith("tl.tileop.comm_"):
            if node.op.name == "tl.tileop.comm_broadcast":
                # args: src, dst, size, src_core, direction
                size_expr = node.args[2]
                if isinstance(size_expr, tir.IntImm) and int(size_expr) > 0:
                    size = analyzer.simplify(size_expr)
                else:
                    # Infer size from src region and dst buffer
                    size0 = get_region_size(node.args[0])
                    size1 = get_region_size(node.args[1])
                    if isinstance(size0, tir.IntImm) and isinstance(size1, tir.IntImm):
                        if int(size0) > 0 and int(size1) > 0:
                            size = tir.IntImm("int32", min(int(size0), int(size1)))
                        else:
                            size = analyzer.simplify(size0 if int(size0) > 0 else size1)
                    else:
                        size = analyzer.simplify(size0)

                src_core = node.args[3]
                direction = direction_from_arg(node.args[4])

                if direction == 0 or direction == 1:
                    mask = horizontal_mask(src_core) if direction == 0 else vertical_mask(src_core)
                    add_expected(src_core, direction, mask)
                elif direction == 2 and isinstance(src_core, tir.IntImm):
                    # 2D broadcast: only supports constant src_core in C++
                    src_core_val = int(src_core)
                    src_core_col = src_core_val % current_mesh_ncol
                    add_expected(src_core, 1, vertical_mask(src_core))
                    for i in range(current_mesh_nrow):
                        row_src_core = tir.IntImm("int32", i * current_mesh_ncol + src_core_col)
                        add_expected(row_src_core, 0, horizontal_mask(row_src_core))

            elif node.op.name == "tl.tileop.comm_put":
                # args: src, dst, size, src_core, dst_core
                size_expr = node.args[2]
                if isinstance(size_expr, tir.IntImm) and int(size_expr) > 0:
                    size = analyzer.simplify(size_expr)
                else:
                    # Infer size from src region, fallback to dst buffer if src is just a pointer/point
                    size = get_region_size(node.args[0])
                    if isinstance(size, tir.IntImm) and int(size) == 1:
                        size = get_region_size(node.args[1])

                src_core = node.args[3]
                dst_core = node.args[4]

                # Put logic in C++ requires constant core IDs
                if isinstance(src_core, tir.IntImm) and isinstance(dst_core, tir.IntImm):
                    src_core_val = int(src_core)
                    dst_core_val = int(dst_core)
                    src_row, src_col = (
                        src_core_val // current_mesh_ncol,
                        src_core_val % current_mesh_ncol,
                    )
                    dst_row, dst_col = (
                        dst_core_val // current_mesh_ncol,
                        dst_core_val % current_mesh_ncol,
                    )

                    if src_row == dst_row:
                        add_expected(src_core, 0, 1 << dst_col)
                    elif src_col == dst_col:
                        add_expected(src_core, 1, 1 << dst_row)
                    else:
                        intermediate_core = dst_row * current_mesh_ncol + src_col
                        add_expected(src_core, 1, 1 << dst_row)
                        add_expected(tir.IntImm("int32", intermediate_core), 0, 1 << dst_col)

            elif node.op.name == "tl.tileop.comm_allgather":
                # args: send, recv, direction, size
                direction = direction_from_arg(node.args[2])

                size_expr = node.args[3]
                if isinstance(size_expr, tir.IntImm) and int(size_expr) > 0:
                    size = analyzer.simplify(size_expr)
                else:
                    # Infer size from send region and recv buffer
                    size0 = get_region_size(node.args[0])
                    size1 = get_region_size(node.args[1])
                    if isinstance(size0, tir.IntImm) and isinstance(size1, tir.IntImm):
                        if int(size0) > 0 and int(size1) > 0:
                            size = tir.IntImm("int32", min(int(size0), int(size1)))
                        else:
                            size = analyzer.simplify(size0 if int(size0) > 0 else size1)
                    else:
                        size = analyzer.simplify(size0)

                if direction == 0:  # horizontal
                    add_expected(None, 0, None, has_src_core=False)
                elif direction == 1:  # vertical
                    add_expected(None, 1, None, has_src_core=False)
                elif direction == 2:  # all
                    add_expected(None, 1, None, has_src_core=False)
                    add_expected(None, 0, None, has_src_core=False)

    if not isinstance(func, tir.PrimFunc):
        raise ValueError(f"Expected PrimFunc, got {type(func)}")

    # 1. Gather expectations from func
    if func.attrs is not None:
        if "mesh_nrow" in func.attrs:
            current_mesh_nrow = int(func.attrs["mesh_nrow"])
        if "mesh_ncol" in func.attrs:
            current_mesh_ncol = int(func.attrs["mesh_ncol"])
    tir.stmt_functor.post_order_visit(func.body, visitor)

    # 2. Return the check function
    def check(mod: IRModule):
        script = mod.script()
        broadcast_arg_lists = [args for line in script.splitlines() if (args := broadcast_call_args(line.strip())) is not None]

        def strip_broadcast_metadata(args):
            fixed_args = list(args)
            if fixed_args and fixed_args[-1].startswith("T.odma_unit("):
                fixed_args.pop()
            return fixed_args

        for core, direction, mask, has_src_core in expected_broadcasts:
            if mask is None:
                mask_pattern = r".*?"
            else:
                mask_pattern = rf"(?:T\.int64\({mask}\)|{mask})"
            if has_src_core:
                if not core.lstrip("-").isdigit():
                    # Lowering may inline let-bound dynamic coordinates and
                    # rename the kernel thread variable. Verify the lowered
                    # broadcast ABI here; per-test checks own its exact route.
                    message = f"Expected broadcast_ with a dynamic source core, direction={direction}, mask={mask} not found in IRModule"
                    found = False
                    for args in broadcast_arg_lists:
                        fixed_args = strip_broadcast_metadata(args)
                        if len(fixed_args) != 6:
                            continue
                        if fixed_args[2] != str(direction):
                            continue
                        if mask is not None and fixed_args[3] not in {f"T.int64({mask})", str(mask)}:
                            continue
                        found = True
                        break
                    assert found, message
                    continue

                # Match T.broadcast_(..., direction, mask, src_offset_byte, src_core, ...)
                escaped_core = re.escape(core).replace(r"\ ", r"\s*")
                pattern = rf"T\.broadcast_\(.*?,\s*.*?,\s*{direction},\s*{mask_pattern},\s*.*?,\s*{escaped_core}(?:,|\))"
                message = f"Expected broadcast_ with core={core}, direction={direction}, mask={mask} not found in IRModule"
            else:
                # Match the dynamic allgather form without src_core:
                # T.broadcast_(src_region, dst_region, direction, mask,
                #              src_offset_byte[, odma_unit])
                message = f"Expected broadcast_ without src_core, direction={direction}, mask={mask} not found in IRModule"
                found = False
                for args in broadcast_arg_lists:
                    fixed_args = strip_broadcast_metadata(args)
                    if len(fixed_args) != 5:
                        continue
                    if fixed_args[2] != str(direction):
                        continue
                    if mask is not None and fixed_args[3] not in {f"T.int64({mask})", str(mask)}:
                        continue
                    found = True
                    break
                assert found, message
                continue
            assert re.search(pattern, script), message

    return check


def verify_SunmmioSync(mod: IRModule):
    script = mod.script()
    lines = [l.strip() for l in script.split("\n")]
    assert "sync_token_id" not in script
    assert "sync_null_token" not in script
    assert "wait_token" not in script

    barrier_init_masks = [l.split("barrier_init(", 1)[1].rsplit(")", 1)[0] for l in lines if "barrier_init(" in l]
    initialized_masks = set(barrier_init_masks)
    barrier_init_calls = []
    barrier_arrive_calls = []
    unit_sync_calls = []

    def collect_sync_calls(node):
        if not isinstance(node, tir.Call) or not hasattr(node.op, "name"):
            return
        if node.op.name == "tl.barrier_init":
            barrier_init_calls.append(node)
        elif node.op.name == "tl.barrier_arrive_and_wait":
            barrier_arrive_calls.append(node)
        elif node.op.name == "tl.sunmmio_sync":
            unit_sync_calls.append(node)

    for func in mod.functions.values():
        if isinstance(func, tir.PrimFunc):
            tir.stmt_functor.post_order_visit(func.body, collect_sync_calls)

    def structurally_equal_args(lhs, rhs):
        return len(lhs) == len(rhs) and all(ir.structural_equal(a, b) for a, b in zip(lhs, rhs))

    def has_matching_barrier_init(arrive):
        if len(arrive.args) == 1:
            return any(structurally_equal_args(init.args, arrive.args) for init in barrier_init_calls)
        # Dynamic barriers use init(-1, candidates...) and
        # arrive_and_wait(runtime_expr, candidates...).
        return any(len(init.args) > 1 and structurally_equal_args(init.args[1:], arrive.args[1:]) for init in barrier_init_calls)

    for arrive in barrier_arrive_calls:
        assert has_matching_barrier_init(arrive), f"{arrive} does not have a matching barrier_init"
    known_sync_units = (1 << 7) - 1
    for sync in unit_sync_calls:
        assert len(sync.args) == 1 and isinstance(sync.args[0], tir.IntImm), f"invalid unit sync marker close to {sync}"
        mask = int(sync.args[0])
        assert mask > 0, f"unit sync mask must be nonzero, got {mask}"
        assert mask & ~known_sync_units == 0, f"unit sync mask contains unknown bits: {mask}"
    # Check order of barrier_init(mask) and barrier_arrive_and_wait(mask).
    for mask in initialized_masks:
        idx_barrier = script.find(f"barrier_init({mask})")
        idx_arrive = script.find(f"barrier_arrive_and_wait({mask})")
        if idx_arrive == -1:
            continue
        assert idx_barrier != -1, f"barrier_init({mask}) is not found in script"
        assert idx_barrier < idx_arrive, f"barrier_init({mask}) is after barrier_arrive_and_wait({mask})"


def verify_tiles_ops(prim_func: tir.PrimFunc):
    has_fill = False
    has_reduce = False
    has_tiles = False

    def visitor(node):
        nonlocal has_fill, has_reduce, has_tiles
        if isinstance(node, tir.Call):
            if node.op.name == "tl.fill":
                has_fill = True
            elif node.op.name == "tl.reduce":
                has_reduce = True
        elif isinstance(node, tir.For) and "tile.loop_parallel" in node.annotations and "tile.tiled_buffer" in node.annotations:
            has_tiles = True

    tir.stmt_functor.post_order_visit(prim_func.body, visitor)

    def check_lower_tile_op(mod: IRModule):
        script = mod.script()
        if has_fill:
            # fill and clear
            assert "tile.loop_parallel" in script, "Expected tile.loop_parallel in script for fill/clear"
            assert "tile.loop_stage" in script, "Expected tile.loop_stage in script for fill/clear"
            assert "tile.tiled_buffer" in script, "Expected tile.tiled_buffer in script for fill/clear"

        if has_reduce:
            assert "reduce_tile_op" in script, "Expected reduce_tile_op block in script"
            assert "shared_acc" in script, "Expected shared_acc buffer in script"

    def check_legalize_tiles_loop(mod: IRModule):
        script = mod.script()
        if has_tiles:
            assert "tile.buffer_new_shape" in script, "Expected tile.buffer_new_shape in script for Tiles"
            assert "tile.dim_map" in script, "Expected tile.dim_map in script for Tiles"
            assert "tile.tile_size" in script, "Expected tile.tile_size in script for Tiles"
            # At this stage, loop_stage should be 1
            assert '"tile.loop_stage": 1' in script, "Expected tile.loop_stage: 1 in script"

    def check_tiles_loop(mod: IRModule):
        script = mod.script()
        if has_tiles:
            assert "tile.interior" in script, "Expected tile.interior in script for Tiles"
            assert "tile.interior_axis" in script, "Expected tile.interior_axis in script for Tiles"
            # At this stage, loop_stage should be 2
            assert '"tile.loop_stage": 2' in script, "Expected tile.loop_stage: 2 in script"

    return {
        "LowerTileOp": check_lower_tile_op,
        "LegalizeTilesLoop": check_legalize_tiles_loop,
        "TilesLoop": check_tiles_loop,
    }


def verify_host_mod_separation(mod: IRModule):
    script = mod.script()

    # Host module characteristics based on string representation
    assert "tir.is_entry_func" in script, "Host module should contain the entry function attribute"
    assert "tir.is_global_func" not in script, "Host module should not contain device global function attributes"
    assert "T.call_packed" in script or "T.call_extern" in script, (
        "Host module should contain packed or extern calls for kernel launch/errors"
    )
    assert "T.launch_thread" not in script, "Host module should not contain device thread launches"


def verify_device_mod_separation(mod: IRModule):
    script = mod.script()

    # Device module characteristics based on string representation
    assert "tir.is_global_func" in script, "Device module should contain global function attributes"
    assert "tir.is_entry_func" not in script, "Device module should not contain the host entry function attribute"
    assert "__tvm_error_" not in script, "Device module should not contain host-side error checking"


def get_or_add_default_verify(func: tir.PrimFunc, test_config: dict = None):
    verify_ops = verify_tiles_ops(func)
    verify_comm = verify_comm_lower(func)

    default_verify = {
        "LowerTileOp": {
            "formal_verify": [verify_ops["LowerTileOp"], verify_comm],
        },
        "LegalizeTilesLoop": {
            "formal_verify": [verify_ops["LegalizeTilesLoop"]],
        },
        "TilesLoop": {
            "formal_verify": [verify_ops["TilesLoop"]],
        },
        "InjectSunmmioSync": {
            "formal_verify": [verify_SunmmioSync],
        },
        "HostMod": {"formal_verify": [verify_host_mod_separation]},
        "DeviceMod": {
            "formal_verify": [verify_SunmmioSync, verify_comm, verify_device_mod_separation],
        },
    }

    if test_config is None:
        return default_verify

    for key, value in default_verify.items():
        if key not in test_config:
            test_config[key] = {}
        if "formal_verify" not in test_config[key]:
            test_config[key]["formal_verify"] = []
        test_config[key]["formal_verify"].extend(value["formal_verify"])

    return test_config
