"""Write-iteration clocks: actual generated flags, retained values, and boundaries.

The small command interpreter explores every interleaving of the emitted DMA
streams. It checks data versions as well as flag balance/deadlock, rather than
assuming that matching numbers of set/wait calls constitute a correct protocol.
"""

from collections import defaultdict

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, region, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit


def _program(*, versions=2, period=3, trips=7, reads=True, second_write=False, mixed=None, nested=False, common_start=None):
    a = tirx.decl_buffer((2 * max(trips, 1), 64), "float32", name="A")
    c = tirx.decl_buffer((2 * max(trips, 1), 64), "float32", name="C")
    cached = tirx.decl_buffer((64,), "float32", name="cached", scope="shared.dyn")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    common = tirx.const(True, "bool") if common_start is None else i >= common_start
    write_guard = i % period == 0 if common_start is None else tirx.And(common, i % period == 0)
    tasks = [unit(copy(a, cached, src_indices=[i, 0], src_shape=[1, 64]), guard=write_guard)]
    if second_write:
        write_guard = i % 2 == 0 if common_start is None else tirx.And(common, i % 2 == 0)
        tasks.append(unit(copy(a, cached, src_indices=[trips + i, 0], src_shape=[1, 64]), guard=write_guard))
    read_guard = None if reads is True else i % 2 == 1 if reads else i < 0
    if common_start is not None:
        read_guard = common if read_guard is None else tirx.And(common, read_guard)
    read = unit(copy(cached, c, dst_indices=[i, 0], dst_shape=[1, 64]), guard=read_guard)
    if nested:
        # Includes empty consumer loops, without suppressing the generation.
        read = unit(tirx.For(j, 0, i % 3, tirx.ForKind.SERIAL, read), core=None)
    tasks.append(read)
    buffers = [cached]
    counts = {cached.data: versions}
    modes = {cached.data: "write_counter"}
    if mixed:
        ordinary = tirx.decl_buffer((64,), "float32", name="ordinary", scope="shared.dyn")
        buffers.append(ordinary)
        counts[ordinary.data] = 2
        modes[ordinary.data] = mixed
        tasks.insert(0, unit(copy(a, ordinary, src_indices=[i, 0], src_shape=[1, 64])))
        tasks.append(unit(copy(ordinary, c, dst_indices=[trips + i, 0], dst_shape=[1, 64])))
    owner = tirx.For(i, 0, trips, tirx.ForKind.SERIAL, seq(*tasks), annotations={"multi_buffer_eligible": list(counts)})
    return kernel(
        unit(owner, core=None),
        buffers=buffers,
        params=[a, c],
        annotations={"tl.buffer_versions_map": counts, "tl.buffer_version_mode": modes},
    )


def _physical(before):
    mod = transform.PrepareMultiBuffer()(before)
    # Exercise serialization: the prepared clock kind must survive decoding.
    mod = tvm.ir.load_json(tvm.ir.save_json(mod))
    mod = transform.ResolveCore()(mod)
    mod = transform.InsertSync()(mod)
    return transform.MaterializeMultiBuffer()(mod)


def _streams(mod, *, coalesce_allocations=False, counter_updates=None):
    scalar, variables, reference = {}, {}, {}
    streams = defaultdict(list)
    fence = 0
    versioned = {
        buffer.name
        for block in nodes(mod, tirx.SBlock)
        for buffer in block.alloc_buffers
        if buffer.scope() == "shared.dyn" and len(buffer.shape) == 2
    }

    def emit(pipe, command):
        nonlocal fence
        if pipe != "S" and fence:
            streams[pipe].append(("fence", fence))
        streams[pipe].append(command)
        if pipe == "S":
            fence = len(streams[pipe])

    def value(expr):
        if isinstance(expr, tirx.IntImm):
            return int(expr)
        if isinstance(expr, tirx.Var):
            return variables[expr]
        if isinstance(expr, tirx.BufferLoad):
            return scalar[expr.buffer.data]
        if isinstance(expr, tirx.Cast):
            return value(expr.value)
        if isinstance(expr, tirx.Not):
            return not value(expr.a)
        if isinstance(expr, tirx.Select):
            return value(expr.true_value if value(expr.condition) else expr.false_value)
        if isinstance(expr, tirx.And):
            return value(expr.a) and value(expr.b)
        if isinstance(expr, tirx.Or):
            return value(expr.a) or value(expr.b)
        operations = {
            tirx.Add: lambda a, b: a + b,
            tirx.Sub: lambda a, b: a - b,
            tirx.Mul: lambda a, b: a * b,
            tirx.FloorDiv: lambda a, b: a // b,
            tirx.FloorMod: lambda a, b: a % b,
            tirx.EQ: lambda a, b: a == b,
            tirx.NE: lambda a, b: a != b,
            tirx.LT: lambda a, b: a < b,
            tirx.LE: lambda a, b: a <= b,
            tirx.GT: lambda a, b: a > b,
            tirx.GE: lambda a, b: a >= b,
        }
        return operations[type(expr)](value(expr.a), value(expr.b))

    def address(region):
        load = region.args[0]
        return (load.buffer.name, tuple(value(index) for index in load.indices))

    def visit(stmt):
        if isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                visit(child)
        elif isinstance(stmt, tirx.SBlockRealize):
            visit(stmt.block)
        elif isinstance(stmt, (tirx.AttrStmt, tirx.SBlock)):
            visit(stmt.body)
        elif isinstance(stmt, tirx.For):
            start, extent = value(stmt.min), value(stmt.extent)
            for index in range(start, start + extent, 1 if stmt.step is None else value(stmt.step)):
                variables[stmt.loop_var] = index
                visit(stmt.body)
        elif isinstance(stmt, tirx.IfThenElse):
            branch = stmt.then_case if value(stmt.condition) else stmt.else_case
            if branch is not None:
                visit(branch)
        elif isinstance(stmt, tirx.BufferStore):
            scalar[stmt.buffer.data] = value(stmt.value)
            if counter_updates is not None and "version_counter" in stmt.buffer.name and scalar[stmt.buffer.data] >= 0:
                counter_updates.append((stmt.buffer.name, scalar[stmt.buffer.data]))
        elif isinstance(stmt, tirx.Bind):
            variables[stmt.var] = value(stmt.value)
        elif isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
            call = stmt.value
            name = call.op.name
            if name in ("tl.ascend_set_flag", "tl.ascend_wait_flag"):
                source, destination = str(call.args[0].value).split("_")
                operation = "set" if name == "tl.ascend_set_flag" else "wait"
                emit(source if operation == "set" else destination, (operation, (source, destination, value(call.args[1]))))
            elif name == "tl.tileop.ascend_copy":
                src, dst = map(address, call.args[:2])

                def logical(address):
                    name, indices = address
                    return (name, indices[1:]) if name in versioned else address

                expected = src if src[0] == "A" else reference[logical(src)]
                reference[logical(dst)] = expected
                emit("MTE2" if src[0] == "A" else "MTE3" if dst[0] == "C" else "V", ("copy", src, dst, expected))
            elif name == "tl.ascend_pipe_barrier":
                # This model uses ordered, atomic DMA completions. Same-pipe
                # overlap hazards need separate hardware/footprint coverage.
                pass
            else:
                raise AssertionError(f"unmodeled call {name}")

    visit(mod["main"].body)
    if coalesce_allocations:
        root = next(block for block in nodes(mod, tirx.SBlock) if block.name_hint == "tilelang_root")
        aliases = root.annotations["tl.buffer_alias_map"]
        buffers = {buf.name: buf for buf in root.alloc_buffers if buf.scope() == "shared.dyn"}
        groups = []
        for buffer in buffers.values():
            for group in groups:
                if all(any(var.same_as(other.data) for var in aliases[buffer.data]) for other in group):
                    group.append(buffer)
                    break
            else:
                groups.append([buffer])
        allocation = {buf.name: f"allocation{index}" for index, group in enumerate(groups) for buf in group}

        def physical(address):
            name, indices = address
            if name not in allocation:
                return address
            offset = 0
            for index, extent in zip(indices, buffers[name].shape):
                offset = offset * int(extent) + index
            return allocation[name], (offset,)

        for pipe, commands in streams.items():
            streams[pipe] = [
                ("copy", physical(command[1]), physical(command[2]), command[3]) if command[0] == "copy" else command
                for command in commands
            ]
    return streams


def _check_all_interleavings(streams):
    scalar_lane = list(streams).index("S") if "S" in streams else None
    streams = list(streams.values())
    initial = ((0,) * len(streams), frozenset(), ())
    pending, visited = [initial], {initial}
    complete = 0
    while pending:
        positions, flags, items = pending.pop()
        memory = dict(items)
        if all(position == len(stream) for position, stream in zip(positions, streams)):
            assert not flags, "undrained flag tokens"
            complete += 1
            continue
        successors = 0
        for lane, (position, stream) in enumerate(zip(positions, streams)):
            if position == len(stream):
                continue
            command = stream[position]
            new_flags, new_memory = set(flags), dict(memory)
            if command[0] == "fence":
                if positions[scalar_lane] < command[1]:
                    continue
            elif command[0] == "wait":
                if command[1] not in flags:
                    continue
                new_flags.remove(command[1])
            elif command[0] == "set":
                assert command[1] not in flags, "double publication of one flag"
                new_flags.add(command[1])
            else:
                _, src, dst, expected = command
                observed = src if src[0] == "A" else memory.get(src)
                assert observed == expected, f"wrong generation at {src}: {observed} != {expected}"
                new_memory[dst] = observed
            successors += 1
            advanced = list(positions)
            advanced[lane] += 1
            state = (tuple(advanced), frozenset(new_flags), tuple(sorted(new_memory.items())))
            if state not in visited:
                visited.add(state)
                pending.append(state)
        assert successors, "generated protocol deadlocks"
    assert complete


@pytest.mark.parametrize("versions", [1, 2, 3])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"second_write": True},
        {"reads": False},
        {"reads": "conditional"},
        {"nested": True},
        {"trips": 0},
        {"mixed": "iteration"},
        {"mixed": "counter"},
        {"mixed": "counter", "period": 1},
        {"mixed": "auto", "second_write": True, "nested": True},
        {"mixed": "iteration", "common_start": 3},
        {"mixed": "counter", "common_start": 3},
        {"mixed": "counter", "common_start": 3, "nested": True},
    ],
)
def test_write_generations_and_mixed_clocks(versions, options):
    after = _physical(_program(versions=versions, **options))
    _check_all_interleavings(_streams(after))
    # Version-one protocols still need the acquired-generation state.
    counters = [buffer for block in nodes(after, tirx.SBlock) for buffer in block.alloc_buffers if "version_counter" in buffer.name]
    assert counters


def test_retained_reads_do_not_repeat_handshakes():
    streams = _streams(_physical(_program(trips=7, period=3)))
    commands = [command for stream in streams.values() for command in stream]
    # Three writes: READY + FREE, each set/wait once. Two FREE slots are
    # initialized and drained. Reads between writes add no hardware events.
    assert sum(command[0] in ("set", "wait") for command in commands) == 3 * 4 + 2 * 2


@pytest.mark.parametrize("bind_guards", [False, True])
@pytest.mark.parametrize(
    "case", ["invariant", "invariant-snapshot", "delayed", "periodic", "interval", "gaps", "disjoint", "step", "multiple-writes"]
)
def test_guarded_read_history_and_sync_placement(case, bind_guards):
    trips = 8
    a, c = [tirx.decl_buffer((2 * trips, 64), "float32", name=name) for name in ("A", "C")]
    x = tirx.decl_buffer((64,), "float32", name="x", scope="shared.dyn")
    i, start = tirx.Var("i", "int32"), tirx.Var("start", "int32")
    parameters, prelude = [a, c, start], []
    if case == "invariant-snapshot":
        predicate = tirx.decl_buffer((1,), "int32", name="predicate")
        parameters, prelude = [a, c, predicate], [unit(tirx.Bind(start, predicate[0]))]
    common, refresh = i >= start, (i - start) % 2 == 0
    if case in ("invariant", "invariant-snapshot"):
        common, refresh = start > 0, i % 2 == 0
    elif case == "periodic":
        common, refresh = tirx.all(i >= start, (i - start) % 2 == 0), i == start
    elif case == "interval":
        common = tirx.all(i >= start, i < start + 3)
    elif case == "gaps":
        common, refresh = tirx.all(i >= start, (i - start) % 3 != 2), (i - start) % 6 == 0
    elif case == "disjoint":
        common, refresh = tirx.any(i == start, i >= start + 5), i == start
    elif case == "step":
        common, refresh = i >= 2 * start, (i - 2 * start) % 4 == 0
    bindings = []
    if bind_guards:
        enabled, renew = tirx.Var("enabled", "bool"), tirx.Var("renew", "bool")
        bindings = [unit(tirx.Bind(enabled, common)), unit(tirx.Bind(renew, refresh), guard=enabled)]
        common, refresh = enabled, renew
    tasks = [
        unit(copy(a, x, src_indices=[i, 0], src_shape=[1, 64]), guard=tirx.And(common, refresh)),
        unit(copy(x, c, dst_indices=[i, 0], dst_shape=[1, 64]), guard=common),
    ]
    if case == "multiple-writes":
        tasks += [
            unit(copy(a, x, src_indices=[trips + i, 0], src_shape=[1, 64]), guard=tirx.And(common, i % 3 == 0)),
            unit(copy(x, c, dst_indices=[trips + i, 0], dst_shape=[1, 64]), guard=common),
        ]
    owner = tirx.For(
        i,
        0,
        trips,
        tirx.ForKind.SERIAL,
        seq(*bindings, *tasks),
        annotations={"multi_buffer_eligible": [x.data]},
        step=2 if case == "step" else None,
    )
    before = kernel(
        seq(*prelude, unit(owner, core=None, guard=start >= 0)),
        buffers=[x],
        params=parameters,
        annotations={"tl.buffer_versions_map": {x.data: 2}, "tl.buffer_version_mode": {x.data: "write_counter"}},
    )
    after = _physical(before)
    in_loop = [(stmt, parents) for stmt, parents in statements(after) if any(isinstance(p, tirx.For) for p in parents)]
    reader_guards = [
        next(p.condition for p in reversed(parents) if isinstance(p, tirx.IfThenElse))
        for stmt, parents in in_loop
        if isinstance(stmt, tirx.Evaluate)
        and isinstance(stmt.value, tirx.Call)
        and stmt.value.op.name == "tl.tileop.ascend_copy"
        and stmt.value.args[1].args[0].buffer.name == "C"
    ]
    analyzer = tvm.arith.Analyzer()
    for stmt, parents in in_loop:
        if (
            isinstance(stmt, tirx.Evaluate)
            and isinstance(stmt.value, tirx.Call)
            and stmt.value.op.name
            in (
                "tl.ascend_set_flag",
                "tl.ascend_wait_flag",
                "tl.ascend_pipe_barrier",
            )
        ):
            active = tirx.all(*(p.condition for p in parents if isinstance(p, tirx.IfThenElse)))
            assert analyzer.can_prove(tirx.Or(tirx.Not(active), reader_guards[0])), "sync escaped the common guard"
    for first in (0, 1, 3, trips):
        func = after["main"]
        value = tirx.const(first, "int32")
        if case == "invariant-snapshot":
            body = tirx.stmt_functor.ir_transform(
                func.body,
                None,
                lambda node, value=value: value if isinstance(node, tirx.BufferLoad) and node.buffer.name == "predicate" else None,
            )
        else:
            body = tirx.stmt_functor.substitute(func.body, {func.params[-1]: value})
        streams = _streams(tvm.IRModule({"main": func.with_body(body)}))
        _check_all_interleavings(streams)
        commands = [command for stream in streams.values() for command in stream]
        writes = sum(command[0] == "copy" and command[1][0] == "A" for command in commands)
        if case != "multiple-writes":
            assert sum(command[0] in ("set", "wait") for command in commands) == writes * 4 + 2 * 2


@pytest.mark.parametrize(
    "case", ["late-first-write", "changing-snapshot", "conditional-snapshot", "independent-read-guard", "after-read", "unreachable-write"]
)
def test_guard_history_does_not_invent_an_initialization(case):
    a, c = [tirx.decl_buffer((8, 64), "float32", name=name) for name in ("A", "C")]
    predicate = tirx.decl_buffer((8,), "int32", name="predicate")
    x = tirx.decl_buffer((64,), "float32", name="x", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    common, refresh = i >= 1, i % 2 == 0
    bindings = []
    if case == "changing-snapshot":
        enabled = tirx.Var("enabled", "bool")
        bindings = [unit(tirx.Bind(enabled, predicate[i] > 0))]
        common = enabled
    write_guard = tirx.And(common, refresh)
    read_guard = common if case != "independent-read-guard" else i % 2 == 1
    if case == "conditional-snapshot":
        enabled = tirx.Var("enabled", "bool")
        bindings = [unit(tirx.Bind(enabled, predicate[i] > 0), guard=i >= 0)]
        read_guard = tirx.And(i >= 0, enabled)
        write_guard = tirx.And(read_guard, i % 2 == 0)
    if case == "independent-read-guard":
        write_guard = i % 3 == 2
    if case == "after-read":
        write_guard = read_guard = i >= 1
    if case == "unreachable-write":
        write_guard = i == 1
    tasks = [
        unit(copy(a, x, src_indices=[i, 0], src_shape=[1, 64]), guard=write_guard),
        unit(copy(x, c, dst_indices=[i, 0], dst_shape=[1, 64]), guard=read_guard),
    ]
    if case == "after-read":
        tasks.reverse()
    owner = tirx.For(
        i,
        0,
        8,
        tirx.ForKind.SERIAL,
        seq(*bindings, *tasks),
        annotations={"multi_buffer_eligible": [x.data]},
        step=2 if case == "unreachable-write" else None,
    )
    before = kernel(
        unit(owner, core=None),
        buffers=[x],
        params=[a, c, predicate],
        annotations={"tl.buffer_versions_map": {x.data: 2}, "tl.buffer_version_mode": {x.data: "write_counter"}},
    )
    with pytest.raises(tvm.error.InternalError, match="initialization"):
        transform.PrepareMultiBuffer()(before)


def test_write_only_dma_ring_fences_slot_reuse_and_completion():
    before = _program()

    def remove_reader(node):
        if isinstance(node, tirx.For):
            return tirx.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                node.body.seq[0],
                annotations=node.annotations,
            )

    before = tvm.IRModule({"main": before["main"].with_body(tirx.stmt_functor.ir_transform(before["main"].body, None, remove_reader))})
    after = _physical(before)
    assert not calls(after, "tl.ascend_set_flag")
    # One guarded barrier before ring wrap, plus a final completion barrier.
    assert len(calls(after, "tl.ascend_pipe_barrier")) == 2
    _check_all_interleavings(_streams(after))


def test_irregular_refresh_is_not_interpreted_as_an_iteration_distance():
    before = _program()
    i = nodes(before, tirx.For)[0].loop_var

    def mutate(node):
        if isinstance(node, tirx.IfThenElse):
            return tirx.IfThenElse(tirx.any(i == 0, i == 1, i == 5), node.then_case, None)

    before = tvm.IRModule({"main": before["main"].with_body(tirx.stmt_functor.ir_transform(before["main"].body, None, mutate))})
    _check_all_interleavings(_streams(_physical(before)))


def test_independent_write_clocks_in_the_same_owner():
    a = tirx.decl_buffer((8, 64), "float32", name="A")
    c = tirx.decl_buffer((8, 64), "float32", name="C")
    x, y = [tirx.decl_buffer((64,), "float32", name=name, scope="shared.dyn") for name in ("x", "y")]
    i = tirx.Var("i", "int32")
    body = seq(
        unit(copy(a, x, src_indices=[i, 0], src_shape=[1, 64]), guard=i % 2 == 0),
        unit(copy(a, y, src_indices=[4 + i, 0], src_shape=[1, 64]), guard=i % 3 == 0),
        unit(copy(x, c, dst_indices=[i, 0], dst_shape=[1, 64])),
        unit(copy(y, c, dst_indices=[4 + i, 0], dst_shape=[1, 64])),
    )
    before = kernel(
        unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [x.data, y.data]}), core=None),
        buffers=[x, y],
        params=[a, c],
        annotations={
            "tl.buffer_versions_map": {x.data: 2, y.data: 2},
            "tl.buffer_version_mode": {x.data: "write_counter", y.data: "write_counter"},
        },
    )
    counters = nodes(transform.PrepareMultiBuffer()(before), tirx.For)[0].annotations["tl.multi_buffer_counter_map"]
    assert not counters[x.data].same_as(counters[y.data])
    _check_all_interleavings(_streams(_physical(before)))


@pytest.mark.parametrize("owner_policy", ["explicit", "inferred", "nested-inferred"])
@pytest.mark.parametrize("second_writes", [False, True], ids=["read-only", "read-then-update"])
def test_disjoint_owners_share_the_last_written_generation(owner_policy, second_writes):
    before = _program(trips=3, period=2)
    func = before["main"]
    root = func.body.block
    cached = root.alloc_buffers[0]
    out = next(buffer for buffer in func.buffer_map.values() if buffer.name == "C")
    source = next(buffer for buffer in func.buffer_map.values() if buffer.name == "A")
    j = tirx.Var("reader_owner", "int32")
    read = unit(copy(cached, out, dst_indices=[3 + j, 0], dst_shape=[1, 64]))
    body = read
    if second_writes:
        body = seq(read, unit(copy(source, cached, src_indices=[3 + j, 0], src_shape=[1, 64]), guard=j == 1), read)
    reader = unit(
        tirx.For(
            j,
            0,
            2,
            tirx.ForKind.SERIAL,
            body,
            annotations={"multi_buffer_eligible": [cached.data]},
        ),
        core=None,
    )
    before = kernel(seq(root.body, reader), buffers=root.alloc_buffers, params=list(func.buffer_map.values()), annotations=root.annotations)
    if owner_policy != "explicit":

        def remove_claim(node):
            if isinstance(node, tirx.For):
                return tirx.For(node.loop_var, node.min, node.extent, node.kind, node.body)

        body = tirx.stmt_functor.ir_transform(before["main"].body.block.body, None, remove_claim)
        if owner_policy == "nested-inferred":
            body = unit(tirx.For(tirx.Var("outer", "int32"), 0, 2, tirx.ForKind.SERIAL, body), core=None)
        before = kernel(body, buffers=root.alloc_buffers, params=list(func.buffer_map.values()), annotations=root.annotations)
        before = transform.AnnotateMultiBufferEligible()(before)
    owners = [loop for loop in nodes(before, tirx.For) if cached.data in loop.annotations.get("multi_buffer_eligible", [])]
    assert len(owners) == 2
    updates = []
    streams = _streams(_physical(before), counter_updates=updates)
    count = (2 + int(second_writes)) * (2 if owner_policy == "nested-inferred" else 1)
    assert [generation for _, generation in updates] == list(range(count))
    _check_all_interleavings(streams)


def test_disjoint_write_protocols_reuse_flags_and_buffers_after_drain():
    a = tirx.decl_buffer((8, 64), "float32", name="A")
    c = tirx.decl_buffer((8, 64), "float32", name="C")
    buffers, loops, counts, modes = [], [], {}, {}
    for index in range(3):
        buffer = tirx.decl_buffer((64,), "float32", name=f"cached{index}", scope="shared.dyn")
        i = tirx.Var(f"i{index}", "int32")
        body = seq(
            unit(copy(a, buffer, src_indices=[index * 2 + i, 0], src_shape=[1, 64])),
            unit(copy(buffer, c, dst_indices=[index * 2 + i, 0], dst_shape=[1, 64])),
        )
        loops.append(unit(tirx.For(i, 0, 2, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [buffer.data]}), core=None))
        buffers.append(buffer)
        counts[buffer.data], modes[buffer.data] = 2, "write_counter"
    before = kernel(
        seq(*loops), buffers=buffers, params=[a, c], annotations={"tl.buffer_versions_map": counts, "tl.buffer_version_mode": modes}
    )
    after = _physical(before)
    streams = _streams(after)
    _check_all_interleavings(streams)
    event_ids = defaultdict(set)
    for stream in streams.values():
        for command in stream:
            if command[0] in ("set", "wait"):
                source, destination, event_id = command[1]
                event_ids[source, destination].add(event_id)
    assert len(event_ids["MTE2", "MTE3"]) == 2
    assert len(event_ids["MTE3", "MTE2"]) == 2
    assert len(event_ids["MTE2", "S"]) == 1
    root = next(block for block in nodes(after, tirx.SBlock) if block.name_hint == "tilelang_root")
    aliases = root.annotations["tl.buffer_alias_map"]
    stored = {buffer.name: buffer.data for buffer in root.alloc_buffers}
    assert all(len(aliases[stored[buffer.name]]) == 2 for buffer in buffers)


@pytest.mark.parametrize("y_versions", [2, 3])
def test_same_write_event_shares_clock_and_compatible_consumer_channel(y_versions):
    a, c = tirx.decl_buffer((64,), "float32", name="A"), tirx.decl_buffer((64,), "float32", name="C")
    x, y = [tirx.decl_buffer((64,), "float32", name=name, scope="shared.dyn") for name in ("x", "y")]
    i = tirx.Var("i", "int32")
    tasks = seq(unit(seq(copy(a, x), copy(a, y)), guard=i % 3 == 0), unit(copy(x, c)), unit(copy(y, c)))
    owner = tirx.For(i, 0, 7, tirx.ForKind.SERIAL, tasks, annotations={"multi_buffer_eligible": [x.data, y.data]})
    before = kernel(
        unit(owner, core=None),
        buffers=[x, y],
        params=[a, c],
        annotations={
            "tl.buffer_versions_map": {x.data: 2, y.data: y_versions},
            "tl.buffer_version_mode": {x.data: "write_counter", y.data: "write_counter"},
        },
    )
    after = transform.PrepareMultiBuffer()(before)
    counters = nodes(after, tirx.For)[0].annotations["tl.multi_buffer_counter_map"]
    assert counters[x.data].same_as(counters[y.data])
    after = transform.InsertSync()(transform.ResolveCore()(after))
    # Different ring sizes share the write clock but need separate channels.
    acquired = [buffer for block in nodes(after, tirx.SBlock) for buffer in block.alloc_buffers if "_acquired_" in buffer.name]
    assert len(acquired) == (1 if y_versions == 2 else 2)
    _check_all_interleavings(_streams(_physical(before)))


@pytest.mark.parametrize("subcores", [1, 2])
def test_write_clock_cross_core_flags_cover_every_consumer_bank(subcores):
    accum = tirx.decl_buffer((1, 64), "float32", name="accum", scope="shared.l0c")
    ub = tirx.decl_buffer((1, 64), "float32", name="ub", scope="shared.dyn")
    out = tirx.decl_buffer((1, 64), "float32", name="C")
    i = tirx.Var("i", "int32")
    vector = 1 if subcores == 1 else 5
    body = seq(unit(copy(accum, ub), core=2, guard=i % 3 == 0), unit(copy(ub, out), core=vector))
    before = kernel(
        unit(tirx.For(i, 0, 7, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [ub.data]}), core=None),
        buffers=[accum, ub],
        params=[out],
        annotations={
            "vector_count": tirx.IntImm("int32", subcores),
            "tl.buffer_versions_map": {ub.data: 2},
            "tl.buffer_version_mode": {ub.data: "write_counter"},
        },
    )
    prepared = transform.PrepareMultiBuffer()(before)
    resolved = transform.ResolveCore()(prepared)
    updates = [task for task in nodes(resolved, tirx.AttrStmt) if task.attr_key == "tl.ascend_task" and nodes(task.body, tirx.BufferStore)]
    assert {int(task.node["core_mask"]) for task in updates} == {vector | 2}
    after = transform.InsertSync()(resolved)
    cube_drains = [
        int(call.args[2])
        for task in nodes(after, tirx.AttrStmt)
        if task.attr_key == "tl.ascend_task" and int(task.node["core_mask"]) == 2
        for call in calls(task.body, "tl.ascend_cross_core_wait_flag")
        if isinstance(call.args[2], tirx.IntImm)
    ]
    assert len(cube_drains) == subcores * 2
    assert {event_id // 16 for event_id in cube_drains} == set(range(subcores))
    assert all(int(call.args[0]) == 4 for call in calls(after, "tl.ascend_cross_core_set_flag"))


@pytest.mark.parametrize("bind_aliases", [False, True])
def test_periodic_partial_values_cover_tail_readers(bind_aliases):
    a = tirx.decl_buffer((6, 64), "float16", name="A")
    c = tirx.decl_buffer((16,), "float16", name="C", scope="shared.l0a")
    x = tirx.decl_buffer((64,), "float16", name="cached", scope="shared.l1")
    i = tirx.Var("i", "int32")
    span = tirx.min(64, (6 - i) * 16)
    refresh = i % 4 == 0
    bindings = []
    if bind_aliases:
        bound_span, bound_refresh = tirx.Var("span", "int32"), tirx.Var("refresh", "bool")
        bindings = [unit(tirx.Bind(bound_span, span), core=2), unit(tirx.Bind(bound_refresh, refresh), core=2)]
        span, refresh = bound_span, bound_refresh
    body = seq(
        *bindings,
        unit(copy(a, x, src_indices=[i, 0], src_shape=[1, span], dst_shape=[span]), core=2, guard=refresh),
        unit(copy(x, c, src_indices=[i % 4 * 16], src_shape=[16]), core=2),
    )
    before = kernel(
        unit(tirx.For(i, 0, 6, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [x.data]}), core=None),
        buffers=[x, c],
        params=[a],
        annotations={"tl.buffer_versions_map": {x.data: 2}, "tl.buffer_version_mode": {x.data: "write_counter"}},
    )
    after = _physical(before)
    assert calls(after, "tl.ascend_wait_flag")


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("dtype", ["uint8", "int16"])
def test_sf_padding_does_not_require_a_region_coverage_proof(transpose, dtype):
    from tilelang.layout import make_ascend_sf_layout

    a = tirx.decl_buffer((2, 17) if transpose else (17, 2), dtype, name="A")
    x = tirx.decl_buffer((32, 2), dtype, name="cached", scope="shared.l1")
    c = tirx.decl_buffer((32, 2), dtype, name="C", scope="shared.l0a")
    i = tirx.Var("i", "int32")
    body = seq(
        unit(copy(a, x, dst_shape=[17, 2], annotations={"transpose": tirx.IntImm("int32", int(transpose))}), core=2, guard=i % 3 == 0),
        unit(copy(x, c), core=2),
    )
    before = kernel(
        unit(tirx.For(i, 0, 5, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [x.data]}), core=None),
        buffers=[x, c],
        params=[a],
        annotations={
            "tl.buffer_versions_map": {x.data: 2},
            "tl.buffer_version_mode": {x.data: "write_counter"},
            "layout_map": {x: make_ascend_sf_layout(x)},
        },
    )
    assert calls(_physical(before), "tl.ascend_wait_flag")


@pytest.mark.parametrize("snapshot", [False, True])
def test_sf_padding_preserves_the_dynamic_extent_snapshot(snapshot):
    from tilelang.layout import make_ascend_sf_layout

    a = tirx.decl_buffer((2, 64), "int16", name="A")
    x = tirx.decl_buffer((64, 2), "int16", name="cached", scope="shared.l1")
    c = tirx.decl_buffer((64, 2), "int16", name="C", scope="shared.l0a")
    lengths = tirx.decl_buffer((1,), "int32", name="lengths")
    i, n = tirx.Var("i", "int32"), tirx.Var("n", "int32")
    padded = (n + 15) // 16 * 16
    body = seq(
        unit(copy(a, x, src_shape=[2, n], dst_shape=[n, 2], annotations={"transpose": tirx.IntImm("int32", 1)}), core=2, guard=i % 2 == 0),
        unit(copy(x, c, src_shape=[padded, 2], dst_shape=[padded, 2]), core=2),
    )
    owner = unit(
        tirx.For(i, 0, 3, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [x.data]}),
        core=None,
        guard=(n > 0) & (n <= 64),
    )
    before = kernel(
        seq(unit(tirx.Bind(n, lengths[0]), core=2), owner) if snapshot else owner,
        buffers=[x, c],
        params=[a, lengths] if snapshot else [a, n],
        annotations={
            "tl.buffer_versions_map": {x.data: 2},
            "tl.buffer_version_mode": {x.data: "write_counter"},
            "layout_map": {x: make_ascend_sf_layout(x)},
        },
    )
    assert calls(_physical(before), "tl.ascend_wait_flag")


@pytest.mark.parametrize("case", ["read-before-write", "read-modify-write", "cross-stage"])
def test_ambiguous_write_generations_are_diagnosed(case):
    before = _program()
    if case == "read-before-write":

        def mutate(node):
            if isinstance(node, tirx.IfThenElse):
                return tirx.IfThenElse(node.condition & (nodes(before, tirx.For)[0].loop_var > 0), node.then_case, None)

        pattern = "initialization"
    elif case == "read-modify-write":

        def mutate(node):
            if isinstance(node, tirx.Evaluate) and isinstance(node.value, tirx.Call) and node.value.op.name == "tl.tileop.ascend_copy":
                dst = node.value.args[1].args[0].buffer
                if dst.name == "cached":
                    return tirx.BufferStore(dst, dst[0] + 1, [0])

        pattern = "read-modify-write"
    else:

        def mutate(node):
            if (
                isinstance(node, tirx.AttrStmt)
                and node.attr_key == "tl.schedule_unit"
                and calls(node.body, "tl.tileop.ascend_copy")
                and not nodes(node.body, tirx.For)
                and not nodes(node.body, tirx.IfThenElse)
            ):
                return tirx.AttrStmt({"stage": 1}, node.attr_key, node.value, node.body)

        pattern = "multiple schedule stages"
    body = tirx.stmt_functor.ir_transform(before["main"].body, None, mutate)
    before = tvm.IRModule({"main": before["main"].with_body(body)})
    with pytest.raises(tvm.error.InternalError, match=pattern):
        transform.PrepareMultiBuffer()(before)


@pytest.mark.parametrize("case", ["opaque", "mutable-guard", "loop-break"])
def test_unsupported_write_protocols_fail_explicitly(case):
    before = _program()
    cached = before["main"].body.block.alloc_buffers[0]

    def mutate(node):
        if case == "mutable-guard" and isinstance(node, tirx.IfThenElse):
            return tirx.IfThenElse(cached[0] > 0, node.then_case, None)
        if (
            isinstance(node, tirx.Evaluate)
            and isinstance(node.value, tirx.Call)
            and node.value.op.name == "tl.tileop.ascend_copy"
            and node.value.args[1].args[0].buffer.name == "cached"
        ):
            if case == "opaque":
                return seq(node, tirx.Evaluate(tirx.call_extern("int32", "opaque_write", cached.data)))
            if case == "loop-break":
                return seq(node, tirx.Evaluate(tirx.call_intrin("handle", tvm.ir.Op.get("tl.loop_break"))))

    before = tvm.IRModule({"main": before["main"].with_body(tirx.stmt_functor.ir_transform(before["main"].body, None, mutate))})
    pattern = {
        "opaque": "opaque pointer",
        "mutable-guard": "unsnapshotted guard",
        "loop-break": "loop_break",
    }[case]
    with pytest.raises(tvm.error.InternalError, match=pattern):
        _physical(before)


def test_write_counter_requires_an_explicit_ring_size():
    from tilelang.ascend.language import annotate_buffer_versions

    buffer = tirx.decl_buffer((64,), "float32", scope="shared.dyn")
    with pytest.raises(ValueError, match="explicit version count"):
        annotate_buffer_versions({buffer: "write_counter"})


@pytest.mark.parametrize("ordinary_mode", ["iteration", "counter", "auto"])
def test_write_completion_eliminates_mixed_mode_raw(ordinary_mode):
    """One real acquire discharges both operands of the same producer Task."""
    a, c = [tirx.decl_buffer((8, 64), "float32", name=name) for name in ("A", "C")]
    x, y = [tirx.decl_buffer((64,), "float32", name=name, scope="shared.dyn") for name in ("x", "y")]
    i = tirx.Var("i", "int32")
    body = seq(
        unit(seq(copy(a, x, src_indices=[i, 0], src_shape=[1, 64]), copy(a, y, src_indices=[4 + i, 0], src_shape=[1, 64]))),
        unit(seq(copy(x, c, dst_indices=[i, 0], dst_shape=[1, 64]), copy(y, c, dst_indices=[4 + i, 0], dst_shape=[1, 64]))),
    )
    before = kernel(
        unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [x.data, y.data]}), core=None),
        buffers=[x, y],
        params=[a, c],
        annotations={
            "tl.buffer_versions_map": {x.data: 2, y.data: 2},
            "tl.buffer_version_mode": {x.data: "write_counter", y.data: ordinary_mode},
        },
    )
    after = _physical(before)
    _check_all_interleavings(_streams(after))
    ready = [call for call in calls(after, "tl.ascend_set_flag") if call.args[0].value == "MTE2_MTE3"]
    assert len(ready) == 1, "the ordinary RAW must use the write clock's completion certificate"


@pytest.mark.parametrize("ordinary_mode", ["iteration", "counter", "auto"])
@pytest.mark.parametrize("reverse", [False, True])
def test_closed_write_channel_reuses_flags_and_allocation_with_old_mode(ordinary_mode, reverse):
    a, c = [tirx.decl_buffer((8, 64), "float32", name=name) for name in ("A", "C")]
    x, y = [tirx.decl_buffer((64,), "float32", name=name, scope="shared.dyn") for name in ("x", "y")]
    loops = []
    for index, buf in enumerate((x, y)):
        i = tirx.Var(f"i{index}", "int32")
        body = seq(
            unit(copy(a, buf, src_indices=[4 * index + i, 0], src_shape=[1, 64])),
            unit(copy(buf, c, dst_indices=[4 * index + i, 0], dst_shape=[1, 64])),
        )
        loops.append(unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [buf.data]}), core=None))
    before = kernel(
        seq(*(loops[::-1] if reverse else loops)),
        buffers=[x, y],
        params=[a, c],
        annotations={
            "tl.buffer_versions_map": {x.data: 2, y.data: 2},
            "tl.buffer_version_mode": {x.data: "write_counter", y.data: ordinary_mode},
        },
    )
    after = _physical(before)
    streams = _streams(after, coalesce_allocations=True)
    _check_all_interleavings(streams)
    ids = {
        command[1][2] for commands in streams.values() for command in commands if command[0] == "set" and command[1][:2] == ("MTE2", "MTE3")
    }
    assert len(ids) == 2, "disjoint new/old READY channels should reuse the same ring"
    root = next(block for block in nodes(after, tirx.SBlock) if block.name_hint == "tilelang_root")
    aliases = root.annotations["tl.buffer_alias_map"]
    storage = {buf.name: buf.data for buf in root.alloc_buffers}
    assert any(other.same_as(storage["y"]) for other in aliases[storage["x"]])


def test_snapshot_footprint_is_stable_within_one_owner_invocation():
    a, c = [tirx.decl_buffer((64,), "float32", name=name) for name in ("A", "C")]
    lengths = tirx.decl_buffer((1,), "int32", name="lengths")
    x = tirx.decl_buffer((64,), "float32", name="x", scope="shared.dyn")
    i, n = tirx.Var("i", "int32"), tirx.Var("n", "int32")
    body = seq(unit(copy(a, x, src_shape=[n], dst_shape=[n]), guard=i % 2 == 0), unit(copy(x, c, src_shape=[n], dst_shape=[n])))
    owner = unit(
        tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [x.data]}), core=None, guard=(n > 0) & (n <= 64)
    )
    before = kernel(
        seq(unit(tirx.Bind(n, lengths[0])), owner),
        buffers=[x],
        params=[a, c, lengths],
        annotations={"tl.buffer_versions_map": {x.data: 2}, "tl.buffer_version_mode": {x.data: "write_counter"}},
    )
    assert calls(_physical(before), "tl.ascend_wait_flag")


def test_guarded_initialization_keeps_each_owner_footprint_separate():
    a, c = [tirx.decl_buffer((2, 4, 64), "float32", name=name) for name in ("A", "C")]
    x = tirx.decl_buffer((64,), "float32", name="x", scope="shared.dyn")
    tile, i, span = [tirx.Var(name, "int32") for name in ("tile", "i", "span")]
    body = seq(
        unit(copy(a, x, src_indices=[tile, i, 0], src_shape=[1, 1, span], dst_shape=[span]), guard=tirx.all(i >= 1, i % 2 == 1)),
        unit(copy(x, c, src_shape=[span], dst_indices=[tile, i, 0], dst_shape=[1, 1, span]), guard=i >= 1),
    )
    owner = unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [x.data]}), core=None)
    before = kernel(
        unit(tirx.For(tile, 0, 2, tirx.ForKind.SERIAL, seq(unit(tirx.Bind(span, 32 + tile * 16)), owner)), core=None),
        buffers=[x],
        params=[a, c],
        annotations={"tl.buffer_versions_map": {x.data: 2}, "tl.buffer_version_mode": {x.data: "write_counter"}},
    )
    assert calls(_physical(before), "tl.ascend_wait_flag")


def test_stable_partial_value_survives_disjoint_owners():
    a, c = [tirx.decl_buffer((64,), "float32", name=name) for name in ("A", "C")]
    x = tirx.decl_buffer((64,), "float32", name="x", scope="shared.dyn")
    loops = []
    for index, operation in enumerate((copy(a, x, src_shape=[32], dst_shape=[32]), copy(x, c, src_shape=[32], dst_shape=[32]))):
        iv = tirx.Var(f"i{index}", "int32")
        loops.append(
            unit(tirx.For(iv, 0, 2, tirx.ForKind.SERIAL, unit(operation), annotations={"multi_buffer_eligible": [x.data]}), core=None)
        )
    before = kernel(
        seq(*loops),
        buffers=[x],
        params=[a, c],
        annotations={"tl.buffer_versions_map": {x.data: 2}, "tl.buffer_version_mode": {x.data: "write_counter"}},
    )
    _check_all_interleavings(_streams(_physical(before)))


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("versions", [1, 2, 3])
def test_two_consumer_pipes_with_mixed_mode(conditional, versions):
    a, c = [tirx.decl_buffer((8, 64), "float32", name=name) for name in ("A", "C")]
    x, y = [tirx.decl_buffer((64,), "float32", name=name, scope="shared.dyn") for name in ("x", "y")]
    i = tirx.Var("i", "int32")
    guard = i % 2 == 0 if conditional else None
    body = seq(
        unit(copy(a, x, src_indices=[i, 0], src_shape=[1, 64]), guard=i % 2 == 0),
        unit(copy(x, c, dst_indices=[i, 0], dst_shape=[1, 64])),
        unit(copy(x, y), guard=guard),
        unit(copy(y, c, dst_indices=[4 + i, 0], dst_shape=[1, 64]), guard=guard),
    )
    before = kernel(
        unit(tirx.For(i, 0, 3, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [x.data, y.data]}), core=None),
        buffers=[x, y],
        params=[a, c],
        annotations={
            "tl.buffer_versions_map": {x.data: versions, y.data: 2},
            "tl.buffer_version_mode": {x.data: "write_counter", y.data: "counter" if conditional else "iteration"},
        },
    )
    _check_all_interleavings(_streams(_physical(before)))


def test_changing_view_geometry_is_not_a_complete_generation():
    a, c = [tirx.decl_buffer((4, 64), "float32", name=name) for name in ("A", "C")]
    x = tirx.decl_buffer((4, 64), "float32", name="x", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    view = tirx.decl_buffer((1, 64), "float32", name="view", data=x.data, scope="shared.dyn", elem_offset=i * 64)
    body = seq(
        unit(copy(a, view, src_indices=[i, 0], src_shape=[1, 64]), guard=i % 2 == 0),
        unit(copy(view, c, dst_indices=[i, 0], dst_shape=[1, 64])),
    )
    before = kernel(
        unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [x.data]}), core=None),
        buffers=[x],
        params=[a, c],
        annotations={"tl.buffer_versions_map": {x.data: 2}, "tl.buffer_version_mode": {x.data: "write_counter"}},
    )
    with pytest.raises(tvm.error.InternalError, match="loop-invariant buffer view geometry"):
        transform.PrepareMultiBuffer()(before)


@pytest.mark.parametrize("guard_reader", [False, True])
@pytest.mark.parametrize("reader_snapshot", ["shared", "alias", "independent"])
def test_conditional_owner_keeps_initialization_predicate(guard_reader, reader_snapshot):
    a, c = [tirx.decl_buffer((64,), "float32", name=name) for name in ("A", "C")]
    predicate = tirx.decl_buffer((1,), "int32", name="predicate")
    x = tirx.decl_buffer((64,), "float32", name="x", scope="shared.dyn")
    i, j, enabled = [tirx.Var(name, "int32") for name in ("i", "j", "enabled")]
    bindings = [unit(tirx.Bind(enabled, predicate[0]))]
    read_enabled = enabled
    if reader_snapshot != "shared":
        read_enabled = tirx.Var("read_enabled", "int32")
        bindings.append(unit(tirx.Bind(read_enabled, enabled if reader_snapshot == "alias" else predicate[0])))
    write = unit(
        tirx.For(i, 0, 2, tirx.ForKind.SERIAL, unit(copy(a, x))),
        core=None,
        guard=enabled > 0,
    )
    read = unit(
        tirx.For(j, 0, 2, tirx.ForKind.SERIAL, unit(copy(x, c))),
        core=None,
        guard=read_enabled > 0 if guard_reader else None,
    )
    before = kernel(
        seq(*bindings, write, read),
        buffers=[x],
        params=[a, c, predicate],
        annotations={"tl.buffer_versions_map": {x.data: 2}, "tl.buffer_version_mode": {x.data: "write_counter"}},
    )
    before = transform.AnnotateMultiBufferEligible()(before)
    if guard_reader and reader_snapshot != "independent":
        assert calls(_physical(before), "tl.ascend_wait_flag")
    else:
        with pytest.raises(tvm.error.InternalError, match="initialization"):
            transform.PrepareMultiBuffer()(before)


def test_partial_writes_in_one_task_jointly_define_value():
    a, c = [tirx.decl_buffer((64,), "float32", name=name) for name in ("A", "C")]
    x = tirx.decl_buffer((64,), "float32", name="x", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    writer = unit(
        seq(copy(a, x, src_shape=[16], dst_shape=[16]), copy(a, x, src_indices=[16], src_shape=[16], dst_indices=[16], dst_shape=[16])),
        guard=i % 2 == 0,
    )
    body = seq(writer, unit(copy(x, c, src_shape=[32], dst_shape=[32])))
    before = kernel(
        unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [x.data]}), core=None),
        buffers=[x],
        params=[a, c],
        annotations={"tl.buffer_versions_map": {x.data: 2}, "tl.buffer_version_mode": {x.data: "write_counter"}},
    )
    # Eligibility deliberately does not require the union to cover an entire
    # allocation. The Task defines the portion read by this generation.
    assert calls(_physical(before), "tl.ascend_wait_flag")


@pytest.mark.parametrize(
    "modes",
    [
        ("iteration", "write_counter", "counter", "write_counter"),
        ("write_counter", "counter", "write_counter", "iteration"),
        ("write_counter",) * 4,
    ],
)
@pytest.mark.parametrize("versions", [1, 2, 3])
@pytest.mark.parametrize("conditional", [False, True])
def test_chained_channel_lifetimes_with_shared_allocations(modes, versions, conditional):
    a, c = [tirx.decl_buffer((16, 64), "float32", name=name) for name in ("A", "C")]
    buffers, loops = [], []
    for index, mode in enumerate(modes):
        buffer = tirx.decl_buffer((64,), "float32", name=f"buffer{index}", scope="shared.dyn")
        buffers.append(buffer)
        i = tirx.Var(f"i{index}", "int32")
        body = seq(
            unit(
                copy(a, buffer, src_indices=[index * 4 + i, 0], src_shape=[1, 64]),
                guard=i % 2 == 0 if conditional and mode == "write_counter" else None,
            ),
            unit(copy(buffer, c, dst_indices=[index * 4 + i, 0], dst_shape=[1, 64])),
        )
        loops.append(unit(tirx.For(i, 0, 3, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [buffer.data]}), core=None))
    before = kernel(
        seq(*loops),
        buffers=buffers,
        params=[a, c],
        annotations={
            "tl.buffer_versions_map": {buf.data: versions for buf in buffers},
            "tl.buffer_version_mode": {buf.data: mode for buf, mode in zip(buffers, modes)},
        },
    )
    after = _physical(before)
    _check_all_interleavings(_streams(after, coalesce_allocations=True))
    # Ring three can exceed the hardware flag budget. Check that the common
    # spill pass accepts the new clock expressions as well.
    transform.RewriteFlagToBuf()(after)


@pytest.mark.parametrize("versions", [1, 2, 3])
@pytest.mark.parametrize("case", ["guards", "interleaved", "late-guard", "nested", "two-pipes", "read-modify-write"])
def test_one_update_per_owner_iteration(versions, case):
    trips = 4
    a, c = [tirx.decl_buffer((3 * trips, 64), "float32", name=name) for name in ("A", "C")]
    x, y = [tirx.decl_buffer((64,), "float32", name=name, scope="shared.dyn") for name in ("x", "y")]
    i, j, late = tirx.Var("i", "int32"), tirx.Var("j", "int32"), tirx.Var("late", "bool")
    first = unit(copy(a, x, src_indices=[i, 0], src_shape=[1, 64]), guard=i % 2 == 0)
    second_guard = i % 3 == 0
    tasks = [first]
    if case in ("interleaved", "late-guard"):
        tasks += [unit(copy(x, c, dst_indices=[i, 0], dst_shape=[1, 64]))]
    if case == "late-guard":
        tasks += [unit(tirx.Bind(late, second_guard))]
        second_guard = late
    second = unit(copy(a, x, src_indices=[trips + i, 0], src_shape=[1, 64]), guard=second_guard)
    if case == "nested":
        # Zero-trip, skipped and repeated inner writes still form one update.
        nested_write = unit(copy(a, x, src_indices=[trips + i, 0], src_shape=[1, 64]), guard=j % 2 == 0)
        nested_read = unit(copy(x, c, dst_indices=[i, 0], dst_shape=[1, 64]))
        second = unit(tirx.For(j, 0, i % 3, tirx.ForKind.SERIAL, seq(nested_write, nested_read)), core=None)
    elif case == "two-pipes":
        second = unit(copy(y, x), guard=second_guard)
    elif case == "read-modify-write":
        second = unit(copy(x, x), guard=i % 2 == 0)
    tasks += [second, unit(copy(x, c, dst_indices=[2 * trips + i, 0], dst_shape=[1, 64]))]
    before = kernel(
        seq(
            unit(copy(a, y, src_indices=[2 * trips, 0], src_shape=[1, 64])),
            unit(tirx.For(i, 0, trips, tirx.ForKind.SERIAL, seq(*tasks), annotations={"multi_buffer_eligible": [x.data]}), core=None),
        ),
        buffers=[x, y],
        params=[a, c],
        annotations={"tl.buffer_versions_map": {x.data: versions}, "tl.buffer_version_mode": {x.data: "write_counter"}},
    )
    updates = []
    streams = _streams(_physical(before), counter_updates=updates)
    _check_all_interleavings(streams)
    expected = 2 if case == "read-modify-write" else 3
    assert [generation for _, generation in updates] == list(range(expected))


def test_read_modify_write_cannot_start_a_later_update():
    a, c = [tirx.decl_buffer((64,), "float32", name=name) for name in ("A", "C")]
    x = tirx.decl_buffer((64,), "float32", name="x", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    body = seq(unit(copy(a, x), guard=i == 0), unit(copy(x, x)), unit(copy(x, c)))
    before = kernel(
        unit(tirx.For(i, 0, 3, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [x.data]}), core=None),
        buffers=[x],
        params=[a, c],
        annotations={"tl.buffer_versions_map": {x.data: 2}, "tl.buffer_version_mode": {x.data: "write_counter"}},
    )
    with pytest.raises(tvm.error.InternalError, match="start an update with a read-modify-write"):
        transform.PrepareMultiBuffer()(before)


@pytest.mark.parametrize("complete", [False, True])
def test_write_contract_is_not_a_must_write_proof_for_allocation_reuse(complete):
    a = tirx.decl_buffer((32, 16), "float16", name="A")
    out = tirx.decl_buffer((16, 16), "float16", name="out", scope="shared.l0a")
    buffers = [tirx.decl_buffer((32, 16), "float16", name=name, scope="shared.l1") for name in ("x", "y")]
    loops = []
    for index, buffer in enumerate(buffers):
        i = tirx.Var(f"i{index}", "int32")
        body = seq(
            # This declares a larger destination when complete=False. DMA
            # only defines the source's 16 rows; eligibility accepts the
            # storage contract, while reuse needs independent physical proof.
            unit(copy(a, buffer, src_shape=[32 if complete else 16, 16]), core=2),
            unit(copy(buffer, out, src_shape=[16, 16]), core=2),
        )
        loops.append(unit(tirx.For(i, 0, 2, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [buffer.data]}), core=None))
    before = kernel(
        seq(*loops),
        buffers=[*buffers, out],
        params=[a],
        annotations={
            "tl.buffer_versions_map": {b.data: 2 for b in buffers},
            "tl.buffer_version_mode": {b.data: "write_counter" for b in buffers},
        },
    )
    after = _physical(before)
    root = next(block for block in nodes(after, tirx.SBlock) if block.name_hint == "tilelang_root")
    stored = {buffer.name: buffer.data for buffer in root.alloc_buffers}
    aliases = root.annotations["tl.buffer_alias_map"]
    assert any(member.same_as(stored["y"]) for member in aliases[stored["x"]]) == complete


@pytest.mark.parametrize("ordinary_mode", ["iteration", "counter", "auto"])
@pytest.mark.parametrize("versions", [1, 2, 3])
def test_interleaved_update_with_an_ordinary_ring(ordinary_mode, versions):
    before = _program(versions=versions, trips=4, second_write=True, mixed=ordinary_mode)

    def interleave(node):
        if not isinstance(node, tirx.For) or "multi_buffer_eligible" not in node.annotations:
            return None
        body = list(node.body.seq)
        # Original order: old W, new W1, new W2, new R, old R. Read both
        # storages between the two writes, as well as after publication.
        return tirx.For(node.loop_var, node.min, node.extent, node.kind, seq(*body[:2], *body[3:], *body[2:]), annotations=node.annotations)

    before = tvm.IRModule({"main": before["main"].with_body(tirx.stmt_functor.ir_transform(before["main"].body, None, interleave))})
    _check_all_interleavings(_streams(_physical(before), coalesce_allocations=True))


@pytest.mark.parametrize("versions", [1, 2, 3])
def test_all_three_iteration_write_guard_combinations(versions):
    trips = 3
    for first_mask in range(1, 8, 2):
        for second_mask in range(8):
            before = _program(versions=versions, trips=trips, second_write=True)
            i = nodes(before, tirx.For)[0].loop_var
            masks = iter((first_mask, second_mask))

            def replace_guard(node, i=i, masks=masks):
                if isinstance(node, tirx.IfThenElse):
                    mask = next(masks)
                    guard = tirx.any(*(i == j for j in range(trips) if mask & (1 << j))) if mask else tirx.const(False, "bool")
                    return tirx.IfThenElse(guard, node.then_case, None)
                if isinstance(node, tirx.For):
                    # Read between W1 and W2, then read the completed update.
                    body = list(node.body.seq)
                    return tirx.For(
                        node.loop_var,
                        node.min,
                        node.extent,
                        node.kind,
                        seq(body[0], body[2], body[1], body[2]),
                        annotations=node.annotations,
                    )

            before = tvm.IRModule(
                {"main": before["main"].with_body(tirx.stmt_functor.ir_transform(before["main"].body, None, replace_guard))}
            )
            updates = []
            streams = _streams(_physical(before), counter_updates=updates)
            _check_all_interleavings(streams)
            assert len(updates) == (first_mask | second_mask).bit_count()


@pytest.mark.parametrize("initialization", ["none", "separate", "atomic", "clear"])
def test_implicit_read_modify_write_uses_the_same_write_first_analysis(initialization):
    a = tirx.decl_buffer((4, 64), "float32", name="A")
    r = tirx.decl_buffer((4, 64), "float32", name="r", scope="shared.dyn")
    x = tirx.decl_buffer((64,), "float32", name="x", scope="shared.dyn")
    c = tirx.decl_buffer((64,), "float32", name="C")
    i = tirx.Var("i", "int32")
    reduction = tirx.Evaluate(
        tirx.Call(
            "handle",
            tvm.ir.Op.get("tl.tileop.reduce"),
            [region(r, 1), region(x, 2), "sum", 0, tirx.const(initialization == "clear", "bool")],
        )
    )
    fill = tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get("tl.tileop.fill"), [region(x, 2), tirx.const(0, "float32")]))
    clear_reduction = tirx.Evaluate(
        tirx.Call("handle", tvm.ir.Op.get("tl.tileop.reduce"), [region(r, 1), region(x, 2), "sum", 0, tirx.const(True, "bool")])
    )
    tasks = [unit(seq(clear_reduction, reduction))] if initialization == "atomic" else [unit(reduction)]
    if initialization == "separate":
        tasks.insert(0, unit(fill))
    tasks.append(unit(copy(x, c)))
    before = kernel(
        seq(
            unit(copy(a, r)),
            unit(tirx.For(i, 0, 2, tirx.ForKind.SERIAL, seq(*tasks), annotations={"multi_buffer_eligible": [x.data]}), core=None),
        ),
        buffers=[r, x],
        params=[a, c],
        annotations={"tl.buffer_versions_map": {x.data: 2}, "tl.buffer_version_mode": {x.data: "write_counter"}},
    )
    if initialization == "none":
        with pytest.raises(tvm.error.InternalError, match="start an update with a read-modify-write"):
            transform.PrepareMultiBuffer()(before)
    else:
        assert calls(_physical(before), "tl.ascend_wait_flag")
