"""Clock/flag/slot integration on explicitly scheduled DMA programs.

These tests compose Prepare, ResolveCore, InsertSync and Materialize: a legal
flag protocol must protect the physical slot selected by the prepared clock.
The model explores all pipe interleavings, checking values, balance and deadlock.
It treats copies as atomic tiles; partial writes and real overlap need the
numerical tests in test_schedule_integration.py.
"""

from collections import defaultdict

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

from testing.ascend._ir import calls, copy, kernel, nodes, seq, statements
from testing.ascend.auto_schedule._scheduled_ir import unit


def _program(*, versions=2, period=3, trips=7, reads=True, second_write=False, mixed=None, nested=False):
    a = tirx.decl_buffer((2 * max(trips, 1), 64), "float32", name="A")
    c = tirx.decl_buffer((2 * max(trips, 1), 64), "float32", name="C")
    cached = tirx.decl_buffer((64,), "float32", name="cached", scope="shared.dyn")
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    tasks = [unit(copy(a, cached, src_indices=[i, 0], src_shape=[1, 64]), guard=i % period == 0)]
    if second_write:
        tasks.append(unit(copy(a, cached, src_indices=[trips + i, 0], src_shape=[1, 64]), guard=i % 2 == 0))
    read_guard = None if reads is True else i % 2 == 1 if reads else i < 0
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
    """Return physical IR and write-clock storage identities for the model."""
    mod = transform.PrepareMultiBuffer()(before)
    counters = {
        loop.annotations["tl.multi_buffer_counter_map"][storage].data
        for loop in nodes(mod, tirx.For)
        for storage in loop.annotations.get("tl.write_counter_buffers", [])
    }
    mod = transform.ResolveCore()(mod)
    mod = transform.InsertSync()(mod)
    return transform.MaterializeMultiBuffer()(mod), counters


def _streams(mod, counters=(), *, coalesce_allocations=False, counter_updates=None):
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
            if counter_updates is not None and stmt.buffer.data in counters and scalar[stmt.buffer.data] >= 0:
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


@pytest.mark.parametrize(
    "versions, options",
    [
        pytest.param(1, {}, id="single-slot"),
        pytest.param(2, {}, id="ring-wrap"),
        pytest.param(3, {}, id="three-slots"),
        pytest.param(2, {"reads": False}, id="unused-generations"),
        pytest.param(2, {"reads": "conditional"}, id="conditional-consumer"),
        pytest.param(2, {"nested": True}, id="empty-consumer-loop"),
        pytest.param(2, {"trips": 0}, id="empty-owner"),
        pytest.param(2, {"mixed": "iteration"}, id="mixed-iteration"),
        pytest.param(2, {"mixed": "counter"}, id="mixed-counter"),
        pytest.param(2, {"mixed": "auto", "second_write": True}, id="mixed-auto"),
    ],
)
def test_write_generations_and_mixed_clocks(versions, options):
    _check_all_interleavings(_streams(*_physical(_program(versions=versions, **options))))


def test_retained_reads_do_not_repeat_handshakes():
    streams = _streams(*_physical(_program(trips=7, period=3)))
    commands = [command for stream in streams.values() for command in stream]
    # Three writes: READY + FREE, each set/wait once. Two FREE slots are
    # initialized and drained. Reads between writes add no hardware events.
    assert sum(command[0] in ("set", "wait") for command in commands) == 3 * 4 + 2 * 2


@pytest.mark.parametrize(
    "case, bind_guards",
    [
        ("invariant", False),
        ("invariant-snapshot", True),
        ("delayed", True),
        ("periodic", False),
        ("interval", False),
        ("gaps", True),
        ("disjoint", False),
        ("step", False),
        ("multiple-writes", True),
    ],
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
    after, counters = _physical(before)
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
    assert reader_guards
    guarded_syncs = 0
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
            guarded_syncs += 1
            active = tirx.all(*(p.condition for p in parents if isinstance(p, tirx.IfThenElse)))
            assert analyzer.can_prove(tirx.Or(tirx.Not(active), reader_guards[0])), "sync escaped the common guard"
    assert guarded_syncs
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
        streams = _streams(tvm.IRModule({"main": func.with_body(body)}), counters)
        _check_all_interleavings(streams)
        commands = [command for stream in streams.values() for command in stream]
        writes = sum(command[0] == "copy" and command[1][0] == "A" for command in commands)
        if case != "multiple-writes":
            assert sum(command[0] in ("set", "wait") for command in commands) == writes * 4 + 2 * 2


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
    after, counters = _physical(before)
    assert not calls(after, "tl.ascend_set_flag")
    # One guarded barrier before ring wrap, plus a final completion barrier.
    assert len(calls(after, "tl.ascend_pipe_barrier")) == 2
    _check_all_interleavings(_streams(after, counters))


def test_irregular_refresh_is_not_interpreted_as_an_iteration_distance():
    before = _program()
    i = nodes(before, tirx.For)[0].loop_var

    def mutate(node):
        if isinstance(node, tirx.IfThenElse):
            return tirx.IfThenElse(tirx.any(i == 0, i == 1, i == 5), node.then_case, None)

    before = tvm.IRModule({"main": before["main"].with_body(tirx.stmt_functor.ir_transform(before["main"].body, None, mutate))})
    _check_all_interleavings(_streams(*_physical(before)))


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
    _check_all_interleavings(_streams(*_physical(before)))


@pytest.mark.parametrize("second_writes", [False, True], ids=["read-only", "read-then-update"])
def test_disjoint_owners_share_the_last_written_generation(second_writes):
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
    owners = [loop for loop in nodes(before, tirx.For) if cached.data in loop.annotations.get("multi_buffer_eligible", [])]
    assert len(owners) == 2
    updates = []
    streams = _streams(*_physical(before), counter_updates=updates)
    count = 2 + int(second_writes)
    assert [generation for _, generation in updates] == list(range(count))
    _check_all_interleavings(streams)


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
    prepared = transform.PrepareMultiBuffer()(before)
    counters = nodes(prepared, tirx.For)[0].annotations["tl.multi_buffer_counter_map"]
    assert counters[x.data].same_as(counters[y.data])
    after, clocks = _physical(before)
    # Different ring sizes share a clock but need separate READY channels.
    ready = [call for call in calls(after, "tl.ascend_set_flag") if call.args[0].value == "MTE2_MTE3"]
    assert len(ready) == (1 if y_versions == 2 else 2)
    _check_all_interleavings(_streams(after, clocks))


@pytest.mark.parametrize(
    "middle_reads, y_versions",
    [(("x",), 2), (("y",), 2), (("x", "y"), 2), (("x", "y"), 3)],
    ids=["representative", "other-member", "sibling-owners", "unequal-rings"],
)
def test_shared_write_clock_preserves_every_members_intermediate_reads(middle_reads, y_versions):
    a, c = [tirx.decl_buffer((24, 64), "float32", name=name) for name in ("A", "C")]
    x, y = [tirx.decl_buffer((64,), "float32", name=name, scope="shared.dyn") for name in ("x", "y")]
    loops = []
    for index, name in enumerate(middle_reads):
        i = tirx.Var(f"i{index}", "int32")
        middle = x if name == "x" else y
        body = seq(
            unit(
                seq(*(copy(a, buf, src_indices=[12 * index + i, 0], src_shape=[1, 64]) for buf in (x, y))),
                guard=i % 2 == 0,
            ),
            unit(copy(middle, c, dst_indices=[12 * index + i, 0], dst_shape=[1, 64])),
            unit(seq(*(copy(a, buf, src_indices=[12 * index + 4 + i, 0], src_shape=[1, 64]) for buf in (x, y)))),
            unit(copy(x, c, dst_indices=[12 * index + 8 + i, 0], dst_shape=[1, 64])),
            unit(copy(y, c, dst_indices=[12 * index + 8 + i, 0], dst_shape=[1, 64])),
        )
        loops.append(unit(tirx.For(i, 0, 3, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [x.data, y.data]}), core=None))
    before = kernel(
        seq(*loops),
        buffers=[x, y],
        params=[a, c],
        annotations={
            "tl.buffer_versions_map": {x.data: 2, y.data: y_versions},
            "tl.buffer_version_mode": {x.data: "write_counter", y.data: "write_counter"},
        },
    )
    prepared = transform.PrepareMultiBuffer()(before)
    counters = [loop.annotations["tl.multi_buffer_counter_map"] for loop in nodes(prepared, tirx.For)]
    assert all(mapping[x.data].same_as(counters[0][x.data]) and mapping[x.data].same_as(mapping[y.data]) for mapping in counters)
    updates = []
    streams = _streams(*_physical(before), counter_updates=updates)
    assert [generation for _, generation in updates] == list(range(3 * len(middle_reads)))
    _check_all_interleavings(streams)


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
    after, counters = _physical(before)
    _check_all_interleavings(_streams(after, counters))
    ready = [call for call in calls(after, "tl.ascend_set_flag") if call.args[0].value == "MTE2_MTE3"]
    assert len(ready) == 1, "the ordinary RAW must use the write clock's completion certificate"


@pytest.mark.parametrize("conditional, versions", [(False, 1), (False, 2), (True, 3)])
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
    _check_all_interleavings(_streams(*_physical(before)))


def _disjoint_protocols(modes, versions, *, guarded=False):
    a, c = [tirx.decl_buffer((16, 64), "float32", name=name) for name in ("A", "C")]
    buffers, loops = [], []
    for index, mode in enumerate(modes):
        buffer = tirx.decl_buffer((64,), "float32", name=f"buffer{index}", scope="shared.dyn")
        buffers.append(buffer)
        i = tirx.Var(f"i{index}", "int32")
        body = seq(
            unit(
                copy(a, buffer, src_indices=[index * 4 + i, 0], src_shape=[1, 64]),
                guard=i % 2 == 0 if guarded and mode == "write_counter" else None,
            ),
            unit(copy(buffer, c, dst_indices=[index * 4 + i, 0], dst_shape=[1, 64])),
        )
        loops.append(unit(tirx.For(i, 0, 4, tirx.ForKind.SERIAL, body, annotations={"multi_buffer_eligible": [buffer.data]}), core=None))
    before = kernel(
        seq(*loops),
        buffers=buffers,
        params=[a, c],
        annotations={
            "tl.buffer_versions_map": {buf.data: versions for buf in buffers},
            "tl.buffer_version_mode": {buf.data: mode for buf, mode in zip(buffers, modes)},
        },
    )
    return before, buffers


@pytest.mark.parametrize(
    "modes, versions",
    [
        pytest.param(("write_counter",) * 3, 2, id="write-only"),
        pytest.param(("iteration", "write_counter"), 2, id="iteration-before-write"),
        pytest.param(("write_counter", "iteration"), 2, id="write-before-iteration"),
        pytest.param(("counter", "write_counter"), 2, id="counter-before-write"),
        pytest.param(("write_counter", "counter"), 2, id="write-before-counter"),
    ],
)
def test_disjoint_protocols_reuse_flags_and_allocations(modes, versions):
    before, buffers = _disjoint_protocols(modes, versions)
    after, counters = _physical(before)
    _check_all_interleavings(_streams(after, counters, coalesce_allocations=True))
    ready = {
        command[1][2]
        for commands in _streams(after, counters).values()
        for command in commands
        if command[0] == "set" and command[1][:2] == ("MTE2", "MTE3")
    }
    assert len(ready) == versions
    root = after["main"].body.block
    aliases = root.annotations["tl.buffer_alias_map"]
    stored = {buffer.name: buffer.data for buffer in root.alloc_buffers}
    assert all(stored[other.name] in aliases[stored[buffer.name]] for buffer in buffers for other in buffers if not buffer.same_as(other))


@pytest.mark.parametrize(
    "case, versions",
    [("guards", 2), ("interleaved", 1), ("late-guard", 2), ("nested", 3), ("two-pipes", 2), ("read-modify-write", 2)],
)
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
    streams = _streams(*_physical(before), counter_updates=updates)
    _check_all_interleavings(streams)
    expected = 2 if case == "read-modify-write" else 3
    assert [generation for _, generation in updates] == list(range(expected))


@pytest.mark.parametrize("ordinary_mode, versions", [("iteration", 1), ("counter", 2), ("auto", 3)])
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
    _check_all_interleavings(_streams(*_physical(before), coalesce_allocations=True))


def test_guarded_protocols_drain_before_allocation_reuse_and_flag_spill():
    before, _ = _disjoint_protocols(("iteration", "write_counter", "counter", "write_counter"), 3, guarded=True)
    after, counters = _physical(before)
    _check_all_interleavings(_streams(after, counters, coalesce_allocations=True))
    spilled = transform.RewriteFlagToBuf()(after)
    assert calls(spilled, "tl.ascend_get_buf") and calls(spilled, "tl.ascend_rls_buf")
