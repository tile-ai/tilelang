"""Ownership fast path: static UB/fragment Parallel and native fallbacks."""

from itertools import product

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import nodes, kernel, seq, copy, region


def _buffer(name, shape=(32, 32), dtype="float16", scope="shared.dyn", **kwargs):
    return tirx.decl_buffer(shape, dtype, name=name, scope=scope, **kwargs)


def _vf(body, threads=128, tx=None, annotations=None):
    if tx is None:
        tx = tirx.IterVar(tvm.ir.Range(0, threads), tirx.Var("simtvf_tx", "int32"), tirx.IterVar.ThreadIndex, "threadIdx.x")
    block = tirx.SBlock([], [], [], "SIMT_VF", tirx.AttrStmt(tx, "thread_extent", threads, body), annotations=annotations or {})
    return tirx.SBlockRealize([], True, block)


def _parallel(shape, body, annotations=None):
    indices = [tirx.Var(f"i{k}", "int32") for k in range(len(shape))]
    stmt = body(*indices)
    for index, extent in reversed(list(zip(indices, shape))):
        stmt = tirx.For(index, 0, extent, tirx.ForKind.PARALLEL, stmt, annotations=annotations or {})
    return stmt


def _infer(mod):
    with mod["main"].attrs["target"]:
        return transform.AscendLayoutInference()(mod)


def _layouts(mod):
    return [n.annotations["parallel_loop_layout"] for n in nodes(mod, tirx.For) if "parallel_loop_layout" in n.annotations]


def _assert_ownership(layout, n, threads, width):
    analyzer = tvm.arith.Analyzer()
    i = tirx.Var("index", "int32")
    analyzer.bind(i, tvm.ir.Range(0, n))
    indices, remaining = [], i
    for extent in reversed(layout.get_input_shape()):
        indices.insert(0, remaining % extent)
        remaining = remaining // extent
    assert analyzer.can_prove_equal(layout.map_forward_thread(indices)[0], (i // width) % threads)


def _assert_partition(layout, threads):
    analyzer = tvm.arith.Analyzer()
    assignments = set()
    for indices in product(*(range(int(n)) for n in layout.get_input_shape())):
        thread = int(analyzer.simplify(layout.map_forward_thread(list(indices))[0]))
        slots = tuple(int(analyzer.simplify(x)) for x in layout.map_forward_index(list(indices)))
        assert 0 <= thread < threads
        assert (thread, slots) not in assignments
        assignments.add((thread, slots))


def _check(mod, width, n=1024, threads=128, *, lower=False):
    inferred = _infer(mod)
    layout = _layouts(inferred)[0]
    if width is None:
        _assert_partition(layout, threads)
    else:
        _assert_ownership(layout, n, threads, width)
    if lower:
        with inferred["main"].attrs["target"]:
            return transform.AscendLowerTileOp()(inferred)


def _elementwise(n=8192, threads=2048, *, constraint=None, dtype="float32", alignments=(0, 0, 0)):
    a, b, c = [_buffer(name, (n,), dtype, data_alignment=alignment) for name, alignment in zip("abc", alignments)]
    annotations = {"coalesced_width": tirx.IntImm("int32", 4)} if constraint == "width" else {}
    loop = _parallel((n,), lambda i: tirx.BufferStore(c, a[i] + b[i], [i]), annotations)
    layout = {"layout_map": {a.data: tilelang.layout.Layout((n,), lambda i: i)}} if constraint == "layout" else {}
    return kernel(_vf(loop, threads), buffers=[a, b, c], annotations=layout)


@pytest.mark.parametrize("buffer_index", [0, 2], ids=["load", "store"])
@pytest.mark.parametrize("alignment", [1, 2, 4, 8, 32, 64])
def test_ownership_base_alignment(buffer_index, alignment):
    alignments = [32, 32, 32]
    alignments[buffer_index] = alignment
    # One weakly aligned access retains native half width=4; an 8B guarantee suffices.
    _check(_elementwise(dtype="float16", alignments=alignments), 2 if alignment >= 8 else 4, 8192, 2048)


@pytest.mark.parametrize("constraint", ["width", "layout"])
def test_ownership_preserves_constraints(constraint):
    # The constrained half layout must differ from the model's width=2 choice.
    _check(_elementwise(constraint=constraint, dtype="float16", alignments=(32, 32, 32)), 4, 8192, 2048)


@pytest.mark.parametrize("width", [1, 4])
def test_ownership_preserves_explicit_loop_partition(width):
    n, threads = 1024, 128
    a, b = [_buffer(name, (n,), data_alignment=32) for name in "ab"]
    layout = tilelang.layout.Fragment(
        (n,),
        forward_fn=lambda i: ((i // width) % threads, (i // (width * threads)) * width + i % width),
    )
    loop = _parallel((n,), lambda i: tirx.BufferStore(b, a[i], [i]), {"parallel_loop_layout": layout})
    mod = kernel(_vf(loop, threads), buffers=[a, b])
    _check(mod, width, n, threads, lower=True)
    # Searching another kernel must not retain a trial as an annotation.
    _check(_elementwise(n, threads, dtype="float16", alignments=(32, 32, 32)), 2, n, threads)
    _check(mod, width, n, threads, lower=True)


@pytest.mark.parametrize(
    "dtype,n,threads,width",
    [
        (dtype, 8192, 128, width)
        for dtype, width in [("int8", 4), ("float16", 2), ("bfloat16", 2), ("int16", 2), ("float32", 1), ("int32", 1), ("int64", 1)]
    ]
    + [("int8", 73728, 128, 4)]  # Exercise repeated warp patterns in a large domain.
    + [("float32", n, threads, None) for n, threads in [(258, 128), (8191, 2048), (256, 16), (130, 33)]],
)
def test_ownership_dtype_and_partition(dtype, n, threads, width):
    _check(_elementwise(n, threads, dtype=dtype), width, n, threads, lower=True)


@pytest.mark.parametrize("dependent", [False, True])
def test_ownership_does_not_change_conflicting_accesses(dependent):
    a = tirx.decl_buffer((512,), "float32", name="a", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    # Either overlapping stores or a loop-carried read/write dependence.
    body = tirx.BufferStore(a, a[i + 1] if dependent else tirx.Cast("float32", i), [i if dependent else i % 128])
    mod = kernel(_vf(tirx.For(i, 0, 256, tirx.ForKind.PARALLEL, body), 32), buffers=[a])
    _assert_ownership(_layouts(_infer(mod))[0], 256, 32, 1 if dependent else 2)


@pytest.mark.parametrize("boundary", ["collective", "copy", "escape", "thread_value", "thread_guard", "inner_if"])
def test_ownership_component_boundaries(boundary):
    left, right = _elementwise(dtype="float16"), _elementwise(dtype="float16")
    buffers = [b for mod in (left, right) for block in nodes(mod, tirx.SBlock) for b in block.alloc_buffers]
    a, b, c = buffers[:3]
    tx = nodes(left, tirx.AttrStmt)[0].node
    loop, sibling = nodes(left, tirx.For)[0], nodes(right, tirx.For)[0]
    suffix = []
    if boundary == "collective":
        values, sums = _buffer("values", (8192,), "float32", "local.fragment"), _buffer("sums", (1,), "float32", "local.fragment")
        buffers += [values, sums]
        reduce = tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get("tl.tileop.reduce"), [region(values, 1), region(sums, 2), "sum", 0, True]))
        suffix = [copy(c, values), reduce]
    elif boundary == "copy":
        suffix = [copy(a, c)]
    elif boundary == "inner_if":
        loop = tirx.For(loop.loop_var, 0, 8192, tirx.ForKind.PARALLEL, tirx.IfThenElse(loop.loop_var < 4096, loop.body, None))
    elif boundary == "escape":
        suffix = [tirx.Evaluate(tirx.call_extern("int32", "escape", a.data))]
    elif boundary == "thread_value":
        loop = _parallel((8192,), lambda i: tirx.BufferStore(c, a[i] + b[i] + tirx.Cast("float16", tx.var), [i]))
    else:
        loop = tirx.IfThenElse(tx.var < 1024, loop, None)
    on = _infer(kernel(_vf(seq(loop, *suffix, sibling), 2048, tx), buffers=buffers))
    _assert_ownership(_layouts(on)[0], 8192, 2048, 4)
    _assert_ownership(_layouts(on)[-1], 8192, 2048, 2)


@pytest.mark.parametrize(
    "strides,dtype,offset,width",
    [(stride, dtype, 0, None) for stride in [(32, 1), (256, 1), (64, 2), (256, 2)] for dtype in ["int8", "float16", "float32", "int64"]]
    + [((256, 1), "float16", offset, 4 if offset == 1 else 1) for offset in [0, 1, 4, 256]]
    + [((1, 32), "float32", 0, 8)],
)
def test_ownership_physical_accesses(strides, dtype, offset, width):
    a, c = [_buffer(name, dtype=dtype, strides=strides, elem_offset=offset) for name in "ac"]
    body = _parallel((32, 32), lambda i, j: tirx.BufferStore(c, a[i, j] + tirx.const(1, dtype), [i, j]))
    lowered = _check(kernel(_vf(body), buffers=[a, c]), width, lower=True)
    # For padded half rows, widths 1/2 have equal entry service; width 1 is
    # the first improving trial. More instructions need not repeat the whole
    # completion latency. Odd offsets have unknown partial-store cost.
    # Transposed float accesses can group beyond vectorization width=1.
    if width == 8:
        assert not any(loop.kind == tirx.ForKind.VECTORIZED for loop in nodes(lowered, tirx.For))


@pytest.mark.parametrize("threads", [128, 2048])
def test_independent_float_work_is_not_charged_serial_instruction_latency(threads):
    # Two b32 accesses publish fewer entries than one b64 for equal work.
    # This also covers the add2048 regression: independent warps and elements
    # can overlap their responses even though the instruction count increases.
    _check(_elementwise(8192, threads, alignments=(32, 32, 32)), 1, 8192, threads, lower=True)


@pytest.mark.parametrize("transpose", [False, True])
def test_ownership_candidate_propagates_through_fragments(transpose):
    a = _buffer("a")
    f = _buffer("f", scope="local.fragment")
    c = _buffer("c")
    producer = _parallel((32, 32), lambda i, j: tirx.BufferStore(f, a[i, j], [i, j]))
    consumer = _parallel((32, 32), lambda i, j: tirx.BufferStore(c, f[j, i] if transpose else f[i, j], [i, j]))
    on = _infer(kernel(_vf(seq(producer, consumer)), buffers=[a, f, c]))
    layouts = _layouts(on)
    assert len(layouts) == 2
    analyzer = tvm.arith.Analyzer()
    for i in range(32):
        for j in range(32):
            producer_thread = layouts[0].map_forward_thread([j, i] if transpose else [i, j])[0]
            consumer_thread = layouts[1].map_forward_thread([i, j])[0]
            assert analyzer.can_prove_equal(producer_thread, consumer_thread)
    with on["main"].attrs["target"]:
        transform.AscendLowerTileOp()(on)


@pytest.mark.parametrize(
    "control,thread_dependent",
    [(control, True) for control in ["then", "else", "nested", "bind", "block", "while", "serial_extent"]]
    + [(control, False) for control in ["then", "bind", "serial", "block_index"]],
)
def test_ownership_enclosing_control(control, thread_dependent):
    mod = _elementwise(n=1024, threads=128, dtype="float16")
    tx = nodes(mod, tirx.AttrStmt)[0].node
    flag, counter = tirx.Var("flag", "int32"), tirx.Var("counter", "int32")
    condition = tx.var < 64 if thread_dependent else flag > 0
    loop = nodes(mod, tirx.For)[0]
    body = tirx.IfThenElse(condition, loop, None)
    if control == "else":
        body = tirx.IfThenElse(tirx.Not(condition), tirx.Evaluate(0), loop)
    elif control == "nested":
        body = tirx.IfThenElse(condition, body, None)
    elif control == "bind":
        active = tirx.Var("active", "bool")
        body = seq(tirx.Bind(active, condition), tirx.IfThenElse(active, loop, None))
    elif control == "block":
        body = tirx.SBlockRealize([], condition, tirx.SBlock([], [], [], "conditional", loop))
    elif control == "while":
        body = tirx.While(condition, loop)
    elif control == "serial_extent":
        body = tirx.For(counter, 0, tirx.if_then_else(condition, 1, 0), tirx.ForKind.SERIAL, loop)
    elif control in ("serial", "block_index"):
        body = tirx.IfThenElse(condition if thread_dependent else counter == 0, loop, None)
        bx = tirx.IterVar(tvm.ir.Range(0, 2), counter, tirx.IterVar.ThreadIndex, "blockIdx.x")
        body = tirx.For(counter, 0, 2, tirx.ForKind.SERIAL, body) if control == "serial" else tirx.AttrStmt(bx, "thread_extent", 2, body)
    buffers = nodes(mod, tirx.SBlock)[-1].alloc_buffers
    _check(kernel(_vf(body, tx=tx), buffers=buffers, params=[flag]), 4, 1024)


@pytest.mark.parametrize("direction", ["load", "store"])
@pytest.mark.parametrize("escapes", [False, True])
def test_ownership_propagates_through_copy_components(direction, escapes):
    a, c = [_buffer(name, (1024,), data_alignment=32) for name in "ac"]
    f1, f2 = [_buffer(name, (1024,), scope="local.fragment") for name in ["f1", "f2"]]
    if direction == "load":
        compute = _parallel((1024,), lambda i: tirx.BufferStore(c, a[i] + f1[i], [i]))
        body = [copy(a, f2), copy(f2, f1), compute]
    else:
        compute = _parallel((1024,), lambda i: tirx.BufferStore(f1, a[i] + 1, [i]))
        body = [compute, copy(f1, f2), copy(f2, c)]
    if escapes:
        body.append(tirx.Evaluate(tirx.call_extern("int32", "consume_fragment", f2.data)))
    mod = kernel(_vf(seq(*body)), buffers=[a, c, f1, f2])
    # Full-tile copies participate in fragment propagation and UB scoring.
    # An external fragment consumer still prevents changing the component.
    _check(mod, 4 if escapes else 2, lower=True)


@pytest.mark.parametrize("kind", ["partial", "thread_offset", "annotation", "conditional", "global"])
def test_ownership_unsupported_copy_retains_native(kind):
    a, c = [_buffer(name, (1024,), data_alignment=32) for name in "ac"]
    f = _buffer("f", (1024,), scope="local.fragment")
    tx = tirx.IterVar(tvm.ir.Range(0, 128), tirx.Var("simtvf_tx", "int32"), tirx.IterVar.ThreadIndex, "threadIdx.x")
    source = _buffer("source", (2048 if kind == "partial" else 1024,), scope="global" if kind == "global" else "shared.dyn")
    kwargs = {"src_shape": [1024]} if kind == "partial" else {}
    if kind == "thread_offset":
        kwargs["src_indices"] = [tx.var % 2]
    if kind == "annotation":
        kwargs["annotations"] = {"coalesced_width": tirx.IntImm("int32", 4)}
    load = copy(source, f, **kwargs)
    if kind == "conditional":
        load = tirx.IfThenElse(tx.var < 64, load, None)
    compute = _parallel((1024,), lambda i: tirx.BufferStore(c, a[i] + f[i], [i]))
    mod = kernel(_vf(seq(load, compute), tx=tx), buffers=[a, c, f, source])
    # The native thread-offset copy scalarizes; the other fixtures use width 4.
    _check(mod, 1 if kind == "thread_offset" else 4)


@pytest.mark.parametrize("metadata", ["range", "elem_offset", "strides", "shape"])
@pytest.mark.parametrize("parameter", [False, True])
def test_ownership_dma_uniform_arguments(metadata, parameter):
    n = 1024
    offset = tirx.Var("offset", "int32")
    kwargs = {"elem_offset": offset} if metadata == "elem_offset" else {"strides": [offset + 1]} if metadata == "strides" else {}
    g = _buffer("g", (n + offset if metadata == "shape" else n,), scope="global", **kwargs)
    a, c = [_buffer(name, (n,), data_alignment=32) for name in "ac"]
    compute = _parallel((n,), lambda i: tirx.BufferStore(c, a[i] + 1, [i]))
    dma = copy(g, a, src_indices=[offset if metadata == "range" else 0], src_shape=[n])
    body = seq(dma, _vf(compute))
    params = [g, offset] if parameter else [g]
    if not parameter:
        # A local binding is outside the supported uniform-argument contract.
        body = seq(tirx.Bind(offset, 0), body)
    _check(kernel(body, buffers=[a, c], params=params), 2 if parameter else 4)


@pytest.mark.parametrize("direction", ["load", "store"])
@pytest.mark.parametrize("metadata", ["elem_offset", "strides", "shape"])
@pytest.mark.parametrize("thread_dependent", [False, True])
def test_ownership_parallel_buffer_metadata(direction, metadata, thread_dependent):
    n, threads = 1024, 128
    tx = tirx.IterVar(tvm.ir.Range(0, threads), tirx.Var("simtvf_tx", "int32"), tirx.IterVar.ThreadIndex, "threadIdx.x")
    uniform = tirx.Var("uniform", "int32")
    owner = tx.var if thread_dependent else uniform
    kwargs = {"elem_offset": owner * n} if metadata == "elem_offset" else {"strides": (n * (owner + 1), 1)} if metadata == "strides" else {}
    shape = (2, n * (owner + 1)) if metadata == "shape" else (2, n)
    g = _buffer("g", shape, scope="global", **kwargs)
    a, c = [_buffer(name, (n,), data_alignment=32) for name in "ac"]
    loop = _parallel(
        (n,),
        lambda i: (
            tirx.BufferStore(c, a[i] + g[1, i], [i])
            if direction == "load"
            else seq(tirx.BufferStore(c, a[i], [i]), tirx.BufferStore(g, a[i], [1, i]))
        ),
    )
    mod = kernel(_vf(loop, threads, tx), buffers=[a, c], params=[uniform, g])
    # Global accesses are outside this fast path, regardless of whether
    # their metadata is thread-dependent or uniform.
    _check(mod, 4, n, threads, lower=True)


def _executed_store_indices(mod, name):
    """Enumerate actual lowered control flow, including padded-lane guards."""
    analyzer = tvm.arith.Analyzer()
    found = []

    def value(expr, bindings):
        return int(analyzer.simplify(tirx.stmt_functor.substitute(expr, bindings)))

    def visit(stmt, bindings):
        if isinstance(stmt, tirx.SBlockRealize):
            if value(stmt.predicate, bindings):
                visit(stmt.block, bindings)
        elif isinstance(stmt, tirx.SBlock):
            visit(stmt.body, bindings)
        elif isinstance(stmt, tirx.AttrStmt):
            if stmt.attr_key == "thread_extent":
                for thread in range(value(stmt.value, bindings)):
                    visit(stmt.body, {**bindings, stmt.node.var: thread})
            else:
                visit(stmt.body, bindings)
        elif isinstance(stmt, tirx.For):
            start = value(stmt.min, bindings)
            for i in range(start, start + value(stmt.extent, bindings)):
                visit(stmt.body, {**bindings, stmt.loop_var: i})
        elif isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                visit(child, bindings)
        elif isinstance(stmt, tirx.IfThenElse):
            child = stmt.then_case if value(stmt.condition, bindings) else stmt.else_case
            if child is not None:
                visit(child, bindings)
        elif isinstance(stmt, tirx.BufferStore) and str(stmt.buffer.name) == name:
            found.append(tuple(value(index, bindings) for index in stmt.indices))

    visit(mod["main"].body, {})
    return found


@pytest.mark.parametrize("n,threads", [(258, 128), (8191, 2048), (256, 16), (130, 33)])
def test_ownership_lowered_tail_coverage(n, threads):
    lowered = _check(_elementwise(n, threads), None, n, threads, lower=True)
    assert sorted(_executed_store_indices(lowered, "c")) == [(i,) for i in range(n)]


@pytest.mark.parametrize("kind", [tirx.ForKind.SERIAL, tirx.ForKind.UNROLLED])
@pytest.mark.parametrize("copies,start,width", [(1, 0, 8), (4, 3, 4), (32, 0, 2)])
def test_ownership_enclosing_copy_multiplicity(kind, copies, start, width):
    a = _buffer("a", strides=(1, 32))
    f = _buffer("f", scope="local.fragment")
    c = _buffer("c")
    compute = _parallel((32, 32), lambda i, j: tirx.BufferStore(f, a[i, j] + 1, [i, j]))
    q = tirx.Var("q", "int32")
    repeated = tirx.For(q, start, copies, kind, copy(f, c))
    actual = _infer(kernel(_vf(seq(compute, repeated)), buffers=[a, f, c]))
    explicit = _infer(kernel(_vf(seq(compute, *(copy(f, c) for _ in range(copies)))), buffers=[a, f, c]))
    # Increasing stores changes the best producer layout: ignoring the outer
    # execution count would choose width 8 for every instance.
    for mod in (actual, explicit):
        _assert_ownership(_layouts(mod)[0], 1024, 128, width)
    with actual["main"].attrs["target"]:
        transform.AscendLowerTileOp()(actual)


def test_ownership_nested_loop_counts():
    a = _buffer("a", strides=(1, 32))
    f = _buffer("f", scope="local.fragment")
    c = _buffer("c")
    compute = _parallel((32, 32), lambda i, j: tirx.BufferStore(f, a[i, j] + 1, [i, j]))
    q, r = tirx.Var("q", "int32"), tirx.Var("r", "int32")
    # Six stores, not three: equal access maps do not imply equal inner bounds.
    repeated = tirx.For(q, 0, 3, tirx.ForKind.SERIAL, tirx.For(r, 0, q + 1, tirx.ForKind.SERIAL, copy(f, c)))
    _check(kernel(_vf(seq(compute, repeated)), buffers=[a, f, c]), 4, lower=True)


def test_ownership_overflowing_serial_invocation_count_retains_native():
    a, c = [_buffer(name, (1024,), data_alignment=32) for name in "ac"]
    compute = _parallel((1024,), lambda i: tirx.BufferStore(c, a[i], [i]))
    q, r = tirx.Var("q", "int64"), tirx.Var("r", "int64")
    extent = tirx.IntImm("int64", 1 << 40)
    repeated = tirx.For(q, 0, extent, tirx.ForKind.SERIAL, tirx.For(r, 0, extent, tirx.ForKind.SERIAL, compute))
    _check(kernel(_vf(repeated), buffers=[a, c]), 4)


@pytest.mark.parametrize("kind", ["fixed", "shifted", "dynamic", "thread_bound", "zero"])
def test_ownership_outer_loop_contract(kind):
    a, c = [_buffer(name, (1024,), data_alignment=32) for name in "ac"]
    q, n = tirx.Var("q", "int32"), tirx.Var("n", "int32")
    tx = tirx.IterVar(tvm.ir.Range(0, 128), tirx.Var("tx", "int32"), tirx.IterVar.ThreadIndex, "threadIdx.x")

    def index(i):
        return (i + q) % 1024 if kind == "shifted" else i

    compute = _parallel((1024,), lambda i: tirx.BufferStore(c, a[index(i)] + q, [index(i)]))
    extent = n if kind == "dynamic" else tx.var % 2 + 1 if kind == "thread_bound" else 0 if kind == "zero" else 3
    body = tirx.For(q, 0, extent, tirx.ForKind.SERIAL, compute)
    # Uniform scalar values may vary; SIMT access maps must stay invariant.
    # Unknown counts and zero work cannot justify replacing native ownership.
    _check(kernel(_vf(body, tx=tx), buffers=[a, c], params=[n]), 2 if kind == "fixed" else 1 if kind == "shifted" else 4)
