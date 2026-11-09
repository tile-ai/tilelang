"""Conservative Ascend schedules for canonical graph reductions and GEMMs.

These rules consume unscheduled TIR produced by Relax legalization.  Pattern
recognition proves the indexing contract before replacing loops with Ascend
tile operations; an unmatched function is left to the graph fallback path.
"""

import math

from tilelang import tvm
from tilelang.ascend import language as T
from tvm import s_tir, tirx


_FLOAT_DTYPES = {"float16", "bfloat16", "float32"}


def _nodes(node, kind):
    found = []
    tirx.stmt_functor.post_order_visit(node, lambda value: found.append(value) if isinstance(value, kind) else None)
    return found


def _same(a, b):
    return a.same_as(b)


def _equal(a, b):
    return tvm.arith.Analyzer().can_prove_equal(a, b)


def _shape(buffer):
    if buffer.scope() != "global" or str(buffer.dtype) not in _FLOAT_DTYPES:
        return None
    if not all(isinstance(value, tirx.IntImm) and int(value) > 0 for value in buffer.shape):
        return None
    shape = tuple(int(value) for value in buffer.shape)
    if buffer.axis_separators or math.prod(shape) > 2**31 - 1:
        return None
    if not isinstance(buffer.elem_offset, tirx.IntImm) or int(buffer.elem_offset) != 0:
        return None
    if buffer.strides:
        stride = 1
        for extent, actual in zip(reversed(shape), reversed(buffer.strides)):
            if not _equal(actual, stride):
                return None
            stride *= extent
    return shape


def _strip_cast(value):
    while isinstance(value, tirx.Cast) and str(value.dtype) in _FLOAT_DTYPES:
        value = value.value
    return value


def _load(value, buffer, indices=None):
    value = _strip_cast(value)
    return (
        isinstance(value, tirx.BufferLoad)
        and _same(value.buffer, buffer)
        and (indices is None or (len(indices) == len(value.indices) and all(_equal(a, b) for a, b in zip(value.indices, indices))))
    )


def _accumulator_load(value, buffer, indices):
    return isinstance(value, tirx.BufferLoad) and _load(value, buffer, indices)


def _input_load(value, buffer, indices):
    while isinstance(value, tirx.Cast):
        if str(value.dtype) not in (str(buffer.dtype), "float32"):
            return False
        value = value.value
    return _load(value, buffer, indices)


def _number(value):
    value = _strip_cast(value)
    return float(value.value) if isinstance(value, (tirx.FloatImm, tirx.IntImm)) else None


def _complete_regions(func, output):
    """Prove realizations cover their declared domains before dropping loops."""
    domains = {}
    parameters = set(func.buffer_map.values())
    output_writers = 0

    def visit(stmt, in_block=False):
        nonlocal output_writers
        if isinstance(stmt, tirx.For):
            if stmt.kind != tirx.ForKind.SERIAL or (stmt.step is not None and not _equal(stmt.step, 1)):
                return False
            domains[stmt.loop_var] = (stmt.min, stmt.extent)
            valid = visit(stmt.body, in_block)
            del domains[stmt.loop_var]
            return valid
        if isinstance(stmt, tirx.SeqStmt):
            return all(visit(child, in_block) for child in stmt.seq)
        if isinstance(stmt, tirx.SBlockRealize):
            if not _equal(stmt.predicate, 1) or len(stmt.iter_values) != len(stmt.block.iter_vars):
                return False
            seen = set()
            for axis, value in zip(stmt.block.iter_vars, stmt.iter_values):
                if isinstance(value, tirx.IntImm) and _equal(axis.dom.extent, 1) and _equal(value, axis.dom.min):
                    continue
                if not isinstance(value, tirx.Var) or value not in domains or value in seen:
                    return False
                minimum, extent = domains[value]
                if not _equal(minimum, axis.dom.min) or not _equal(extent, axis.dom.extent):
                    return False
                seen.add(value)
            return visit(stmt.block, in_block)
        if isinstance(stmt, tirx.SBlock):
            if stmt.iter_vars:
                if not isinstance(stmt.body, tirx.BufferStore):
                    return False
                target = stmt.body.buffer
                if target in parameters and not _same(target, output):
                    return False
                output_writers += int(_same(target, output))
            elif stmt.name_hint != "root":
                return False
            return visit(stmt.body, bool(stmt.iter_vars)) and (stmt.init is None or visit(stmt.init, bool(stmt.iter_vars)))
        return isinstance(stmt, tirx.BufferStore) and in_block

    return visit(func.body) and output_writers == 1


def _prepare(func):
    """Inline only complete injective intermediates, keeping the reduction."""
    if any(param not in func.buffer_map for param in func.params) or len(func.params) < 2:
        return None
    if any(_shape(func.buffer_map[param]) is None for param in func.params):
        return None
    output = func.buffer_map[func.params[-1]]
    if not _complete_regions(func, output):
        return None
    blocks = [block for block in _nodes(func.body, tirx.SBlock) if block.iter_vars]
    reductions = [block for block in blocks if any(it.iter_type == tirx.IterVar.CommReduce for it in block.iter_vars)]
    if len(reductions) != 1:
        return None
    if any(not isinstance(block.body, tirx.BufferStore) or len(block.writes) != 1 for block in blocks):
        return None
    expected_stores = len(blocks) + sum(block.init is not None for block in blocks)
    if len(_nodes(func.body, tirx.BufferStore)) != expected_stores:
        return None
    if any(not _equal(realize.predicate, 1) for realize in _nodes(func.body, tirx.SBlockRealize)):
        return None
    if any(loop.kind != tirx.ForKind.SERIAL or not _equal(loop.min, 0) for loop in _nodes(func.body, tirx.For)):
        return None
    if _nodes(func.body, (tirx.IfThenElse, tirx.While, tirx.AttrStmt, tirx.Evaluate)):
        return None
    try:
        schedule = s_tir.Schedule(func)
        for block in blocks:
            if _same(block, reductions[0]) or _same(block.body.buffer, output):
                continue
            schedule.compute_inline(schedule.get_sblock(block.name_hint))
        func = schedule.mod["main"]
    except (tvm.error.TVMError, ValueError):
        return None
    blocks = [block for block in _nodes(func.body, tirx.SBlock) if block.iter_vars]
    reductions = [block for block in blocks if any(it.iter_type == tirx.IterVar.CommReduce for it in block.iter_vars)]
    if len(reductions) != 1 or len(blocks) > 2:
        return None
    return func, blocks, reductions[0]


def _finish(kernel, original, rule):
    symbol = original.attrs.get("global_symbol", "main") if original.attrs is not None else "main"
    return kernel.with_attr({"global_symbol": symbol, "tirx.noalias": True, "tirx.is_scheduled": True, "tl.graph_schedule": rule})


def _complete_output(block, shape):
    if len(block.iter_vars) != len(shape) or len(block.body.indices) != len(shape):
        return False
    return all(
        axis.iter_type == tirx.IterVar.DataPar and _equal(axis.dom.min, 0) and _equal(axis.dom.extent, extent) and _equal(index, axis.var)
        for axis, extent, index in zip(block.iter_vars, shape, block.body.indices)
    )


def _settings(config, rows):
    cores = int(getattr(config, "num_cores", 32))
    threads = int(getattr(config, "threads", 128))
    if cores <= 0 or threads <= 0 or threads > 1024:
        raise ValueError("Ascend graph schedules require positive num_cores and 1 <= threads <= 1024")
    return min(cores, rows), threads


def try_schedule_reduction(func, config=None):
    """Schedule one canonical last-axis sum/max, optionally followed by mean.

    Input and output may have arbitrary static ranks, but the reduction must
    cover the complete last axis and leave every preceding axis unchanged.
    Float16 and bfloat16 inputs are accumulated in float32. Max also tracks
    NaNs and all-negative-infinity rows explicitly instead of relying on
    target-specific physical identities and NaN handling.
    """
    prepared = _prepare(func)
    if prepared is None:
        return None
    normalized, blocks, block = prepared
    if len(normalized.params) != 2:
        return None
    source, output = (normalized.buffer_map[param] for param in normalized.params)
    source_shape, output_shape = _shape(source), _shape(output)
    if not source_shape or output_shape is None:
        return None
    extent = source_shape[-1]
    if extent > 16384:
        return None
    rows = math.prod(source_shape[:-1])
    if output_shape not in (source_shape[:-1], (*source_shape[:-1], 1)):
        return None
    spatial = [it for it in block.iter_vars if it.iter_type == tirx.IterVar.DataPar]
    reduction = [it for it in block.iter_vars if it.iter_type == tirx.IterVar.CommReduce]
    prefix_rank = len(source_shape) - 1
    if len(reduction) != 1 or len(spatial) not in (prefix_rank, prefix_rank + 1):
        return None
    if len(spatial) == prefix_rank + 1 and (not _equal(spatial[-1].dom.min, 0) or not _equal(spatial[-1].dom.extent, 1)):
        return None
    expected = [it.var for it in spatial[:prefix_rank]] + [reduction[0].var]
    if any(
        not _equal(it.dom.min, 0) or not _equal(it.dom.extent, size)
        for it, size in zip([*spatial[:prefix_rank], reduction[0]], source_shape)
    ):
        return None
    update, initial = block.body, block.init
    if (
        not isinstance(initial, tirx.BufferStore)
        or not _same(initial.buffer, update.buffer)
        or len(initial.indices) != len(update.indices)
        or not all(_equal(a, b) for a, b in zip(initial.indices, update.indices))
    ):
        return None
    value = update.value
    if isinstance(value, tirx.Add):
        op = "sum"
        if _number(initial.value) != 0:
            return None
    elif isinstance(value, tirx.Max):
        op = "max"
        identity = _number(initial.value)
        if identity not in (_number(tirx.min_value(update.buffer.dtype)), -math.inf):
            return None
    else:
        return None
    if not (
        (_accumulator_load(value.a, update.buffer, update.indices) and _input_load(value.b, source, expected))
        or (_accumulator_load(value.b, update.buffer, update.indices) and _input_load(value.a, source, expected))
    ):
        return None
    expected_output = [it.var for it in spatial]
    if len(update.indices) == len(expected_output) + 1:
        expected_output += [0]
    if len(update.indices) != len(expected_output) or not all(_equal(a, b) for a, b in zip(update.indices, expected_output)):
        return None
    scale = 1.0
    scalar = tirx.Var("graph_reduction_result", str(update.buffer.dtype))
    post_template = tirx.Cast(output.dtype, scalar)
    if not _same(update.buffer, output):
        finals = [candidate for candidate in blocks if _same(candidate.body.buffer, output)]
        if len(finals) != 1:
            return None
        if not _complete_output(finals[0], output_shape):
            return None
        final = finals[0].body
        post = _strip_cast(final.value)
        if _load(post, update.buffer):
            pass
        elif op == "sum" and isinstance(post, tirx.Div) and _load(post.a, update.buffer) and _number(post.b) == extent:
            scale = 1.0 / extent
        elif op == "sum" and isinstance(post, tirx.Mul):
            factor = post.b if _load(post.a, update.buffer) else post.a if _load(post.b, update.buffer) else None
            if factor is None or _number(factor) != _number(tirx.const(1.0 / extent, factor.dtype)):
                return None
            scale = 1.0 / extent
        else:
            return None
        # A post-reduction block may only read the corresponding result, not
        # permute rows or broadcast one row to all output elements.
        loads = _nodes(final.value, tirx.BufferLoad)
        if (
            len(loads) != 1
            or len(loads[0].indices) != len(final.indices)
            or not all(_equal(a, b) for a, b in zip(loads[0].indices, final.indices))
        ):
            return None
        post_template = tirx.stmt_functor.ir_transform(
            tirx.Evaluate(final.value), None, lambda node: scalar if isinstance(node, tirx.BufferLoad) else None
        ).value
    cores, threads = _settings(config, rows)
    threads = min(threads, 1 << (extent - 1).bit_length())
    in_dtype, out_dtype = str(source.dtype), str(output.dtype)
    from .elementwise import legalize_scalar_arithmetic

    post_template = legalize_scalar_arithmetic(post_template)

    def apply_post(value):
        return tirx.stmt_functor.substitute(post_template, {scalar: tirx.Cast(scalar.dtype, value)})

    @T.prim_func
    def graph_reduce(X: T.Tensor(source_shape, in_dtype), Y: T.Tensor(output_shape, out_dtype)):
        with T.Kernel(cores) as bx:
            x = T.reshape(X, (rows, extent))
            y = T.reshape(Y, (rows,))
            for tile in T.serial(T.ceildiv(rows, cores)):
                row = tile * cores + bx
                if row < rows:
                    with T.SimtVF(threads=threads):
                        partial = T.alloc_reducer((1,), "float32", op=op)
                        result = T.alloc_fragment((1,), "float32")
                        T.reducer_init(partial)
                        if op == "max":
                            nan_count = T.alloc_reducer((1,), "float32", op="sum")
                            valid_count = T.alloc_reducer((1,), "float32", op="sum")
                            nan_result = T.alloc_fragment((1,), "float32")
                            valid_result = T.alloc_fragment((1,), "float32")
                            T.reducer_init(nan_count)
                            T.reducer_init(valid_count)
                        for k in T.Parallel(extent):
                            value = T.Cast("float32", x[row, k])
                            T.reducer_update(partial[0], value)
                            if op == "max":
                                T.reducer_update(nan_count[0], T.if_then_else(T.isnan(value), T.float32(1), T.float32(0)))
                                T.reducer_update(valid_count[0], T.if_then_else(value > T.float32(-math.inf), T.float32(1), T.float32(0)))
                        T.finalize_reducer(partial, result)
                        if op == "max":
                            T.finalize_reducer(nan_count, nan_result)
                            T.finalize_reducer(valid_count, valid_result)
                        for i in T.Parallel(1):
                            if op == "max":
                                y[row] = apply_post(
                                    T.if_then_else(
                                        nan_result[i] > 0,
                                        T.float32(math.nan),
                                        T.if_then_else(valid_result[i] > 0, result[i], T.float32(-math.inf)),
                                    )
                                )
                            else:
                                y[row] = apply_post(result[i])

    return _finish(graph_reduce, func, "last_axis_" + ("mean" if scale != 1 else op))


@tirx.functor.mutator
class _Epilogue(tirx.PyStmtExprMutator):
    def __init__(self, accumulator, extra_buffers, m, n):
        super().__init__()
        self.accumulator = accumulator
        self.extra_buffers = extra_buffers
        self.m, self.n = m, n
        self.value = tirx.Var("graph_accumulator", "float32")

    def visit_buffer_load_(self, node):
        if _same(node.buffer, self.accumulator):
            if len(node.indices) != 2 or not _equal(node.indices[0], self.m) or not _equal(node.indices[1], self.n):
                raise ValueError("GEMM epilogue reads another output tile")
            return self.value
        if not any(_same(node.buffer, buffer) for buffer in self.extra_buffers):
            raise ValueError("GEMM epilogue reads an unsupported producer")
        shape = _shape(node.buffer)
        if not shape or len(shape) > 2:
            raise ValueError("GEMM epilogue supports only static row-major bias tensors")
        expected = [self.n] if len(shape) == 1 else [self.m, self.n]
        expected = [0 if extent == 1 else index for extent, index in zip(shape, expected)]
        if len(node.indices) != len(expected) or not all(_equal(a, b) for a, b in zip(node.indices, expected)):
            raise ValueError("GEMM epilogue bias uses a non-broadcast index map")
        return node


def try_schedule_gemm(func, config=None):
    """Schedule a canonical 2-D GEMM and a local bias/relu/cast epilogue.

    Both logical NN and NT right operands are supported. NN is explicitly
    transposed by GM-to-L1 dn2nz, then the existing L1 NT GEMM lowering is used.
    Tiles are full, aligned rectangles; nonaligned shapes are not padded by
    this rule and must use the graph fallback.
    """
    prepared = _prepare(func)
    if prepared is None:
        return None
    normalized, blocks, block = prepared
    if len(normalized.params) < 3:
        return None
    buffers = [normalized.buffer_map[param] for param in normalized.params]
    a, b, output = buffers[0], buffers[1], buffers[-1]
    a_shape, b_shape, out_shape = _shape(a), _shape(b), _shape(output)
    if len(a_shape) != 2 or len(b_shape) != 2 or len(out_shape) != 2 or a.dtype != b.dtype or str(a.dtype) not in ("float16", "bfloat16"):
        return None
    m_size, k_size = a_shape
    if out_shape[0] != m_size:
        return None
    n_size = out_shape[1]
    if any(size % 16 for size in (m_size, n_size, k_size)):
        return None
    spatial = [it for it in block.iter_vars if it.iter_type == tirx.IterVar.DataPar]
    reduction = [it for it in block.iter_vars if it.iter_type == tirx.IterVar.CommReduce]
    if len(spatial) != 2 or len(reduction) != 1:
        return None
    m, n, k = spatial[0].var, spatial[1].var, reduction[0].var
    if any(
        not _equal(it.dom.min, 0) or not _equal(it.dom.extent, size) for it, size in zip([*spatial, reduction[0]], (m_size, n_size, k_size))
    ):
        return None
    store, initial = block.body, block.init
    if (
        not isinstance(initial, tirx.BufferStore)
        or not _same(initial.buffer, store.buffer)
        or len(initial.indices) != len(store.indices)
        or not all(_equal(a, b) for a, b in zip(initial.indices, store.indices))
        or _number(initial.value) != 0
        or len(store.indices) != 2
        or not _equal(store.indices[0], m)
        or not _equal(store.indices[1], n)
        or not isinstance(store.value, tirx.Add)
    ):
        return None
    update = store.value
    if _accumulator_load(update.a, store.buffer, store.indices):
        product = update.b
    elif _accumulator_load(update.b, store.buffer, store.indices):
        product = update.a
    else:
        return None
    while isinstance(product, tirx.Cast) and str(product.dtype) == "float32":
        product = product.value
    if not isinstance(product, tirx.Mul) or str(product.dtype) != "float32" or str(store.buffer.dtype) != "float32":
        return None
    lhs, rhs = product.a, product.b
    if not _input_load(lhs, a, [m, k]):
        lhs, rhs = rhs, lhs
    if not _input_load(lhs, a, [m, k]):
        return None
    if _input_load(rhs, b, [k, n]) and b_shape == (k_size, n_size):
        transpose_b = True
    elif _input_load(rhs, b, [n, k]) and b_shape == (n_size, k_size):
        transpose_b = False
    else:
        return None
    epilogue = None
    epilogue_vars = None
    if not _same(store.buffer, output):
        finals = [candidate for candidate in blocks if _same(candidate.body.buffer, output)]
        if len(finals) != 1:
            return None
        final = finals[0]
        if not _complete_output(final, out_shape):
            return None
        em, en = final.body.indices
        if not isinstance(em, tirx.Var) or not isinstance(en, tirx.Var):
            return None
        allowed = (tirx.BufferLoad, tirx.Var, tirx.IntImm, tirx.FloatImm, tirx.Cast, tirx.Add, tirx.Max)
        if any(not isinstance(value, allowed) for value in _nodes(final.body.value, tvm.ir.PrimExpr)):
            return None
        rewriter = _Epilogue(store.buffer, buffers[2:-1], em, en)
        try:
            epilogue = rewriter.visit_expr(final.body.value)
        except (ValueError, tvm.error.TVMError):
            return None
        epilogue_vars = (rewriter.value, em, en)
    elif len(buffers) != 3:
        return None
    tile_m = next(tile for tile in (64, 32, 16) if m_size % tile == 0)
    tile_n = next(tile for tile in (64, 32, 16) if n_size % tile == 0)
    tile_k = next(tile for tile in (64, 32, 16) if k_size % tile == 0)
    tiles_n = n_size // tile_n
    tiles = (m_size // tile_m) * tiles_n
    cores, threads = _settings(config, tiles)
    dtype, out_dtype = str(a.dtype), str(output.dtype)
    mixed = epilogue is not None or out_dtype != "float32"

    def apply_epilogue(value, row, col):
        from .elementwise import legalize_scalar_arithmetic

        if epilogue is None:
            return tirx.Cast(out_dtype, value)
        scalar, em, en = epilogue_vars
        expression = tirx.stmt_functor.substitute(epilogue, {scalar: value, em: tirx.Cast(em.dtype, row), en: tirx.Cast(en.dtype, col)})
        return legalize_scalar_arithmetic(expression)

    @T.prim_func
    def graph_gemm(A: T.Tensor(a_shape, dtype), B: T.Tensor(b_shape, dtype), C: T.Tensor(out_shape, out_dtype)):
        with T.MixedKernel(cores, sids=1) as (bx, sid):
            a_l1 = T.alloc_l1((tile_m, tile_k), dtype)
            b_l1 = T.alloc_l1((tile_n, tile_k), dtype)
            accumulator = T.alloc_l0c((tile_m, tile_n), "float32")
            if mixed:
                accumulator_ub = T.alloc_shared((tile_m, tile_n), "float32")
            for tile in T.serial(T.ceildiv(tiles, cores)):
                tile_id = tile * cores + bx
                if tile_id < tiles:
                    m_start = tile_id // tiles_n * tile_m
                    n_start = tile_id % tiles_n * tile_n
                    for kt in T.Pipelined(k_size // tile_k, num_stages=2):
                        T.copy(A[m_start : m_start + tile_m, kt * tile_k : (kt + 1) * tile_k], a_l1)
                        if transpose_b:
                            T.copy(B[kt * tile_k : (kt + 1) * tile_k, n_start : n_start + tile_n], b_l1, transpose=True)
                        else:
                            T.copy(B[n_start : n_start + tile_n, kt * tile_k : (kt + 1) * tile_k], b_l1)
                        T.gemm(a_l1, b_l1, accumulator, transpose_B=True, clear_accum=kt == 0)
                    if mixed:
                        T.copy(accumulator, accumulator_ub, sub_blockid=0)
                        with T.SimtVF(threads=threads):
                            for i, j in T.Parallel(tile_m, tile_n):
                                C[m_start + i, n_start + j] = apply_epilogue(accumulator_ub[i, j], m_start + i, n_start + j)
                    else:
                        T.copy(accumulator, C[m_start : m_start + tile_m, n_start : n_start + tile_n])

    if len(buffers) > 3:
        # Epilogue bias loads refer to the original buffer handles captured in
        # apply_epilogue. Keep their original ABI positions between A/B and C.
        params = [*graph_gemm.params[:2], *normalized.params[2:-1], graph_gemm.params[-1]]
        buffer_map = dict(graph_gemm.buffer_map)
        buffer_map.update({param: normalized.buffer_map[param] for param in normalized.params[2:-1]})
        graph_gemm = tirx.PrimFunc(params, graph_gemm.body, graph_gemm.ret_type, buffer_map, graph_gemm.attrs)
    return _finish(graph_gemm, func, "gemm_epilogue" if epilogue is not None else "gemm")
