"""Carver's analysis and bounded graph transforms over native TIRx Functions.

Stage/loop/tensor views are Python analysis records, not compiler objects. Their
underlying expressions remain native, and committed transforms rebuild a real,
verified Function. No legacy Schedule, Buffer or global registry is recreated.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace

import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.ir import prim

from tilelang._analyzer import create_analyzer
from tilelang.analysis._visitor import tilelang_metadata
from tilelang.language._native import tensor_offset, tensor_strides
from ._workload import IterAxis, Stage, emit_function, substitute, variables


def post_order_visit(value, fvisit):
    tvm_ffi.structural_walk(value, (object, lambda node: fvisit(node) if isinstance(node, ir.Node) else None), order="post")


def pre_order_visit(value, fvisit):
    tvm_ffi.structural_walk(value, (object, lambda node: fvisit(node) if isinstance(node, ir.Node) else None), order="pre")


def _statements(value):
    return list(value.seq) if isinstance(value, ir.SeqStmt) else [value]


def _one_statement(value):
    while isinstance(value, ir.SeqStmt) and len(value.seq) == 1:
        value = value.seq[0]
    return value


def _nodes(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="post")
    return result


def _stage_nodes(stage, kind):
    return [node for value in (stage.body, stage.init, stage.predicate) if value is not None for node in _nodes(value, kind)]


def _same(left, right):
    return tvm_ffi.structural_equal(left, right)


def specialize_shapes(function, replacements):
    """Specialize shape scalars and rebuild typed definitions and their uses.

    Remove specialized scalar parameters before mapping: a parameter slot
    remains a Var, while expression uses may become constants. Native
    definition hooks renew tensor Vars when their TensorType changes.
    """
    params = [param for param in function.params if param not in replacements]
    seed = tirx.Function(params, function.body, function.ret_type, attrs=function.attrs, loc=function.loc)
    analyzer = create_analyzer()

    def rewrite(value):
        if isinstance(value, ir.Var):
            return replacements.get(value, value)
        if isinstance(value, ir.Expr) and ir.is_prim_expr(value):
            return analyzer.simplify(value)
        return value

    mapped = tvm_ffi.structural_map(seed, (object, rewrite))
    result = tirx.Function(mapped.params, mapped.body, mapped.ret_type, attrs=mapped.attrs, loc=mapped.loc)
    tirx.analysis.check_well_formed(result)
    return result


@dataclass(frozen=True)
class StageRef:
    owner: int
    uid: int
    revision: int


@dataclass(frozen=True)
class LoopRef:
    owner: int
    stage_uid: int
    axis: int
    revision: int


@dataclass(frozen=True, eq=False)
class TensorView:
    tensor: ir.Var
    name: str

    def __hash__(self):
        return hash(self.tensor)

    def __eq__(self, other):
        return isinstance(other, TensorView) and self.tensor.same_as(other.tensor)

    @property
    def shape(self):
        return tuple(self.tensor.ty.shape)

    @property
    def dtype(self):
        return str(self.tensor.ty.dtype.dtype)

    def offset_of(self, indices):
        offset = tensor_offset(self.tensor)
        for index, stride in zip(indices, tensor_strides(self.tensor)):
            offset = offset + index * stride
        return [offset]


@dataclass(frozen=True)
class TensorAccess:
    tensor: TensorView
    region: tuple[ir.Range, ...]


@dataclass(frozen=True)
class LoopView:
    var: ir.Var
    begin: ir.Expr
    end: ir.Expr
    step: ir.Expr

    @property
    def extent(self):
        """Iteration count for the increasing loops accepted by Carver."""
        analyzer = create_analyzer()
        if not analyzer.can_prove(self.step > 0):
            if analyzer.can_prove(self.begin >= self.end):
                return ir.const(0, self.var.ty.dtype)
            raise ValueError("Carver requires a provably increasing finite loop")
        return analyzer.simplify(prim.max(prim.ceildiv(self.end - self.begin, self.step), 0))


@dataclass(frozen=True)
class StageView:
    spec: Stage
    reads: tuple[TensorAccess, ...]
    writes: tuple[TensorAccess, ...]

    @property
    def name_hint(self):
        return self.spec.name

    @property
    def axes(self):
        return self.spec.axes

    @property
    def body(self):
        return _one_statement(self.spec.body)

    @property
    def init(self):
        return self.spec.init


class NativeSchedule:
    """The native subset of stage graph operations used by Carver itself.

    Handles are tied to this owner and revision. Transform clients explicitly
    refresh surviving handles. Failed transactions retain the original IR and
    graph, so a failed tensor-core candidate cannot poison later fallbacks.
    """

    def __init__(self, function):
        if isinstance(function, ir.IRModule):
            if len(function.functions) != 1:
                raise ValueError("Carver analysis requires exactly one function")
            function = next(iter(function.functions.values()))
        if not isinstance(function, tirx.Function):
            raise TypeError("Carver requires a native tirx.Function")
        if function.attrs is not None and function.attrs.get("declaration", False):
            raise ValueError("Carver requires a function definition")
        self.function = function
        self.revision = 0
        self._owner = id(self)
        self._stages = []
        self._prefix = []
        self._rebuildable = True
        self._normalizable = True
        self._next_uid = 0
        self._tensor_names = {}
        self._transaction_depth = 0
        self._cache_tensors = set()
        self._loop_stage_owners = {}
        self._parse(function.body)
        if not self._stages:
            raise ValueError("Carver requires native computation stages (T.sblock/T.block)")
        self._refresh_views()

    @property
    def mod(self):
        return ir.IRModule({"main": self.function})

    @property
    def root(self):
        return StageRef(self._owner, -1, self.revision)

    @property
    def parameter_tensors(self):
        return [self.tensor_view(parameter) for parameter in self.function.params if tirx.is_tensor_var(parameter)]

    def _ref(self, uid):
        return StageRef(self._owner, uid, self.revision)

    def _check(self, ref):
        if ref.owner != self._owner:
            raise ValueError("A Carver handle belongs to a different Function analysis")
        if ref.revision != self.revision:
            raise ValueError("A stale Carver handle must be explicitly refreshed")

    def refresh(self, ref):
        if ref.owner != self._owner:
            raise ValueError("Cannot refresh a foreign Carver handle")
        uid = ref.uid if isinstance(ref, StageRef) else ref.stage_uid
        if uid != -1 and uid not in self._views:
            raise ValueError("Cannot refresh a deleted Carver stage")
        return replace(ref, revision=self.revision)

    def tensor_view(self, tensor):
        if not tirx.is_tensor_var(tensor):
            raise ValueError("Carver tensor analysis requires a native tensor variable")
        if tensor not in self._tensor_names:
            used = set(self._tensor_names.values())
            base = tensor.name
            name, suffix = base, 1
            while name in used:
                name = f"{base}_{suffix}"
                suffix += 1
            self._tensor_names[tensor] = name
        return TensorView(tensor, self._tensor_names[tensor])

    def _read_stage(self, region, loops, outer_predicate):
        metadata = tilelang_metadata(region)
        statements = _statements(region.body)
        bindings = {stmt.var: stmt.value for stmt in statements if isinstance(stmt, ir.Bind)}
        axes = []
        for loop in loops:
            if loop in self._loop_stage_owners:
                # Stages sharing an enclosing loop may communicate through a
                # reused tile. Emitting one complete nest per stage would
                # reorder that communication, so do not rebuild that graph.
                self._rebuildable = False
            self._loop_stage_owners[loop] = self._next_uid
        for stmt in statements:
            if not isinstance(stmt, ir.Evaluate):
                continue
            values = tilelang_metadata(stmt.value)
            if values is not None and "tl.axis" in values:
                info = values["tl.axis"]
                axis = IterAxis(info["var"], ir.Range.from_min_extent(info["begin"], info["extent"]), str(info["kind"]))
                axes.append(axis)
        if loops or axes:
            if len(loops) != len(axes):
                self._normalizable = False
                self._rebuildable = False
            else:
                analyzer = create_analyzer()
                for axis, loop in zip(axes, loops):
                    bound = bindings.get(axis.var, axis.var)
                    if not isinstance(bound, ir.Var) or not bound.same_as(loop.var):
                        self._normalizable = False
                        self._rebuildable = False
                    # Axis metadata describes the block's declared domain. A
                    # containing loop may traverse only part of that domain;
                    # re-emitting the declared bounds would add executions.
                    declared_end = axis.domain.begin + axis.domain.extent
                    if not analyzer.can_prove(loop.begin == axis.domain.begin) or not analyzer.can_prove(loop.end == declared_end):
                        self._normalizable = False
                        self._rebuildable = False
                    if not isinstance(loop.begin, ir.IntImm) or int(loop.begin) != 0:
                        self._normalizable = False
                axes = [replace(axis, step=loop.step) for axis, loop in zip(axes, loops)]
        replacements = {loop.var: axis.var for loop, axis in zip(loops, axes)}
        axis_vars = {axis.var for axis in axes}
        body, init = [], []
        for stmt in statements:
            if isinstance(stmt, ir.Bind) and stmt.var in axis_vars:
                continue
            if isinstance(stmt, ir.Evaluate):
                values = tilelang_metadata(stmt.value)
                if values is not None and all(key in {"tl.axis", "tl.reads", "tl.writes"} for key in values):
                    continue
            init_regions = [
                node
                for node in _nodes(stmt, ir.RegionStmt)
                if (info := tilelang_metadata(node)) is not None and info.get("key") == "tl.sblock.init"
            ]
            if init_regions:
                # Preserve the full original guard; re-emission additionally
                # enforces the first-reduction-coordinate rule.
                init.append(substitute(stmt, replacements))
            else:
                body.append(substitute(stmt, replacements))
        name = str(metadata["value"].get("name", "stage"))
        stage = Stage(
            self._next_uid,
            name,
            tuple(axes),
            ir.SeqStmt(body),
            ir.SeqStmt(init) if init else None,
            substitute(outer_predicate, replacements) if outer_predicate is not None else None,
            init_guarded=bool(init),
        )
        self._next_uid += 1
        self._stages.append(stage)

    def _parse(self, statement, loops=(), predicate=None):
        if isinstance(statement, ir.SeqStmt):
            for child in statement.seq:
                self._parse(child, loops, predicate)
        elif isinstance(statement, ir.For):
            if statement.kind != ir.ForKind.DEFAULT:
                # Carver's emitted stage nests are sequential. A parallel loop
                # can carry placement and inter-thread communication semantics;
                # analysis may inspect it, but a graph rewrite must retain it.
                self._normalizable = False
                self._rebuildable = False
            self._parse(statement.body, (*loops, statement), predicate)
        elif isinstance(statement, ir.If):
            condition = statement.condition if predicate is None else prim.And(predicate, statement.condition)
            self._parse(statement.then_case, loops, condition)
            if statement.else_case is not None:
                inverse = prim.Not(statement.condition)
                self._parse(statement.else_case, loops, inverse if predicate is None else prim.And(predicate, inverse))
        elif isinstance(statement, ir.RegionStmt):
            meta = tilelang_metadata(statement)
            if meta is None or meta.get("key") != "tl.sblock":
                self._rebuildable = False
                self._parse(statement.body, loops, predicate)
                return
            nested = [
                node
                for node in _nodes(statement.body, ir.RegionStmt)
                if (values := tilelang_metadata(node)) is not None and values.get("key") == "tl.sblock"
            ]
            if nested:
                self._parse(statement.body, loops, predicate)
            else:
                self._read_stage(statement, loops, predicate)
        elif isinstance(statement, ir.Bind) and not loops and predicate is None:
            # Prefix binds are emitted before every stage. Once execution of a
            # stage has begun, even an immutable scalar bind may capture a load
            # or call whose value depends on the preceding stores. Keep such
            # functions intact instead of hoisting the binding during rebuild.
            if self._stages:
                self._rebuildable = False
                self._normalizable = False
            self._prefix.append(statement)
        elif isinstance(statement, ir.Evaluate) and isinstance(statement.value, ir.IntImm):
            pass
        else:
            self._rebuildable = False

    def _refresh_views(self):
        for parameter in self.function.params:
            if tirx.is_tensor_var(parameter):
                self.tensor_view(parameter)
        views = {}
        for stage in self._stages:
            stores = _nodes(stage.body, ir.TensorStore)
            destinations = {store.dest for store in stores}

            def accesses(nodes, field, *, init=stage.init, destinations=destinations):
                result, seen = [], set()
                for node in nodes:
                    tensor = getattr(node, field)
                    # The accumulator is not an input port of its reduction.
                    if field == "source" and init is not None and tensor in destinations:
                        continue
                    key = (tensor, tuple(ir.save_json(index) for index in node.indices))
                    if key in seen:
                        continue
                    seen.add(key)
                    result.append(
                        TensorAccess(self.tensor_view(tensor), tuple(ir.Range.from_min_extent(index, 1) for index in node.indices))
                    )
                return tuple(result)

            views[stage.uid] = StageView(stage, accesses(_stage_nodes(stage, ir.TensorLoad), "source"), accesses(stores, "dest"))
        self._views = views

    def get(self, ref):
        self._check(ref)
        if isinstance(ref, StageRef):
            if ref.uid == -1:
                raise ValueError("The root is a stage collection, not a computation stage")
            return self._views[ref.uid]
        axis = self._views[ref.stage_uid].axes[ref.axis]
        step = axis.step if axis.step is not None else ir.const(1, axis.var.ty.dtype)
        return LoopView(axis.var, axis.domain.begin, axis.domain.begin + axis.domain.extent, step)

    def get_sblock(self, name):
        matches = [stage for stage in self._stages if stage.name == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one stage named {name!r}, found {len(matches)}")
        return self._ref(matches[0].uid)

    def get_child_blocks(self, root):
        self._check(root)
        if root.uid != -1:
            return []
        return [self._ref(stage.uid) for stage in self._stages]

    def get_loops(self, block):
        stage = self.get(block)
        return [LoopRef(self._owner, block.uid, index, self.revision) for index in range(len(stage.axes))]

    def get_output_blocks(self, root):
        params = set(self.parameter_tensors)
        return [block for block in self.get_child_blocks(root) if any(write.tensor in params for write in self.get(block).writes)]

    def get_producers(self, block):
        reads = {read.tensor for read in self.get(block).reads}
        return [
            self._ref(stage.uid)
            for stage in self._stages
            if stage.uid != block.uid and any(write.tensor in reads for write in self._views[stage.uid].writes)
        ]

    def get_consumers(self, block):
        writes = {write.tensor for write in self.get(block).writes}
        return [
            self._ref(stage.uid)
            for stage in self._stages
            if stage.uid != block.uid and any(read.tensor in writes for read in self._views[stage.uid].reads)
        ]

    def _commit(self):
        if not self._rebuildable:
            raise ValueError("This Function contains statements outside Carver's stage-transform contract")
        self.function = emit_function(
            self.function.params, self._stages, attrs=self.function.attrs, prefix=self._prefix, loc=self.function.loc
        )
        if self._transaction_depth == 0:
            self.revision += 1
        self._refresh_views()

    @contextmanager
    def transaction(self):
        saved = (
            self.function,
            self.revision,
            list(self._stages),
            list(self._prefix),
            dict(self._tensor_names),
            self._next_uid,
            set(self._cache_tensors),
        )
        self._transaction_depth += 1
        try:
            yield self
        except BaseException:
            self.function, self.revision, self._stages, self._prefix, self._tensor_names, self._next_uid, self._cache_tensors = saved
            self._refresh_views()
            raise
        else:
            if self._transaction_depth == 1 and not self.function.same_as(saved[0]):
                self.revision += 1
        finally:
            self._transaction_depth -= 1

    def normalize(self):
        if not self._normalizable or not self._rebuildable:
            return None
        new_stages = []
        changed = False
        for stage in self._stages:
            replacements = {}
            axes = []
            for axis in stage.axes:
                if axis.step is not None and (not isinstance(axis.step, ir.IntImm) or int(axis.step) != 1):
                    # Native For uses an unconditional < comparison. A negative
                    # step is not a Python range, and its span is not a trip count.
                    return None
                if isinstance(axis.domain.extent, ir.IntImm) and int(axis.domain.extent) == 1:
                    replacements[axis.var] = axis.domain.begin
                else:
                    axes.append(axis)
            if not any(axis.kind == "S" for axis in axes):
                dtype = stage.axes[0].var.ty.dtype if stage.axes else "int64"
                axes.insert(0, IterAxis(ir.Var("unit", ir.PrimType(dtype)), ir.Range(ir.const(0, dtype), ir.const(1, dtype)), "S"))
            transformed = stage.substitute(replacements) if replacements else stage
            transformed = replace(transformed, axes=tuple(axes))
            new_stages.append(transformed)
            changed |= bool(replacements) or tuple(axes) != stage.axes
        if changed:
            with self.transaction():
                self._stages = new_stages
                self._commit()
        return self.get_child_blocks(self.root)

    def _transformable(self, stage):
        if not self._rebuildable or not self._normalizable:
            raise ValueError("Carver transforms require a complete, directly bound stage nest")
        if any(axis.step is not None and (not isinstance(axis.step, ir.IntImm) or int(axis.step) != 1) for axis in stage.axes):
            raise ValueError("Carver stage transforms require unit-step loops")

    @staticmethod
    def _single_store(stage):
        body = _one_statement(stage.body)
        if not isinstance(body, ir.TensorStore):
            raise ValueError("Carver transform requires a single tensor store")
        return body

    @staticmethod
    def _map_stage(stage, callbacks):
        def rewrite(value):
            return tvm_ffi.structural_map(value, callbacks) if value is not None else None

        return replace(stage, body=rewrite(stage.body), init=rewrite(stage.init), predicate=rewrite(stage.predicate))

    def _private_tensor(self, tensor):
        if tensor in self.function.params:
            raise ValueError("Carver cannot remove or change a public tensor parameter")
        # The graph transforms operate on owning tensors, not pointer/view
        # aliases. Aliased storage needs a separate native memory analysis.
        if any(tensor in variables(stmt.value) for stmt in self._prefix if isinstance(stmt, ir.Bind)):
            raise ValueError("Carver cannot transform a tensor with a pointer/view alias")

    def _drop_dead_allocation(self, tensor):
        self._prefix = [
            stmt
            for stmt in self._prefix
            if not (
                isinstance(stmt, ir.Bind)
                and stmt.var.same_as(tensor)
                and isinstance(stmt.value, ir.Call)
                and isinstance(stmt.value.op, ir.Op)
                and stmt.value.op.name == "tirx.alloc_tensor"
            )
        ]

    @staticmethod
    def _pure_value(value):
        # Primitive expression operators are native nodes. A Call can have
        # arbitrary effects; permit only the lazy conditional used in padding.
        return all(isinstance(call.op, ir.Op) and call.op.name == "prim.if_then_else" for call in _nodes(value, ir.Call))

    def _check_noalias(self):
        if not self.function.attrs.get("tirx.noalias", False):
            raise ValueError("Carver graph rewrites require the Function's explicit tirx.noalias contract")
        # noalias separates public arguments; it does not separate local views
        # derived from one argument. Tensor identity is the access model below,
        # so only fresh owning local tensors can participate in these rewrites.
        for binding in _nodes(self.function.body, ir.Bind):
            if not tirx.is_tensor_var(binding.var):
                continue
            value = binding.value
            if not isinstance(value, ir.Call) or not isinstance(value.op, ir.Op) or value.op.name != "tirx.alloc_tensor":
                raise ValueError("Carver graph rewrites cannot treat local tensor views or aliases as disjoint storage")
        # Store/load nodes describe all modeled memory effects. A call in an
        # intervening stage can mutate the same memory without a TensorStore,
        # invalidating the snapshot and reindex checks just as an alias can.
        for stage in self._stages:
            if any(value is not None and not self._pure_value(value) for value in (stage.body, stage.init, stage.predicate)):
                raise ValueError("Carver graph rewrites cannot move memory accesses across calls with unmodeled effects")

    def compute_inline(self, block):
        """Substitute a private injective stage into its later consumers."""
        stage = self.get(block).spec
        self._transformable(stage)
        self._check_noalias()
        store = self._single_store(stage)
        self._private_tensor(store.dest)
        if stage.init is not None or stage.predicate is not None or any(axis.kind != "S" for axis in stage.axes):
            raise ValueError("Only an unguarded spatial producer can be inlined")
        if not self._pure_value(store.value):
            raise ValueError("Cannot duplicate an effectful producer")
        producers = [item for item in self._stages if any(node.dest.same_as(store.dest) for node in _nodes(item.body, ir.TensorStore))]
        if len(producers) != 1:
            raise ValueError("Inlining needs one producer of a private tensor")
        consumers = self.get_consumers(block)
        if not consumers:
            raise ValueError("Cannot inline a stage without consumers")
        source_ids = {load.source for load in _nodes(store.value, ir.TensorLoad)}
        source_pos = self._stages.index(stage)
        for consumer in consumers:
            pos = next(i for i, item in enumerate(self._stages) if item.uid == consumer.uid)
            if pos <= source_pos:
                raise ValueError("Cannot inline across a cyclic or backward dependency")
            if any(node.dest in source_ids for node in _stage_nodes(self._stages[pos], ir.TensorStore)):
                raise ValueError("Cannot duplicate source reads into a consumer with a write to that tensor")
            for item in self._stages[source_pos + 1 : pos]:
                if any(node.dest in source_ids for node in _stage_nodes(item, ir.TensorStore)):
                    raise ValueError("Cannot move a load across an intervening tensor write")
        index_map = tirx.IndexMap([axis.var for axis in stage.axes], list(store.indices), None)
        inverse = index_map.inverse([axis.domain for axis in stage.axes])

        def inline(load):
            if not load.source.same_as(store.dest):
                return load
            indices = inverse.map_indices(load.indices)
            return substitute(store.value, dict(zip((axis.var for axis in stage.axes), indices)))

        with self.transaction():
            self._stages = [self._map_stage(item, (ir.TensorLoad, inline)) for item in self._stages if item.uid != stage.uid]
            self._drop_dead_allocation(store.dest)
            self._commit()

    def reverse_compute_inline(self, block):
        """Fold an injective epilogue without changing reduction arithmetic."""
        stage = self.get(block).spec
        self._transformable(stage)
        self._check_noalias()
        store = self._single_store(stage)
        if stage.init is not None or stage.predicate is not None or any(axis.kind != "S" for axis in stage.axes):
            raise ValueError("Reverse inlining requires an unguarded spatial consumer")
        producers = self.get_producers(block)
        if len(producers) != 1:
            raise ValueError("Reverse inlining requires exactly one producer")
        producer_ref = producers[0]
        producer = self.get(producer_ref).spec
        self._transformable(producer)
        produced = self._single_store(producer)
        self._private_tensor(produced.dest)
        if len(self.get_consumers(producer_ref)) != 1 or producer.predicate is not None:
            raise ValueError("Reverse inlining requires an unguarded producer with one consumer")
        producer_pos, consumer_pos = self._stages.index(producer), self._stages.index(stage)
        if producer_pos >= consumer_pos:
            raise ValueError("Cannot reverse-inline a cyclic or backward dependency")
        for item in self._stages[producer_pos:consumer_pos]:
            if any(node.source.same_as(store.dest) for node in _stage_nodes(item, ir.TensorLoad)) or any(
                node.dest.same_as(store.dest) for node in _stage_nodes(item, ir.TensorStore)
            ):
                raise ValueError("Cannot move an output write across an intervening tensor access")
        loads = _nodes(store.value, ir.TensorLoad)
        if len(loads) != 1 or not loads[0].source.same_as(produced.dest) or not self._pure_value(store.value):
            raise ValueError("Reverse inlining cannot move other input loads or effects")
        load = loads[0]
        # A reduction accumulator must retain its original dtype and rounding
        # points. A cast/bias epilogue stays a separate stage.
        identity = store.value.same_as(load) and store.dest.ty.dtype == produced.dest.ty.dtype
        if producer.init is not None and not identity:
            raise ValueError("A reduction epilogue with arithmetic or a cast cannot be reverse-inlined")
        inverse = tirx.IndexMap([axis.var for axis in stage.axes], list(load.indices), None).inverse([axis.domain for axis in stage.axes])

        def output_indices(indices):
            replacements = dict(zip((axis.var for axis in stage.axes), inverse.map_indices(indices)))
            return [substitute(index, replacements) for index in store.indices], replacements

        def rewrite_store(node):
            if not node.dest.same_as(produced.dest):
                return node
            indices, replacements = output_indices(node.indices)
            value = node.value
            if not identity:
                value = tvm_ffi.structural_map(
                    substitute(store.value, replacements),
                    (ir.TensorLoad, lambda item: node.value if item.source.same_as(produced.dest) else item),
                )
            return ir.TensorStore(store.dest, indices, value, node.loc)

        def rewrite_load(node):
            if not node.source.same_as(produced.dest):
                return node
            indices, _ = output_indices(node.indices)
            return ir.TensorLoad(store.dest, indices, node.loc)

        with self.transaction():
            rewritten = self._map_stage(producer, [(ir.TensorLoad, rewrite_load), (ir.TensorStore, rewrite_store)])
            self._stages = [rewritten if item.uid == producer.uid else item for item in self._stages if item.uid != stage.uid]
            self._drop_dead_allocation(produced.dest)
            self._commit()

    def reindex(self, block, access, *, skip_simplify=False):
        """Materialize an access-axis cache while preserving the public ABI.

        Unlike a tensor view, a reindex cache owns fresh storage. Its copy
        stage records the exact original indices, including offset/strides in
        the original tensor type. Padding loads use an explicit lazy guard.
        """
        del skip_simplify  # Native index simplification is not implicit here.
        view = self.get(block)
        stage = view.spec
        self._transformable(stage)
        self._check_noalias()
        kind, ordinal = access
        if kind not in {"read", "write"}:
            raise ValueError("A reindex access must select read or write")
        selected = (view.reads if kind == "read" else view.writes)[ordinal]
        original = selected.tensor.tensor
        indices = [region.begin for region in selected.region]
        axes_used = set().union(*(variables(index) for index in indices)) if indices else set()
        axes = [axis for axis in stage.axes if axis.var in axes_used]
        shape = [axis.domain.extent for axis in axes]
        cache = ir.Var(f"{original.name}_reindex", tirx.TensorType(shape, original.ty.dtype, "global"))
        copy_axes = [
            IterAxis(ir.Var(f"{axis.var.name}_copy", axis.var.ty), ir.Range(ir.const(0, axis.var.ty.dtype), axis.domain.extent), "S")
            for axis in axes
        ]
        copy_substitutions = {axis.var: copy.var + axis.domain.begin for axis, copy in zip(axes, copy_axes)}
        cache_indices = [axis.var - axis.domain.begin for axis in axes]
        source_indices = [substitute(index, copy_substitutions) for index in indices]
        copy_indices = [axis.var for axis in copy_axes]
        if kind == "read":
            if any(node.dest.same_as(original) for node in _stage_nodes(stage, ir.TensorStore)):
                raise ValueError("Cannot cache an input written by the same stage")
            all_accesses = [item for item in view.reads if item.tensor.tensor.same_as(original)]
            if len(all_accesses) != 1:
                raise ValueError("Reindex requires one index pattern for the selected input tensor")
            value = ir.TensorLoad(original, source_indices)
            guard = ir.const(True)
            for index, extent in zip(source_indices, original.ty.shape):
                guard = prim.And(guard, prim.And(index >= 0, index < extent))
            value = prim.if_then_else(guard, value, ir.const(0, original.ty.dtype.dtype))
            copy_body = ir.TensorStore(cache, copy_indices, value)

            def rewrite_load(load):
                return ir.TensorLoad(cache, cache_indices, load.loc) if load.source.same_as(original) else load

            rewritten = self._map_stage(stage, (ir.TensorLoad, rewrite_load))
        else:
            self._single_store(stage)
            if any(axis.kind == "R" for axis in axes):
                raise ValueError("A reduction output cannot depend on a reduction coordinate")
            if len([item for item in view.writes if item.tensor.tensor.same_as(original)]) != 1:
                raise ValueError("Reindex requires one index pattern for the selected output")
            copy_body = ir.TensorStore(original, source_indices, ir.TensorLoad(cache, copy_indices))

            def rewrite_load(load):
                if not load.source.same_as(original):
                    return load
                if not _same(load.indices, indices):
                    raise ValueError("Reduction accumulator indices differ from its write indices")
                return ir.TensorLoad(cache, cache_indices, load.loc)

            def rewrite_store(store):
                return ir.TensorStore(cache, cache_indices, store.value, store.loc) if store.dest.same_as(original) else store

            rewritten = self._map_stage(stage, [(ir.TensorLoad, rewrite_load), (ir.TensorStore, rewrite_store)])
        copy_predicate = None
        if kind == "write" and stage.predicate is not None:
            used_axes = variables(stage.predicate) & {axis.var for axis in stage.axes}
            if (
                not used_axes.issubset(copy_substitutions)
                or _nodes(stage.predicate, ir.TensorLoad)
                or not self._pure_value(stage.predicate)
            ):
                raise ValueError("Output reindexing requires a pure predicate over its spatial coordinates")
            copy_predicate = substitute(stage.predicate, copy_substitutions)
        with self.transaction():
            copy_stage = Stage(self._next_uid, f"{original.name}_reindex", tuple(copy_axes), copy_body, predicate=copy_predicate)
            self._next_uid += 1
            position = self._stages.index(stage)
            self._stages[position] = rewritten
            self._stages.insert(position if kind == "read" else position + 1, copy_stage)
            self._cache_tensors.add(cache)
            self._commit()
        return self._ref(copy_stage.uid)

    def transform_layout(self, block, access, index_map):
        """Apply a bijective index map to a private owning reindex tensor."""
        view = self.get(block)
        kind, ordinal = access
        selected = (view.reads if kind == "read" else view.writes)[ordinal].tensor.tensor
        if selected not in self._cache_tensors:
            raise ValueError("Carver layout changes require a freshly materialized reindex cache")
        # inverse() rejects lossy/padded maps. Public tensor types, strides and
        # offsets stay unchanged; every cache load/store is rewritten together.
        index_map.inverse(selected.ty.shape)
        new_shape = index_map.map_shape(selected.ty.shape)
        tensor = ir.Var(selected.name, tirx.TensorType(new_shape, selected.ty.dtype, selected.ty.scope))

        def load(node):
            return ir.TensorLoad(tensor, index_map.map_indices(node.indices), node.loc) if node.source.same_as(selected) else node

        def store(node):
            return (
                ir.TensorStore(tensor, index_map.map_indices(node.indices), node.value, node.loc) if node.dest.same_as(selected) else node
            )

        with self.transaction():
            self._stages = [self._map_stage(stage, [(ir.TensorLoad, load), (ir.TensorStore, store)]) for stage in self._stages]
            self._cache_tensors.remove(selected)
            self._cache_tensors.add(tensor)
            self._commit()

    def transform_block_layout(self, block, index_map):
        """Rebind stage axes through an invertible, reduction-order-safe map."""
        stage = self.get(block).spec
        self._transformable(stage)
        self._check_noalias()
        if any(not isinstance(axis.domain.begin, ir.IntImm) or int(axis.domain.begin) != 0 for axis in stage.axes):
            raise ValueError("Block layout transforms require zero-based axes")
        shapes = [axis.domain.extent for axis in stage.axes]
        inverse = index_map.inverse(shapes)
        new_shapes = index_map.map_shape(shapes)
        kinds = {var: axis.kind for var, axis in zip(index_map.initial_indices, stage.axes)}
        new_axes = []
        for ordinal, (index, extent) in enumerate(zip(index_map.final_indices, new_shapes)):
            involved = {kinds[var] for var in variables(index) if var in kinds}
            if len(involved) > 1 or involved - {"S", "R"}:
                raise ValueError("A block layout cannot combine spatial and reduction axes")
            kind = next(iter(involved)) if involved else "S"
            var = ir.Var(f"axis{ordinal}", index.ty)
            new_axes.append(IterAxis(var, ir.Range(ir.const(0, index.ty.dtype), extent), kind))
        old_reduction, new_reduction = ir.const(0, "int64"), ir.const(0, "int64")
        for axis, initial in zip(stage.axes, index_map.initial_indices):
            if axis.kind == "R":
                old_reduction = old_reduction * axis.domain.extent + initial
        for axis, index in zip(new_axes, index_map.final_indices):
            if axis.kind == "R":
                new_reduction = new_reduction * axis.domain.extent + index
        analyzer = create_analyzer()
        context = sym.ConstraintContext()
        for variable, extent in zip(index_map.initial_indices, shapes):
            context.append(sym.RangeConstraint(variable, ir.Range(ir.const(0, variable.ty.dtype), extent)))
        analyzer.sync_constraint_context(context)
        if not analyzer.can_prove(old_reduction == new_reduction):
            raise ValueError("A block layout must preserve the reduction's lexicographic accumulation order")
        # Spatial labels do not prove that iterations are independent. A
        # noalias argument may contain an in-place recurrence or internally
        # overlapping strides. The native schedule has no dependence solver
        # for those cases, so require one independently owned output element
        # per spatial coordinate before changing the iteration order.
        store = self._single_store(stage)
        access_analyzer = create_analyzer()
        access_context = sym.ConstraintContext()
        for axis in stage.axes:
            access_context.append(sym.RangeConstraint(axis.var, axis.domain))
        access_analyzer.sync_constraint_context(access_context)
        for write in _stage_nodes(stage, ir.TensorStore):
            if not write.dest.same_as(store.dest):
                raise ValueError("A block layout requires one independent output tensor")
            if not all(access_analyzer.can_prove(left == right) for left, right in zip(write.indices, store.indices)):
                raise ValueError("A block layout cannot reorder writes to different output coordinates")
        for load in _stage_nodes(stage, ir.TensorLoad):
            if load.source.same_as(store.dest) and not all(
                access_analyzer.can_prove(left == right) for left, right in zip(load.indices, store.indices)
            ):
                raise ValueError("A block layout cannot reorder cross-coordinate in-place dependencies")
        memory_index = access_analyzer.simplify(self.tensor_view(store.dest).offset_of(store.indices)[0])
        if variables(memory_index) & {axis.var for axis in stage.axes if axis.kind == "R"}:
            raise ValueError("A block layout cannot map a reduction coordinate to a different output element")
        spatial_domains = {axis.var: axis.domain for axis in stage.axes if axis.kind == "S"}
        mapping = sym.detect_iter_map([memory_index], spatial_domains, check_level="bijective", analyzer=access_analyzer)
        if mapping.errors or len(mapping.indices) != 1:
            raise ValueError("A block layout requires a bijective spatial mapping to physical output storage")
        replacements = dict(zip((axis.var for axis in stage.axes), inverse.map_indices([axis.var for axis in new_axes])))
        transformed = replace(stage.substitute(replacements), axes=tuple(new_axes))
        with self.transaction():
            self._stages = [transformed if item.uid == stage.uid else item for item in self._stages]
            self._commit()
