"""Native role assignment and per-storage warp-specialization handoffs."""

from __future__ import annotations

from dataclasses import dataclass, field

from tvm_ffi import structural_equal
from tvm_tirx import ir, tirx

from tilelang.warp_specialize import WSOpRef, WSPipeline, WSRole, WSSchedule, WSScope, WSSync
from . import stmt_functor
from .tensor import TensorView
from .ws import _Program, _Schedule, _attrs, _identity, _native, _summary


_ROLES = ("Worker", "Load", "MMA", "Store")


def _private(tensor):
    return isinstance(tensor.ty, tirx.TensorType) and tensor.ty.scope.name.startswith("local")


def _on_chip(tensor):
    return isinstance(tensor.ty, tirx.TensorType) and tensor.ty.scope.name.startswith(("shared", "tmem"))


def _calls(statement):
    result = []
    stmt_functor.post_order_visit(statement, lambda node: result.append(node) if isinstance(node, ir.Call) else None)
    return result


@dataclass
class _Binding:
    tensor: ir.Var
    scope: str
    producer: str
    consumer: str
    opens: dict = field(default_factory=dict)
    closes: dict = field(default_factory=dict)
    name: str = ""
    depth: int = 1


class RoleScheduler:
    def __init__(self, region, body, worker_threads, target):
        self.program = _Program(body)
        self.readonly = set()
        if region is not None:
            self.program.layouts.update(_attrs(region.attrs.get("layout_map", {})))
            attributes = _attrs(region.attrs)
            self.readonly = {_identity(self.program.owner(tensor)) for tensor in attributes.pop("tl.ws_readonly_storage", [])}
            self.program.scopes[WSScope.ROOT].attributes.update(attributes)
        self.worker_threads = worker_threads
        self.target = target
        self.leaves = [operation for operation in self.program.operations.values() if operation.child is None]
        self.kinds = {}
        self.roles = {}
        self.movable = set()
        self.bindings = []

    def _ancestor(self, outer, inner):
        while inner is not None:
            if inner == outer:
                return True
            inner = self.program.scopes[inner].parent
        return False

    def _classify(self):
        for operation in self.leaves:
            kind = int(_native("ClassifyStmt", operation.statement, self.target))
            calls = _calls(operation.statement)
            if kind == 2 or any(call.op.name.startswith("tl.tileop.") and int(_attrs(call.attrs).get("wg_wait", 0)) != 0 for call in calls):
                raise ValueError("Hand-managed asynchronous work requires an explicit completion schedule")
            if kind in (0, 4):
                for call in calls:
                    mask = _attrs(call.attrs).get("cluster_mask", 0)
                    if not isinstance(mask, (int, ir.IntImm)) or int(mask):
                        raise ValueError("Cluster multicast requires a cluster-wide completion protocol")
                for tensor, access in operation.memory.values():
                    if not _on_chip(tensor) or not (access & (2 if kind == 0 else 1)):
                        continue
                    layout = self.program.layouts.get(tensor)
                    if layout is not None and not _native("IsTmaCompatibleLayout", layout, tensor):
                        kind = 6
            self.kinds[operation.identifier] = kind
            role = "Load" if kind in (0, 1, 3) else "Store" if kind == 4 else "MMA" if kind == 5 else "Worker"
            self.roles[operation.identifier] = {role}
            tile_calls = [call for call in calls if call.op.name.startswith("tl.tileop.")]
            if kind != 6 or any(call.op.name != "tl.tileop.copy" for call in tile_calls):
                continue
            binding = isinstance(operation.statement, ir.Bind)
            writes = [(tensor, mask) for tensor, mask in operation.memory.values() if mask & 2]
            reads = [tensor for tensor, mask in operation.memory.values() if mask & 1]
            local_state = all(_private(tensor) for tensor, _ in writes) and (
                binding or all(_private(tensor) or _identity(tensor) in self.readonly for tensor in reads)
            )
            statement = operation.statement
            while isinstance(statement, ir.RegionStmt) and len(statement.body.seq) == 1:
                statement = statement.body.seq[0]
            shared_index = (
                isinstance(statement, ir.TensorStore)
                and _on_chip(statement.dest)
                and not reads
                and _native("SideEffect", statement.value) <= int(ir.CallEffectKind.ReadState)
            )
            safe_calls = all(
                call.op.name == "tl.tileop.copy" or _native("SideEffect", call) <= int(ir.CallEffectKind.ReadState) for call in calls
            )
            if (local_state and (writes or binding) and safe_calls) or shared_index:
                self.roles[operation.identifier] = set()
                self.movable.add(operation.identifier)

    def _place(self):
        definitions = {variable: operation for operation in self.leaves for variable in operation.definitions}
        writers = {}
        for operation in self.leaves:
            for storage_id, (tensor, mask) in operation.memory.items():
                if mask & 2 and (_private(tensor) or operation.identifier in self.movable):
                    writers.setdefault(storage_id, []).append(operation)

        def demand(operation, needed):
            missing = needed - self.roles[operation.identifier]
            if not missing:
                return False
            if operation.identifier not in self.movable:
                raise ValueError(f"Operation {operation.identifier} depends on another role's private state")
            self.roles[operation.identifier].update(missing)
            return True

        def dependencies(memory, reads, needed, current):
            changed = False
            for variable in reads:
                defining = definitions.get(variable)
                if defining is not None and defining is not current:
                    changed |= demand(defining, needed)
            for storage_id, (_, mask) in memory.items():
                if mask & 1:
                    for writer in writers.get(storage_id, []):
                        if writer is not current:
                            changed |= demand(writer, needed)
            return changed

        def propagate():
            changed = True
            while changed:
                changed = False
                for operation in self.leaves:
                    needed = self.roles[operation.identifier]
                    if not needed:
                        continue
                    changed |= dependencies(operation.memory, operation.reads, needed, operation)
                    scope = self.program.scopes[operation.scope]
                    while scope.parent is not None:
                        controls = list(scope.guards)
                        if isinstance(scope.statement, ir.While):
                            controls.append(scope.statement.condition)
                        elif isinstance(scope.statement, ir.For):
                            controls.extend((scope.statement.begin, scope.statement.end, scope.statement.step))
                        for expression in controls:
                            memory, reads, _, _ = _summary(ir.Evaluate(expression))
                            changed |= dependencies(memory, reads, needed, None)
                        scope = self.program.scopes[scope.parent]

        propagate()
        active = set().union(*self.roles.values())
        for operation in self.leaves:
            if not self.roles[operation.identifier]:
                demand(operation, active)
        propagate()
        for scope in reversed(list(self.program.scopes.values())):
            entered = set().union(*(self.roles[operation.identifier] for operation in scope.operations)) if scope.operations else set()
            if scope.parent is not None:
                self.roles[scope.identifier] = entered
        for operation in self.leaves:
            if len(self.roles[operation.identifier]) > 1 and any(_on_chip(tensor) for tensor, _ in operation.memory.values()):
                raise ValueError("Cross-role memory dependencies require a pipeline handoff")
            for storage_id, (tensor, mask) in operation.memory.items():
                if tensor.ty.scope.name == "global" and mask & 2:
                    users = set().union(*(self.roles[other.identifier] for other in self.leaves if storage_id in other.memory))
                    if not users <= self.roles[operation.identifier]:
                        raise ValueError("Global writes cannot be shared by independently scheduled roles")

    def _resolve(self, tensor, scope, bindings):
        touches = []
        for position, operation in enumerate(scope.operations):
            ownership = None
            if operation.child:
                ownership = self._resolve(tensor, self.program.scopes[operation.child], bindings)
            elif _identity(tensor) in operation.memory:
                role = next(iter(self.roles[operation.identifier]))
                ownership = (role, role)
            if ownership is not None:
                touches.append((position, *ownership))
        if not touches:
            return None
        summary = (touches[0][1], touches[-1][2])
        if len(touches) == 1 or all(before[2] == after[1] for before, after in zip(touches, touches[1:] + touches[:1])):
            return summary
        transfer = next(((before[2], after[1]) for before, after in zip(touches, touches[1:]) if before[2] != after[1]), None)
        producer, consumer = transfer or summary
        if any(role not in (producer, consumer) for _, initial, final in touches for role in (initial, final)):
            raise ValueError("One storage cannot pass between more than two roles in a scope")
        pipeline = _Binding(tensor, scope.identifier, producer, consumer)

        def event(events, position, kind):
            events.setdefault(position, []).append(kind)

        event(pipeline.opens, touches[0][0], 0)
        handoff = True
        for index, (position, initial, final) in enumerate(touches):
            if index and touches[index - 1][2] != initial:
                previous = touches[index - 1]
                if previous[2] != (producer if handoff else consumer) or not handoff and previous[1] != consumer:
                    raise ValueError("Storage ownership must alternate between one producer and one consumer")
                event(pipeline.closes, previous[0] + 1, 1 if handoff else 3)
                event(pipeline.opens, position, 2 if handoff else 0)
                handoff = not handoff
            if initial == final:
                if initial == consumer and handoff:
                    raise ValueError("A consumer cannot read before its producer handoff")
                continue
            if not handoff:
                raise ValueError("Nested ownership cannot change inside a consumer span")
            if final == consumer:
                event(pipeline.closes, position + 1, 1)
                if index + 1 < len(touches):
                    event(pipeline.opens, touches[index + 1][0], 2)
                else:
                    event(pipeline.closes, position + 1, 2)
                handoff = False
        if handoff:
            raise ValueError("An automatic pipeline cannot end in its producer span")
        event(pipeline.closes, touches[-1][0] + 1, 3)
        bindings.append(pipeline)
        return summary

    def _guarded_epoch_is_complete(self, users, scope):
        if not users or not users[0].guards or any(operation.scope != scope.identifier for operation in users):
            return False
        guards = users[0].guards
        if any(
            len(operation.guards) != len(guards) or any(not structural_equal(left, right) for left, right in zip(operation.guards, guards))
            for operation in users
        ):
            return False
        variables = set()
        current = scope
        while current is not None:
            if isinstance(current.statement, ir.For):
                variables.add(current.statement.var)
            current = self.program.scopes[current.parent] if current.parent is not None else None
        return all(
            _native("SideEffect", guard) == int(ir.CallEffectKind.Pure)
            and all(variable in variables for variable in tirx.analysis.undefined_vars(guard))
            for guard in guards
        )

    def _pipelines(self):
        names = set()
        for allocation in self.program.allocations.values():
            tensor = allocation.var
            if not _on_chip(tensor):
                continue
            users = [operation for operation in self.leaves if _identity(tensor) in operation.memory]
            if not users or not any(operation.memory[_identity(tensor)][1] & 2 for operation in users):
                continue
            if len(set().union(*(self.roles[operation.identifier] for operation in users))) < 2:
                continue
            bindings = []
            self._resolve(tensor, self.program.scopes[WSScope.ROOT], bindings)
            for index, pipeline in enumerate(bindings):
                scope = self.program.scopes[pipeline.scope]
                base = str(tensor.name) + ("" if index == 0 else "_" + ("root" if scope.parent is None else scope.identifier))
                name = base
                suffix = 2
                while name in names:
                    name = f"{base}_{suffix}"
                    suffix += 1
                names.add(name)
                pipeline.name = name
                recurrent = False
                guarded = False
                for operation in users:
                    inside = self._ancestor(scope.identifier, operation.scope)
                    mask = operation.memory[_identity(tensor)][1]
                    role = next(iter(self.roles[operation.identifier]))
                    recurrent |= inside and bool(mask & (2 if role == pipeline.consumer else 1))
                    if mask & 2 and (not inside or not index or not self._ancestor(bindings[index - 1].scope, operation.scope)):
                        conditions = []
                        stmt_functor.post_order_visit(
                            operation.statement, lambda node: conditions.append(node) if isinstance(node, ir.If) else None
                        )
                        guarded |= bool(operation.guards or conditions)
                annotation = scope.attributes.get("tl.ws_pipeline_depth", {})
                depth = annotation.get(tensor)
                if depth is None:
                    depth = 1 if recurrent else scope.attributes.get("num_stages", 1)
                pipeline.depth = max(1, int(depth))
                if guarded and pipeline.depth > 1 and not self._guarded_epoch_is_complete(users, scope):
                    raise ValueError("Guarded writes cannot expose a stale version of an automatic pipeline")
            self.bindings.extend(bindings)

    def _role_plan(self):
        active = set().union(*self.roles.values())
        plans = []
        cursor = 0
        for role in _ROLES:
            if role not in active:
                continue
            wide_load = role == "Load" and any(
                self.kinds[operation.identifier] in (1, 3) for operation in self.leaves if role in self.roles[operation.identifier]
            )
            width = self.worker_threads // 32 if role == "Worker" else max(4, self.worker_threads // 32) if wide_load else 1
            plans.append([role, cursor, cursor + width, 0 if role == "Worker" else 48 if wide_load else 24])
            cursor += width
        warps = (cursor + 3) // 4 * 4
        if warps * 32 > int(self.target.attrs.get("max_num_threads", 1024)):
            raise ValueError("The scheduled roles exceed the target thread budget")
        shared_bytes = 0
        tensor_bytes = 0
        blocks = min(4, 2048 // (warps * 32))
        for allocation in self.program.allocations.values():
            tensor = allocation.var
            if not _on_chip(tensor):
                continue
            view = TensorView(tensor)
            size = (view.dtype.bits * view.dtype.lanes + 7) // 8
            for extent in view.shape:
                if not isinstance(extent, ir.IntImm):
                    blocks = 1
                    break
                size *= int(extent)
            for pipeline in self.bindings:
                if pipeline.tensor.same_as(tensor):
                    size *= pipeline.depth
            if tensor.ty.scope.name.startswith("shared.tmem") or tensor.ty.scope.name == "tmem":
                tensor_bytes += size
            else:
                shared_bytes += size
        if shared_bytes:
            blocks = min(blocks, 228 * 1024 // shared_bytes)
        if tensor_bytes:
            blocks = min(blocks, 256 * 1024 // tensor_bytes)
        hints = {statement.value.op.name: statement.value.args for statement in self.program.metadata}
        if "tl.annotate_min_blocks_per_sm" in hints:
            blocks = int(hints["tl.annotate_min_blocks_per_sm"][0])
        average = 65536 // (max(1, blocks) * warps * 32) // 8 * 8
        worker_warps = self.worker_threads // 32 if "Worker" in active else 0
        worker_registers = 0
        if worker_warps and worker_warps % 4 == 0:
            budget = warps // 4 * average
            for group in range(worker_warps // 4, warps // 4):
                budget -= max(
                    [
                        24,
                        *[
                            registers
                            for role, begin, end, registers in plans
                            if role != "Worker" and begin < (group + 1) * 4 and group * 4 < end
                        ],
                    ]
                )
            worker_registers = budget // (worker_warps // 4) // 8 * 8
            if worker_registers < max(24, average) or worker_registers > 256:
                worker_registers = 0
        roles = []
        for role, begin, end, registers in plans:
            registers = worker_registers if role == "Worker" else registers if worker_registers else 0
            hint = "tl.annotate_consumer_reg_alloc" if role == "Worker" else "tl.annotate_producer_reg_dealloc"
            if hint in hints:
                registers = int(hints[hint][0])
            if "tl.no_set_max_nreg" in hints:
                registers = 0
            roles.append(WSRole(role, warps_lo=begin, warps_hi=end, max_nreg=registers))
        return warps, roles

    def schedule(self):
        if self.worker_threads <= 0 or self.worker_threads % 32:
            return None
        self._classify()
        self._place()
        self._pipelines()
        if not self.bindings:
            return None
        warps, roles = self._role_plan()
        scopes = []
        constructors = (WSSync.producer_acquire, WSSync.producer_commit, WSSync.consumer_wait, WSSync.consumer_release)
        for scope in reversed(list(self.program.scopes.values())):
            bindings = [pipeline for pipeline in self.bindings if pipeline.scope == scope.identifier]
            bodies = {}
            for role in _ROLES:
                instructions = []
                for position in range(len(scope.operations) + 1):
                    for pipeline in bindings:
                        for kind in pipeline.closes.get(position, []):
                            if role == (pipeline.producer if kind < 2 else pipeline.consumer):
                                instructions.append(constructors[kind](pipeline.name))
                    for pipeline in bindings:
                        for kind in pipeline.opens.get(position, []):
                            if role == (pipeline.producer if kind < 2 else pipeline.consumer):
                                instructions.append(constructors[kind](pipeline.name))
                    if position < len(scope.operations):
                        operation = scope.operations[position]
                        if role in self.roles[operation.identifier]:
                            instructions.append(WSOpRef(operation.identifier))
                if instructions:
                    bodies[role] = instructions
            scopes.append(WSScope(scope.identifier, bodies))
        pipelines = [WSPipeline(pipeline.name, [pipeline.tensor], pipeline.depth) for pipeline in self.bindings]
        result = WSSchedule(warps, roles, pipelines, scopes)
        _Schedule(self.program, result)
        return result
