"""Warp-specialization scheduling over native lexical TIRx statements."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.tirx.function import renew_def

from tilelang.warp_specialize import WSOpRef, WSPipeline, WSRole, WSSchedule, WSScope, WSSync, WSSyncKind
from . import stmt_functor
from .tensor import TensorView


def _native(name, *arguments):
    return tvm_ffi.get_global_func("tl.cuda.analysis." + name)(*arguments)


def _identity(value):
    return value.__chandle__()


def _attrs(value):
    return dict(value) if value is not None else {}


def _call(name, *arguments, attrs=None):
    return ir.Evaluate(ir.Call(name, list(arguments), attrs=attrs or {}, ty="void"))


def _region(node, body, attributes=None, arguments=None):
    return ir.RegionStmt(
        node.op,
        list(node.args) if arguments is None else arguments,
        list(node.body_params),
        node.attrs if attributes is None else attributes,
        body,
        list(node.result_vars),
        loc=node.loc,
    )


def _strip_markers(statement):
    def strip(node):
        if isinstance(node, ir.Call) and node.op.name == "tl.ws_value":
            return node.args[0]
        if isinstance(node, ir.Call) and node.attrs is not None and "tl.ws_op_id" in node.attrs:
            attributes = _attrs(node.attrs)
            attributes.pop("tl.ws_op_id")
            return ir.Call(node.op, list(node.args), attrs=attributes, ty_args=list(node.ty_args), ty=node.ty, loc=node.loc)
        if isinstance(node, ir.For):
            attributes = _attrs(node.annotations)
            for key in ("tl.ws_op_id", "tl.ws_schedule", "tl.ws_pipeline_depth"):
                attributes.pop(key, None)
            return ir.For(node.var, node.begin, node.end, node.step, node.kind, node.body, attributes, loc=node.loc)
        if isinstance(node, ir.RegionStmt):
            attributes = _attrs(node.attrs)
            operation_wrapper = "tl.ws_op_id" in attributes
            for key in ("tl.ws_op_id", "tl.ws_schedule", "tl.ws_pipeline_depth"):
                attributes.pop(key, None)
            if (
                operation_wrapper
                and not attributes
                and not node.body_params
                and not node.result_vars
                and node.op.name in ("tirx.scope", "tl.scope")
            ):
                return node.body
            return _region(node, node.body, attributes)
        return None

    return stmt_functor.ir_transform(statement, None, strip)


def _summary(statement):
    memory, reads, definitions, exits = _native("AccessSummary", statement)
    return (
        {_identity(storage): (storage, int(mask)) for storage, mask in memory.items()},
        {_identity(variable) for variable in reads},
        {_identity(variable) for variable in definitions},
        bool(exits),
    )


@dataclass(eq=False)
class _Operation:
    identifier: str
    statement: ir.Stmt
    scope: str
    guards: tuple = ()
    child: str | None = None
    memory: dict = field(default_factory=dict)
    reads: set = field(default_factory=set)
    definitions: set = field(default_factory=set)


@dataclass(eq=False)
class _Scope:
    identifier: str
    statement: ir.Stmt | None
    parent: str | None
    guards: tuple = ()
    operations: list = field(default_factory=list)
    attributes: dict = field(default_factory=dict)


class _Program:
    def __init__(self, body, scheduled_scopes=None):
        self.scopes = {WSScope.ROOT: _Scope(WSScope.ROOT, None, None)}
        self.operations = {}
        self.allocations = {}
        self.aliases = dict(_native("StorageAliases", body))
        self.layouts = {}
        self.initializers = []
        self.metadata = []
        self.attributes = {}
        self.scheduled_scopes = scheduled_scopes
        self._parse(body, self.scopes[WSScope.ROOT])

    def owner(self, tensor):
        return self.aliases.get(tensor, tensor)

    def memory(self, memory):
        result = {}
        for tensor, mask in memory.values():
            owner = self.owner(tensor)
            identity = _identity(owner)
            result[identity] = (owner, mask | result.get(identity, (None, 0))[1])
        return result

    def _operation(self, identifier, statement, scope, guards, child=None):
        if identifier in self.operations:
            raise ValueError(f"Duplicate warp-specialization operation id: {identifier}")
        memory, reads, definitions, exits = _summary(statement)
        for guard in guards:
            guard_memory, guard_reads, _, _ = _summary(ir.Evaluate(guard))
            reads.update(guard_reads)
            for storage_id, (storage, mask) in guard_memory.items():
                memory[storage_id] = (storage, mask | memory.get(storage_id, (None, 0))[1])
        if exits:
            raise ValueError("Warp-specialized operations cannot contain early exits")
        operation = _Operation(identifier, statement, scope.identifier, guards, child, self.memory(memory), reads, definitions)
        self.operations[identifier] = operation
        scope.operations.append(operation)

    def _parse(self, statement, scope, guards=()):
        if isinstance(statement, ir.SeqStmt):
            for child in statement.seq:
                self._parse(child, scope, guards)
            return
        if isinstance(statement, ir.Bind) and isinstance(statement.var.ty, tirx.TensorType):
            if guards:
                raise ValueError("Warp-specialization allocations must dominate their role programs")
            self.allocations[_identity(statement.var)] = statement
            return
        if isinstance(statement, ir.If):
            self._parse(statement.then_case, scope, (*guards, statement.condition))
            if statement.else_case is not None:
                self._parse(statement.else_case, scope, (*guards, ir.prim.Not(statement.condition)))
            return
        identifier = _native("GetWSOpId", statement)
        if isinstance(statement, ir.Evaluate) and isinstance(statement.value, ir.Call):
            call = statement.value
            if call.op.name == "tl.layout_hint":
                self.layouts[call.args[0]] = call.attrs["layout"]
                return
            if call.op.name in (
                "tl.annotate_producer_reg_dealloc",
                "tl.annotate_consumer_reg_alloc",
                "tl.annotate_min_blocks_per_sm",
                "tl.no_set_max_nreg",
            ):
                if guards or scope.identifier != WSScope.ROOT:
                    raise ValueError("Warp-specialization resource hints must dominate all role programs")
                self.metadata.append(_strip_markers(statement))
                return
        if isinstance(statement, ir.RegionStmt):
            self.layouts.update(_attrs(statement.attrs.get("layout_map", {})))
            if (
                identifier
                and (self.scheduled_scopes is None or str(identifier) in self.scheduled_scopes)
                and len(statement.body.seq) == 1
                and isinstance(statement.body.seq[0], ir.While)
            ):
                loop = statement.body.seq[0]
                child = _Scope(str(identifier), loop, scope.identifier, guards, attributes=_attrs(statement.attrs))
                self.scopes[child.identifier] = child
                self._operation(child.identifier, statement, scope, guards, child.identifier)
                self._parse(loop.body, child)
                return
            if not identifier and statement.op.name in ("tl.scope", "tirx.scope"):
                if statement.body_params or statement.result_vars:
                    raise ValueError("Scheduled lexical scopes cannot introduce unplaced scalar bindings")
                if scope.identifier == WSScope.ROOT:
                    for key, value in statement.attrs.items():
                        if key not in ("tl.ws_schedule", "tl.ws_pipeline_depth", "layout_map"):
                            if key in self.attributes and not tvm_ffi.structural_equal(self.attributes[key], value):
                                raise ValueError(f"Conflicting warp-specialization scope attribute: {key}")
                            self.attributes[key] = value
                self._parse(statement.body, scope, guards)
                return
        if (
            isinstance(statement, ir.For)
            and self.scheduled_scopes is not None
            and str(identifier) in self.scheduled_scopes
            and statement.kind not in (ir.ForKind.DEFAULT, ir.ForKind.UNROLLED)
        ):
            raise ValueError("A warp-specialization scope must be a sequential loop")
        if (
            isinstance(statement, ir.For)
            and statement.kind in (ir.ForKind.DEFAULT, ir.ForKind.UNROLLED)
            and (self.scheduled_scopes is None or str(identifier) in self.scheduled_scopes)
        ):
            if not identifier:
                raise ValueError("Every scheduled loop needs a tl.ws_op_id")
            child = _Scope(str(identifier), statement, scope.identifier, guards, attributes=_attrs(statement.annotations))
            self.scopes[child.identifier] = child
            self._operation(child.identifier, statement, scope, guards, child.identifier)
            self._parse(statement.body, child)
            return
        if not identifier:
            if (
                scope.identifier == WSScope.ROOT
                and not scope.operations
                and not guards
                and isinstance(statement, ir.TensorStore)
                and isinstance(statement.dest.ty, tirx.TensorType)
                and statement.dest.ty.scope.name.startswith("local")
                and _native("SideEffect", statement.value) <= int(ir.CallEffectKind.ReadState)
            ):
                self.initializers.append(statement)
                return
            raise ValueError(f"Unplaced statement without tl.ws_op_id: {type(statement).__name__}")
        self._operation(str(identifier), statement, scope, guards)


_KINDS = {WSSyncKind.PRODUCER_ACQUIRE: 0, WSSyncKind.PRODUCER_COMMIT: 1, WSSyncKind.CONSUMER_WAIT: 2, WSSyncKind.CONSUMER_RELEASE: 3}


@dataclass
class _Placement:
    instruction: object
    stage: int
    spans: dict


class _Schedule:
    def __init__(self, program, schedule):
        self.program = program
        self.schedule = schedule
        self.roles = {}
        self.pipelines = {}
        self.owners = {}
        self.sides = {}
        self.bodies = {}
        self.placements = {}
        self._validate()

    def _validate(self):
        if not 1 <= schedule_warps(self.schedule) <= 32:
            raise ValueError("A warp-specialized block must contain between 1 and 32 warps")
        occupied = set()
        for role in self.schedule.roles:
            name = str(role.name)
            warps = set(range(role.warp_lo, role.warp_hi))
            if name in self.roles or not warps or role.warp_lo < 0 or role.warp_hi > self.schedule.num_warps or occupied & warps:
                raise ValueError("Warp roles must have unique names and disjoint in-bounds ranges")
            if role.max_nreg and (role.max_nreg < 24 or role.max_nreg > 256 or role.max_nreg % 8):
                raise ValueError("Warp role register budgets must be multiples of eight in [24, 256]")
            occupied.update(warps)
            self.roles[name] = role
        self.register_groups = {}
        for role in self.roles.values():
            if not role.max_nreg:
                continue
            for group in range(role.warp_lo // 4, (role.warp_hi + 3) // 4):
                previous = self.register_groups.setdefault(group, int(role.max_nreg))
                if previous != role.max_nreg:
                    raise ValueError("Roles in one warpgroup cannot request conflicting register budgets")
        if sum(min(4, self.schedule.num_warps - group * 4) * 32 * count for group, count in self.register_groups.items()) > 65536:
            raise ValueError("Warp-specialized register budgets exceed the CTA register file")
        for pipeline in self.schedule.pipelines:
            name = str(pipeline.name)
            if name in self.pipelines or pipeline.depth < 1 or pipeline.depth > 1024 or not pipeline.buffers:
                raise ValueError("Pipelines require unique names, nonempty buffers, and depths in [1, 1024]")
            buffers = []
            for tensor in pipeline.buffers:
                tensor = self.program.owner(tensor)
                allocation = self.program.allocations.get(_identity(tensor))
                if (
                    allocation is None
                    or not isinstance(allocation.value, ir.Call)
                    or allocation.value.op.name != "tirx.alloc_tensor"
                    or not tensor.ty.scope.name.startswith("shared")
                ):
                    raise ValueError("Pipeline buffers must be kernel-owned shared or tensor-memory allocations")
                if not any(tensor.same_as(buffer) for buffer in buffers):
                    buffers.append(tensor)
            self.pipelines[name] = WSPipeline(name, buffers, depth=int(pipeline.depth))
        for scope in self.schedule.scopes:
            identifier = str(scope.id)
            if identifier not in self.program.scopes or identifier in self.bodies:
                raise ValueError(f"Unknown or duplicate schedule scope: {identifier}")
            bodies = {}
            for role_name, instructions in scope.bodies.items():
                role_name = str(role_name)
                if role_name not in self.roles:
                    raise ValueError(f"Unknown schedule role: {role_name}")
                spans = {}
                body = []
                current_stage = 0
                for instruction in instructions:
                    if isinstance(instruction, WSSync):
                        pipeline = str(instruction.pipeline)
                        stage = int(instruction.stage)
                        if pipeline not in self.pipelines or stage < 0 or stage > 1024:
                            raise ValueError("Synchronization references an unknown pipeline or invalid stage")
                        if identifier == WSScope.ROOT and stage:
                            raise ValueError("Root-scope synchronization cannot have a software-pipeline offset")
                        previous = self.owners.setdefault(pipeline, identifier)
                        if previous != identifier:
                            raise ValueError("A pipeline must be owned by exactly one scope")
                        kind = _KINDS[instruction.kind]
                        current_stage = stage
                        side = kind < 2
                        if self.sides.setdefault((pipeline, side), role_name) != role_name:
                            raise ValueError("Each pipeline side must have exactly one signaling role")
                        if kind % 2 == 0:
                            if pipeline in spans:
                                raise ValueError(f"Pipeline {pipeline} is acquired twice without a release")
                            spans[pipeline] = (side, stage)
                        elif spans.get(pipeline) != (side, stage):
                            raise ValueError("Synchronization closes an absent span or changes its stage")
                        body.append(_Placement(instruction, stage, dict(spans)))
                        if kind % 2:
                            spans.pop(pipeline)
                    elif isinstance(instruction, WSOpRef):
                        operation = self.program.operations.get(str(instruction.id))
                        if operation is None or operation.scope != identifier:
                            raise ValueError(f"Operation {instruction.id} does not belong to scope {identifier}")
                        active_stages = {stage for _, stage in spans.values()}
                        if len(active_stages) > 1:
                            raise ValueError("An operation cannot consume different logical iterations of one scope")
                        stage = next(iter(active_stages), current_stage)
                        placement = _Placement(instruction, stage, dict(spans))
                        body.append(placement)
                        self.placements.setdefault(operation.identifier, []).append((role_name, placement))
                    else:
                        raise TypeError("A role program may contain only WSOpRef and WSSync")
                if spans:
                    raise ValueError("Every acquired pipeline span must be closed within its role iteration")
                offset = min((entry.stage for entry in body), default=0)
                for entry in body:
                    entry.stage -= offset
                    entry.spans = {name: (side, stage - offset) for name, (side, stage) in entry.spans.items()}
                bodies[role_name] = body
            self.bodies[identifier] = bodies
        if set(self.bodies) != set(self.program.scopes):
            raise ValueError("The schedule must include the root and every lexical loop scope")
        if set(self.placements) != set(self.program.operations):
            missing = set(self.program.operations) - set(self.placements)
            raise ValueError(f"Schedule does not place operations: {sorted(missing)}")
        for name in self.pipelines:
            if (name, True) not in self.sides or (name, False) not in self.sides or self.sides[name, True] == self.sides[name, False]:
                raise ValueError("Each pipeline requires two distinct producer and consumer roles")
            counts = [0, 0, 0, 0]
            for body in self.bodies[self.owners[name]].values():
                for entry in body:
                    if isinstance(entry.instruction, WSSync) and str(entry.instruction.pipeline) == name:
                        counts[_KINDS[entry.instruction.kind]] += 1
            if not counts[0] or len(set(counts)) != 1:
                raise ValueError("Pipeline acquire, commit, wait, and release counts must balance")
        self._validate_accesses()
        for scope, bodies in self.bodies.items():
            self._verify_progress(scope, bodies)
        self._verify_hierarchy()

    def _ancestor(self, ancestor, descendant):
        while descendant is not None:
            if ancestor == descendant:
                return True
            descendant = self.program.scopes[descendant].parent
        return False

    def role_memory(self, operation, role):
        if operation.child is None:
            return operation.memory
        memory = {}
        scope = self.program.scopes[operation.child]
        controls = list(scope.guards)
        if isinstance(scope.statement, ir.While):
            controls.append(scope.statement.condition)
        elif isinstance(scope.statement, ir.For):
            controls.extend((scope.statement.begin, scope.statement.end, scope.statement.step))
        sources = [self.program.memory(_summary(ir.Evaluate(expression))[0]) for expression in controls]
        for entry in self.bodies[scope.identifier].get(role, []):
            if isinstance(entry.instruction, WSOpRef):
                sources.append(self.role_memory(self.program.operations[str(entry.instruction.id)], role))
        for source in sources:
            for storage_id, (storage, mask) in source.items():
                memory[storage_id] = (storage, mask | memory.get(storage_id, (None, 0))[1])
        return memory

    def _validate_accesses(self):
        buffer_pipelines = {}
        for name, pipeline in self.pipelines.items():
            for tensor in pipeline.buffers:
                existing = buffer_pipelines.setdefault(_identity(tensor), [])
                for other in existing:
                    if (
                        not (self._ancestor(self.owners[name], self.owners[other]) or self._ancestor(self.owners[other], self.owners[name]))
                        or self.owners[name] == self.owners[other]
                    ):
                        raise ValueError("A buffer's pipelines must belong to strictly nested scopes")
                existing.append(name)
        for identifier, placements in self.placements.items():
            operation = self.program.operations[identifier]
            if len({role for role, _ in placements}) != len(placements):
                raise ValueError(f"Operation {identifier} is placed twice in one role")
            if (
                not operation.child
                and len(placements) > 1
                and any(
                    tensor_id in buffer_pipelines or (mask & 2 and not tensor.ty.scope.name.startswith("local"))
                    for tensor_id, (tensor, mask) in operation.memory.items()
                    if isinstance(tensor.ty, tirx.TensorType)
                )
            ):
                raise ValueError(f"Operation {identifier} cannot be duplicated across roles")
            writes = {
                name
                for tensor_id, (_, mask) in operation.memory.items()
                if mask & 2
                for name in buffer_pipelines.get(tensor_id, [])
                if self.owners[name] == operation.scope
            }
            if not operation.child and len(writes) > 1:
                raise ValueError("A scheduled operation can produce at most one pipeline per scope")
            for role, placement in placements:
                for tensor_id, (_, mask) in self.role_memory(operation, role).items():
                    for name in buffer_pipelines.get(tensor_id, []):
                        if self.owners[name] != operation.scope:
                            continue
                        if name not in placement.spans:
                            inner_handoff = (
                                operation.child
                                and role == self.sides[name, False]
                                and any(
                                    other != name
                                    and self._ancestor(operation.child, self.owners[other])
                                    and {self.sides[name, True], role} == {self.sides[other, True], self.sides[other, False]}
                                    for other in buffer_pipelines.get(tensor_id, [])
                                )
                            )
                            forwarded_write = (
                                operation.child
                                and self.owners[name] == WSScope.ROOT
                                and self.pipelines[name].depth == 1
                                and any(
                                    other != name
                                    and self._ancestor(operation.child, self.owners[other])
                                    and self.sides[other, True] == role
                                    and self.sides[other, False] == self.sides[name, True]
                                    for other in buffer_pipelines.get(tensor_id, [])
                                )
                            )
                            if not inner_handoff and not forwarded_write:
                                raise ValueError(f"Operation {identifier} accesses {name} outside an acquired span")
                            continue
                        if placement.spans[name][0] and not (mask & 2):
                            continue
        for scope in self.program.scopes.values():
            if scope.parent is not None:
                entering = {role for role, _ in self.placements[scope.identifier]}
                if set(self.bodies[scope.identifier]) - entering:
                    raise ValueError(f"Scope {scope.identifier} schedules a role that never enters it")
            for position, before in enumerate(scope.operations):
                for after in scope.operations[position + 1 :]:
                    scalar_dependency = bool(before.definitions & after.reads)
                    for after_role, _ in self.placements[after.identifier]:
                        before_roles = {role for role, _ in self.placements[before.identifier]}
                        before_memory = self.role_memory(before, after_role) if after_role in before_roles else before.memory
                        after_memory = self.role_memory(after, after_role)
                        hazards = {
                            tensor_id
                            for tensor_id, (tensor, mask) in before_memory.items()
                            if tensor_id in after_memory
                            and (mask | after_memory[tensor_id][1]) & 2
                            and (after_role in before_roles or not tensor.ty.scope.name.startswith("local"))
                        }
                        if not hazards and not scalar_dependency:
                            continue
                        if after_role in before_roles:
                            program = self.bodies[scope.identifier][after_role]
                            indices = {
                                str(entry.instruction.id): (entry.stage, index)
                                for index, entry in enumerate(program)
                                if isinstance(entry.instruction, WSOpRef)
                            }
                            if indices[before.identifier] > indices[after.identifier]:
                                raise ValueError("Role schedule reverses a source-program dependency")
                            if scalar_dependency and indices[before.identifier][0] != indices[after.identifier][0]:
                                raise ValueError("A scalar definition cannot be used at a different stage")
                            continue
                        if scalar_dependency:
                            raise ValueError("A scalar definition must be replicated into every role that uses it")
                        if before.child or after.child:
                            continue
                        for tensor_id in hazards:
                            protected = any(
                                self.owners[name] == scope.identifier
                                and {self.sides[name, True], self.sides[name, False]} & before_roles
                                and after_role in (self.sides[name, True], self.sides[name, False])
                                for name in buffer_pipelines.get(tensor_id, [])
                            )
                            if not protected:
                                raise ValueError("Cross-role memory dependencies require a pipeline handoff")

    def _verify_progress(self, scope, bodies):
        words = {
            role: [
                (str(entry.instruction.pipeline), _KINDS[entry.instruction.kind], entry.stage)
                for entry in body
                if isinstance(entry.instruction, WSSync)
            ]
            for role, body in bodies.items()
        }
        words = {role: word for role, word in words.items() if word}
        if not words:
            return
        names = sorted({pipeline for word in words.values() for pipeline, _, _ in word})
        max_stage = max(stage for word in words.values() for _, _, stage in word)

        def simulate(iterations):
            tokens = {name: [[True, False] for _ in range(self.pipelines[name].depth)] for name in names}
            counters = {(role, name): 0 for role in words for name in names}
            positions = {role: [0, 0] for role in words}
            seen = {}
            for _ in range(100000):
                state = (
                    tuple((role, min(clock, max_stage), index) for role, (clock, index) in positions.items()),
                    tuple(tuple(tuple(slot) for slot in tokens[name]) for name in names),
                    tuple(value % self.pipelines[name].depth for (role, name), value in counters.items()),
                )
                if iterations is None and all(clock >= max_stage for clock, _ in positions.values()) and state in seen:
                    old = seen[state]
                    if all(positions[role][0] > old[role] for role in words):
                        return
                    raise ValueError(f"Warp-specialization schedule cannot make progress in scope {scope}")
                seen[state] = {role: clock for role, (clock, _) in positions.items()}
                progress = False
                unfinished = False
                for role, word in words.items():
                    clock, index = positions[role]
                    if iterations is not None and clock >= iterations + max_stage:
                        continue
                    unfinished = True
                    pipeline, kind, stage = word[index]
                    active = clock >= stage and (iterations is None or clock - stage < iterations)
                    slot = counters[role, pipeline] % self.pipelines[pipeline].depth
                    token = tokens[pipeline][slot]
                    if active:
                        if kind == 0 and not token[0] or kind == 2 and not token[1]:
                            continue
                        if kind == 0:
                            token[0] = False
                        elif kind == 1:
                            token[1] = True
                            counters[role, pipeline] += 1
                        elif kind == 2:
                            token[1] = False
                        else:
                            token[0] = True
                            counters[role, pipeline] += 1
                    index += 1
                    if index == len(word):
                        clock += 1
                        index = 0
                    positions[role] = [clock, index]
                    progress = True
                if not unfinished:
                    return
                if not progress:
                    raise ValueError(f"Warp-specialization schedule deadlocks in scope {scope}")
            raise ValueError("Warp-specialization synchronization state space exceeds the verification budget")

        for iterations in range(1, min(max_stage + 2, 64)):
            simulate(iterations)
        simulate(None)

    def _verify_hierarchy(self):
        risky = False
        for bodies in self.bodies.values():
            edges = {}
            for body in bodies.values():
                children = [
                    (self.program.operations[str(entry.instruction.id)].child, entry.stage)
                    for entry in body
                    if isinstance(entry.instruction, WSOpRef) and self.program.operations[str(entry.instruction.id)].child is not None
                ]
                risky |= any(stage for _, stage in children)
                for before, after in zip(children, children[1:]):
                    edges.setdefault(before[0], set()).add(after[0])

            def cyclic(node, active, finished):
                if node in active:
                    return True
                if node in finished:
                    return False
                if any(cyclic(child, active | {node}, finished) for child in edges.get(node, ())):
                    return True
                finished.add(node)
                return False

            finished = set()
            risky |= any(cyclic(node, set(), finished) for node in edges)
        if not risky:
            return
        analyzer = sym.Analyzer()

        def events(identifier, role, sample, substitutions):
            scope = self.program.scopes[identifier]
            entries = self.bodies[identifier].get(role, [])
            if not entries:
                return
            for guard in scope.guards:
                value = analyzer.simplify(stmt_functor.substitute(guard, substitutions))
                if isinstance(value, ir.IntImm) and not int(value):
                    return
            if scope.statement is None:
                count = 1
            elif isinstance(scope.statement, ir.For):
                loop = scope.statement
                begin = stmt_functor.substitute(loop.begin, substitutions)
                step = stmt_functor.substitute(loop.step, substitutions)
                end = stmt_functor.substitute(loop.end, substitutions)
                length = analyzer.simplify(-((begin - end) // step))
                count = max(0, int(length)) if isinstance(length, ir.IntImm) else sample
            else:
                count = sample
            for clock in range(count + max(entry.stage for entry in entries)):
                for entry in entries:
                    if not 0 <= clock - entry.stage < count:
                        continue
                    if isinstance(entry.instruction, WSSync):
                        yield str(entry.instruction.pipeline), _KINDS[entry.instruction.kind]
                    else:
                        child = self.program.operations[str(entry.instruction.id)].child
                        if child is not None:
                            current = dict(substitutions)
                            if isinstance(scope.statement, ir.For):
                                current[loop.var] = analyzer.simplify(begin + (clock - entry.stage) * step)
                            yield from events(child, role, sample, current)

        limit = max((pipeline.depth for pipeline in self.pipelines.values()), default=1)
        limit += max((entry.stage for bodies in self.bodies.values() for body in bodies.values() for entry in body), default=0) + 1
        for sample in sorted({0, 1, 2, limit}):
            streams = {role: events(WSScope.ROOT, role, sample, {}) for role in self.roles}
            pending = {role: next(stream, None) for role, stream in streams.items()}
            tokens = {name: [[True, False] for _ in range(pipeline.depth)] for name, pipeline in self.pipelines.items()}
            counters = {(role, name): 0 for role in streams for name in self.pipelines}
            for _ in range(100000):
                if all(event is None for event in pending.values()):
                    break
                progress = False
                for role, event in pending.items():
                    if event is None:
                        continue
                    name, kind = event
                    token = tokens[name][counters[role, name] % self.pipelines[name].depth]
                    if kind == 0 and not token[0] or kind == 2 and not token[1]:
                        continue
                    if kind == 0:
                        token[0] = False
                    elif kind == 1:
                        token[1] = True
                        counters[role, name] += 1
                    elif kind == 2:
                        token[1] = False
                    else:
                        token[0] = True
                        counters[role, name] += 1
                    pending[role] = next(streams[role], None)
                    progress = True
                if not progress:
                    raise ValueError("Warp-specialization schedule deadlocks across lexical scopes")
            else:
                raise ValueError("Warp-specialization cross-scope verification exceeds the state-space budget")


def schedule_warps(schedule):
    return int(schedule.num_warps)


class _Materializer:
    def __init__(self, region, thread, target):
        self.region, self.thread, self.target = region, thread, target
        self.program = _Program(region.body, {str(scope.id) for scope in region.attrs["tl.ws_schedule"].scopes})
        self.program.layouts.update(_attrs(region.attrs.get("layout_map", {})))
        self.checked = _Schedule(self.program, region.attrs["tl.ws_schedule"])
        self.counters = {name: TensorView.create(f"ws_phase_{name}", [1], "int64", scope="local") for name in self.checked.pipelines}
        self.barriers = {
            name: TensorView.create(f"ws_barrier_{name}", [2 * pipeline.depth], "uint64", scope="shared.barrier")
            for name, pipeline in self.checked.pipelines.items()
        }
        self.storages = {}
        self.buffer_pipelines = {}
        self.prefix = []
        self.kinds = {
            operation.identifier: int(_native("ClassifyStmt", operation.statement, target))
            for operation in self.program.operations.values()
            if operation.child is None
        }
        self.completions = {}
        self.async_proxy_reads = set()
        for name, pipeline in self.checked.pipelines.items():
            storage_ids = {_identity(tensor) for tensor in pipeline.buffers}
            for side in (True, False):
                role = self.checked.sides[name, side]
                kinds = {
                    self.kinds[operation.identifier]
                    for operation in self.program.operations.values()
                    if operation.child is None
                    and self.checked._ancestor(self.checked.owners[name], operation.scope)
                    and storage_ids.intersection(operation.memory)
                    and role in {name for name, _ in self.checked.placements[operation.identifier]}
                }
                self.completions[name, side] = (5 in kinds, 1 in kinds, not kinds or bool(kinds - {1, 5}))
                if kinds.intersection({4, 5}):
                    self.async_proxy_reads.add((name, side))
        self._allocate()

    def _allocate(self):
        for name, pipeline in self.checked.pipelines.items():
            for tensor in pipeline.buffers:
                self.buffer_pipelines.setdefault(tensor, []).append(name)
        for tensor, names in self.buffer_pipelines.items():
            names.sort(key=lambda name: self._scope_depth(self.checked.owners[name]))
            storage = tensor
            for name in reversed(names):
                storage = _native("ExpandPipelineTensor", storage, self.checked.pipelines[name].depth)
            self.storages[tensor] = storage
        for alias, owner in self.program.aliases.items():
            if not alias.same_as(owner) and owner in self.storages:
                self.storages[alias] = _native("ExpandPipelineAlias", alias, owner, self.storages[owner])
                self.buffer_pipelines[alias] = self.buffer_pipelines[owner]
        zeros = {tensor: [ir.const(0)] * len(self.buffer_pipelines[tensor]) for tensor in self.storages}
        for allocation in self.program.allocations.values():
            if not isinstance(allocation.value, ir.Call) or allocation.value.op.name not in ("tirx.alloc_tensor", "tirx.decl_tensor"):
                raise ValueError("Warp-specialization tensor definitions require explicit allocation or declaration")
            if any(
                _identity(variable)
                in {_identity(scope.statement.var) for scope in self.program.scopes.values() if isinstance(scope.statement, ir.For)}
                for variable in tirx.analysis.undefined_vars(allocation.value)
            ):
                raise ValueError("Warp-specialization allocation geometry must dominate all role loops")
            storage = self.storages.get(allocation.var, allocation.var)
            arguments = [
                _native("RewriteBufferVersions", ir.Evaluate(argument), self.storages, zeros).value for argument in allocation.value.args
            ]
            value = ir.Call(allocation.value.op, arguments, attrs=_attrs(allocation.value.attrs), ty_args=[storage.ty], ty=storage.ty)
            self.prefix.append(ir.Bind(storage, value, loc=allocation.loc))
        self.prefix.extend(self.program.initializers)
        self.prefix.extend(self.program.metadata)
        for name, pipeline in self.checked.pipelines.items():
            producer = self.checked.roles[self.checked.sides[name, True]]
            consumer = self.checked.roles[self.checked.sides[name, False]]
            barrier = self.barriers[name]
            counts = []
            for side, role in ((True, producer), (False, consumer)):
                tcgen, cp_async, synchronous = self.completions[name, side]
                warps = role.warp_hi - role.warp_lo
                if tcgen and warps != 1:
                    raise ValueError("TCGEN05 issue and completion require a single-warp role")
                count = int(tcgen) * warps + (int(cp_async) + int(synchronous)) * warps * 32
                counts.extend([ir.const(count)] * pipeline.depth)
            self.prefix.append(
                ir.Bind(barrier.tensor, ir.Call("tirx.alloc_tensor", [], attrs={"barrier_init": counts}, ty_args=[barrier.tensor.ty]))
            )
            self.prefix.extend([self.counters[name].allocate(), self.counters[name].store(0, 0)])

    def _scope_depth(self, scope):
        depth = 0
        while scope is not None:
            depth += 1
            scope = self.program.scopes[scope].parent
        return depth

    def _barrier(self, name, side):
        pipeline = self.checked.pipelines[name]
        return self.barriers[name].load(side * pipeline.depth + self.counters[name].load(0) % pipeline.depth)

    def _synchronize(self, instruction):
        name = str(instruction.pipeline)
        pipeline = self.checked.pipelines[name]
        count = self.counters[name].load(0)
        phase = (count // pipeline.depth % 2).astype("int32")
        kind = _KINDS[instruction.kind]
        if kind == 0:
            return ir.If(count >= pipeline.depth, _call("tl.mbarrier_wait_parity", self._barrier(name, 1), 1 - phase), None)
        if kind == 2:
            return _call("tl.mbarrier_wait_parity", self._barrier(name, 0), phase)
        producer_side = kind == 1
        barrier = self._barrier(name, 0 if producer_side else 1)
        tcgen, cp_async, synchronous = self.completions[name, producer_side]
        statements = []
        if tcgen or cp_async or synchronous and (name, not producer_side) in self.async_proxy_reads:
            statements.append(_call("tl.fence_proxy_async"))
        if tcgen:
            statements.append(_call("tl.tcgen05_mma_arrive", tirx.address_of(barrier)))
        if cp_async:
            statements.extend(
                [
                    _call("tirx.ptx.cp_async_commit_group", ir.StringImm("async"), ir.StringImm("commit_group"), ir.StringImm("")),
                    _call("tl.ptx_cp_async_barrier_noinc", barrier),
                ]
            )
        if synchronous:
            statements.append(_call("tl.mbarrier_arrive", barrier))
        statements.append(self.counters[name].store(0, count + 1))
        return ir.SeqStmt(statements)

    def _versions(self, spans, operation):
        result = {}
        for tensor, names in self.buffer_pipelines.items():
            indices = []
            for name in names:
                count = self.counters[name].load(0)
                if (
                    name not in spans
                    and not self.checked._ancestor(self.checked.owners[name], operation.scope)
                    and not operation.memory.get(_identity(self.program.owner(tensor)), (None, 0))[1] & 2
                ):
                    count = ir.prim.Max(count - 1, ir.const(0, "int64"))
                indices.append(count % self.checked.pipelines[name].depth)
            result[tensor] = indices
        return result

    def _operation(self, operation, role, substitutions, spans):
        if operation.child:
            return self._scope(operation.child, role, substitutions, spans)
        statement = _strip_markers(operation.statement)
        statement = stmt_functor.substitute(statement, substitutions)
        if self.storages:
            statement = _native("RewriteBufferVersions", statement, self.storages, self._versions(spans, operation))
        kind = self.kinds[operation.identifier]
        if kind == 0:
            for name, pipeline in self.checked.pipelines.items():
                if any(operation.memory.get(_identity(tensor), (None, 0))[1] & 2 for tensor in pipeline.buffers):
                    if name in spans and not spans[name][0]:
                        raise ValueError(
                            "A TMA write must execute in its pipeline's producer span; the consumer never observes it finishing"
                        )
            pipelines = [
                name
                for name, (producer, _) in spans.items()
                if producer
                and any(operation.memory.get(_identity(tensor), (None, 0))[1] & 2 for tensor in self.checked.pipelines[name].buffers)
            ]
            if pipelines:
                innermost = max(self._scope_depth(self.checked.owners[name]) for name in pipelines)
                pipelines = [name for name in pipelines if self._scope_depth(self.checked.owners[name]) == innermost]

            def attach_completion(node):
                if isinstance(node, ir.Call) and node.op.name == "tl.tileop.copy" and pipelines:
                    if len(pipelines) != 1:
                        raise ValueError("A scheduled TMA copy needs exactly one producer completion pipeline")
                    attributes = _attrs(node.attrs)
                    attributes.update(
                        is_tma_copy=True,
                        barrier=self._barrier(pipelines[0], 0),
                        leader_scope_threads=(self.checked.roles[role].warp_hi - self.checked.roles[role].warp_lo) * 32,
                    )
                    return ir.Call(node.op, list(node.args), attrs=attributes, ty_args=list(node.ty_args), ty=node.ty, loc=node.loc)
                return None

            statement = stmt_functor.ir_transform(statement, None, attach_completion)
        if kind in (1, 5):
            managed = any(
                (self.completions[name, producer][1] if kind == 1 else self.completions[name, producer][0])
                for name, (producer, _) in spans.items()
            )
            if kind == 5 and not managed:
                raise ValueError("Asynchronous TMEM MMA scheduling requires an explicit completion watermark")
            if managed:

                def attach_async(node):
                    if not isinstance(node, ir.Call) or not node.op.name.startswith("tl.tileop."):
                        return None
                    attributes = _attrs(node.attrs)
                    name = node.op.name
                    if kind == 1 and name in ("tl.tileop.copy", "tl.tileop.async_copy"):
                        attributes["no_implicit_async_commit_wait"] = ir.const(1)
                    elif kind == 5:
                        attributes["is_tcgen05"] = ir.const(1)
                        name = {
                            "tl.tileop.gemm": "tl.tileop.tcgen05_gemm",
                            "tl.tileop.gemm_blockscaled": "tl.tileop.tcgen05_gemm_blockscaled",
                        }.get(name, name)
                    return ir.Call(name, list(node.args), attrs=attributes, ty_args=list(node.ty_args), ty=node.ty, loc=node.loc)

                statement = stmt_functor.ir_transform(statement, None, attach_async)
        for guard in reversed(operation.guards):
            guard = stmt_functor.substitute(guard, substitutions)
            statement = ir.If(guard, statement, None)
        if isinstance(statement, ir.Bind):
            return statement
        if kind == 4:
            return ir.SeqStmt([statement, _call("tl.tma_store_wait", ir.const(0))])
        return statement

    def _uniform(self, expression, substitutions):
        expanded = stmt_functor.substitute(expression, substitutions)
        if any(variable.same_as(self.thread) for variable in tirx.analysis.undefined_vars(expanded)):
            raise ValueError("Warp-specialized loop bounds and scope guards must be role-uniform")
        memory = self.program.memory(_summary(ir.Evaluate(expanded))[0])
        if any(tensor in self.buffer_pipelines for tensor, _ in memory.values()):
            raise ValueError("Pipeline-dependent loop bounds and guards are not uniform across roles")
        if _native("SideEffect", expanded) > int(ir.CallEffectKind.ReadState):
            raise ValueError("Warp-specialized control flow cannot have effectful conditions")

    def _scope(self, identifier, role, substitutions, inherited):
        scope = self.program.scopes[identifier]
        entries = self.checked.bodies[identifier].get(role, [])
        if not entries:
            return ir.SeqStmt([])
        max_stage = max(entry.stage for entry in entries)
        if isinstance(scope.statement, ir.While):
            if max_stage:
                body = self._staged_while(scope, role, substitutions, inherited, entries)
                for guard in reversed(scope.guards):
                    self._uniform(guard, substitutions)
                    body = ir.If(stmt_functor.substitute(guard, substitutions), body, None)
                return body
            self._uniform(scope.statement.condition, substitutions)
            memory, _, _, _ = _summary(ir.Evaluate(scope.statement.condition))
            entering = set(self.checked.bodies[identifier])
            for operation in self.program.operations.values():
                if operation.child or not any(storage_id in memory and mask & 2 for storage_id, (_, mask) in operation.memory.items()):
                    continue
                if {name for name, _ in self.checked.placements[operation.identifier]} != entering:
                    raise ValueError("While-loop control state must be updated identically in every role")
                for node in self._expressions(operation.statement):
                    self._uniform(node, substitutions)
            clock, iterations = ir.const(0), ir.const(1)
        elif scope.statement is None:
            clock, iterations = ir.const(0), ir.const(1)
        else:
            loop = scope.statement
            for bound in (loop.begin, loop.end, loop.step):
                self._uniform(bound, substitutions)
            step = sym.Analyzer().simplify(stmt_functor.substitute(loop.step, substitutions))
            if not isinstance(step, ir.IntImm) or not int(step):
                raise ValueError("Warp-specialized loops require a nonzero constant step")
            begin = stmt_functor.substitute(loop.begin, substitutions)
            end = stmt_functor.substitute(loop.end, substitutions)
            iterations = sym.Analyzer().simplify(ir.prim.Max(-((begin - end) // step), ir.const(0, loop.var.ty)))
            clock = ir.Var(f"ws_time_{identifier}_{role}", loop.var.ty)
        statements = []
        for entry in entries:
            current = dict(substitutions)
            if isinstance(scope.statement, ir.For):
                current[scope.statement.var] = begin + (clock - entry.stage) * step
            spans = {**inherited, **entry.spans}
            if isinstance(entry.instruction, WSSync):
                statement = self._synchronize(entry.instruction)
            else:
                operation = self.program.operations[str(entry.instruction.id)]
                statement = self._operation(operation, role, current, spans)
            if max_stage:
                statement = ir.If(ir.prim.And(clock >= entry.stage, clock - entry.stage < iterations), statement, None)
            statements.append(statement)
        body = ir.SeqStmt(statements)
        prefix = []
        if max_stage:
            prefix, body = self._stage_scalars(body, identifier, role)
        if isinstance(scope.statement, ir.While):
            body = ir.While(stmt_functor.substitute(scope.statement.condition, substitutions), body, loc=scope.statement.loc)
        elif scope.statement is not None:
            attributes = _attrs(scope.statement.annotations)
            for key in ("tl.ws_op_id", "num_stages", "software_pipeline_stage", "software_pipeline_order"):
                attributes.pop(key, None)
            body = ir.For(clock, 0, iterations + max_stage, 1, scope.statement.kind, body, annotations=attributes, loc=scope.statement.loc)
            if max_stage:
                body = ir.If(iterations > 0, body, None)
        if prefix:
            body = ir.SeqStmt([*prefix, body])
        for guard in reversed(scope.guards):
            self._uniform(guard, substitutions)
            body = ir.If(stmt_functor.substitute(guard, substitutions), body, None)
        return body

    @staticmethod
    def _stage_scalars(body, identifier, role):
        bindings = []
        stmt_functor.post_order_visit(
            body, lambda node: bindings.append(node) if isinstance(node, ir.Bind) and isinstance(node.var.ty, ir.PrimType) else None
        )
        storage = {
            binding.var: TensorView.create(f"ws_scalar_{identifier}_{role}_{binding.var.name}", [1], binding.var.ty, scope="local")
            for binding in bindings
        }
        if not storage:
            return [], body

        def rewrite(node):
            if isinstance(node, ir.Bind) and node.var in storage:
                return storage[node.var].store(0, node.value)
            return None

        body = stmt_functor.ir_transform(body, rewrite, None)
        body = stmt_functor.substitute(body, {variable: tensor.load(0) for variable, tensor in storage.items()})
        return [tensor.allocate() for tensor in storage.values()], body

    def _staged_while(self, scope, role, substitutions, inherited, entries):
        self._uniform(scope.statement.condition, substitutions)
        controls = set(self.program.memory(_summary(ir.Evaluate(scope.statement.condition))[0]))
        operations = [operation for operation in scope.operations if operation.child is None]
        updates = set()
        previous = None
        while previous != (controls, updates):
            previous = (set(controls), set(updates))
            for operation in operations:
                if any(storage_id in controls and mask & 2 for storage_id, (_, mask) in operation.memory.items()):
                    updates.add(operation.identifier)
                    controls.update(operation.memory)
        if not updates:
            raise ValueError("Stage-shifted while loops need explicit replicated local control updates")
        state = {}
        entering = set(self.checked.bodies[scope.identifier])
        for operation in operations:
            if operation.identifier not in updates:
                continue
            if {name for name, _ in self.checked.placements[operation.identifier]} != entering:
                raise ValueError("While-loop control state must be updated identically in every role")
            for tensor, _ in operation.memory.values():
                if not isinstance(tensor.ty, tirx.TensorType) or tensor.ty.scope.name != "local":
                    raise ValueError("Stage-shifted while control must depend only on private local state")
                if _identity(tensor) not in self.program.allocations:
                    raise ValueError("Stage-shifted while control must use kernel-owned local state")
                state[tensor] = self.program.allocations[_identity(tensor)]
            for expression in self._expressions(operation.statement):
                self._uniform(expression, substitutions)
        update_entries = [entry for entry in entries if isinstance(entry.instruction, WSOpRef) and str(entry.instruction.id) in updates]
        update_ids = {id(entry) for entry in update_entries}
        seen_update = False
        for entry in entries:
            if isinstance(entry.instruction, WSOpRef):
                identifier = str(entry.instruction.id)
                if identifier in updates:
                    seen_update = True
                elif seen_update and controls.intersection(self.program.operations[identifier].memory):
                    raise ValueError("While control updates must follow every operation using that iteration's state")
        stages = sorted({0, *(entry.stage for entry in entries if id(entry) not in update_ids)})
        maps = {0: {}}
        prefix = []
        for stage in stages[1:]:
            mapping = {tensor: ir.Var(f"ws_state_{scope.identifier}_{role}_{stage}_{tensor.name}", tensor.ty) for tensor in state}
            for alias, owner in self.program.aliases.items():
                if owner in state and not alias.same_as(owner):
                    mapping[alias] = ir.Var(f"ws_state_{scope.identifier}_{role}_{stage}_{alias.name}", alias.ty)
            for allocation in self.program.allocations.values():
                if allocation.var not in mapping:
                    continue
                prefix.append(stmt_functor.substitute(allocation, {**substitutions, **mapping}))
                if allocation.var in state:
                    indices = [ir.Var(f"ws_state_index_{axis}", "int32") for axis in range(len(allocation.var.ty.shape))]
                    copy = ir.TensorStore(mapping[allocation.var], indices, ir.TensorLoad(allocation.var, indices))
                    for index, extent in reversed(list(zip(indices, allocation.var.ty.shape))):
                        copy = ir.For(index, 0, extent, 1, ir.ForKind.DEFAULT, copy)
                    prefix.append(copy)
            maps[stage] = mapping
        clock = TensorView.create(f"ws_clock_{scope.identifier}_{role}", [1], "int64", scope="local")
        prefix.extend([clock.allocate(), clock.store(0, 0)])
        conditions = {
            stage: stmt_functor.substitute(scope.statement.condition, {**substitutions, **mapping}) for stage, mapping in maps.items()
        }
        active = {stage: ir.prim.And(clock.load(0) >= stage, condition) for stage, condition in conditions.items()}
        statements = []
        for entry in entries:
            if id(entry) in update_ids:
                continue
            current = {**substitutions, **maps[entry.stage]}
            spans = {**inherited, **entry.spans}
            if isinstance(entry.instruction, WSSync):
                statement = self._synchronize(entry.instruction)
            else:
                statement = self._operation(self.program.operations[str(entry.instruction.id)], role, current, spans)
            statements.append(ir.If(active[entry.stage], statement, None))
        for stage in stages:
            current = {**substitutions, **maps[stage]}
            statements.append(
                ir.If(
                    active[stage],
                    ir.SeqStmt(
                        [
                            self._operation(self.program.operations[str(entry.instruction.id)], role, current, {**inherited, **entry.spans})
                            for entry in update_entries
                        ]
                    ),
                    None,
                )
            )
        statements.append(clock.store(0, clock.load(0) + 1))
        scalar_allocations, body = self._stage_scalars(ir.SeqStmt(statements), scope.identifier, role)
        prefix.extend(scalar_allocations)
        condition = conditions[0]
        for stage in stages[1:]:
            condition = ir.prim.Or(condition, conditions[stage])
        return ir.SeqStmt([*prefix, ir.While(condition, body, loc=scope.statement.loc)])

    @staticmethod
    def _expressions(statement):
        expressions = []

        def visit(node):
            if isinstance(node, ir.TensorStore):
                expressions.extend([node.value, *node.indices])
            elif isinstance(node, ir.Bind):
                expressions.append(node.value)
            elif isinstance(node, ir.Evaluate):
                expressions.append(node.value)

        stmt_functor.post_order_visit(statement, visit)
        return expressions

    def materialize(self):
        private = {
            allocation.var: allocation
            for allocation in self.program.allocations.values()
            if allocation.var.ty.scope.name.startswith("local")
        }
        role_storages = []
        roles = []
        for role_name, role in self.checked.roles.items():
            substitutions = {self.thread: self.thread - role.warp_lo * 32}
            body = self._scope(WSScope.ROOT, role_name, substitutions, {})
            used = set(tirx.analysis.undefined_vars(body))
            while True:
                previous = set(used)
                owners = {self.program.owner(variable) for variable in used if variable in private}
                for allocation in private.values():
                    if allocation.var in used:
                        used.update(tirx.analysis.undefined_vars(allocation.value))
                for initializer in self.program.initializers:
                    if self.program.owner(initializer.dest) in owners:
                        used.update(tirx.analysis.undefined_vars(initializer))
                if used == previous:
                    break
            storage = {variable: ir.Var(variable.name, variable.ty) for variable in private if variable in used}
            body = stmt_functor.substitute(body, storage)
            if role.max_nreg:
                body = ir.SeqStmt([_call("tl.set_max_nreg", ir.const(role.max_nreg), ir.const(int(role.max_nreg > 64))), body])
            body = renew_def(tirx.Function([], body, ret_type=ir.TupleType([]))).body
            condition = ir.prim.And(self.thread >= role.warp_lo * 32, self.thread < role.warp_hi * 32)
            role_storages.append((condition, {**substitutions, **storage}))
            roles.append(ir.If(condition, body, None))
        bodies = []
        for statement in self.prefix:
            variable = (
                statement.var if isinstance(statement, ir.Bind) else statement.dest if isinstance(statement, ir.TensorStore) else None
            )
            if variable not in private:
                bodies.append(statement)
                continue
            for condition, storage in role_storages:
                if variable not in storage:
                    continue
                rewritten = stmt_functor.substitute(statement, storage)
                bodies.append(ir.If(condition, rewritten, None) if isinstance(statement, ir.TensorStore) else rewritten)
        occupied = {warp for role in self.checked.roles.values() for warp in range(role.warp_lo, role.warp_hi)}
        for group, count in self.checked.register_groups.items():
            idle = [warp for warp in range(group * 4, min((group + 1) * 4, self.checked.schedule.num_warps)) if warp not in occupied]
            while idle:
                begin = idle.pop(0)
                end = begin + 1
                while idle and idle[0] == end:
                    end = idle.pop(0) + 1
                condition = ir.prim.And(self.thread >= begin * 32, self.thread < end * 32)
                roles.append(ir.If(condition, _call("tl.set_max_nreg", ir.const(count), ir.const(int(count > 64))), None))
        bodies.append(ir.RegionStmt("tirx.scope", [], [], ir.DictAttrs({"tl.ws_concurrent": True}), roles))
        attributes = _attrs(self.region.attrs)
        attributes.update(self.program.attributes)
        attributes.pop("tl.ws_schedule")
        attributes.pop("layout_map", None)
        attributes["warp_specialize"] = True
        layouts = {}
        for tensor, layout in self.program.layouts.items():
            if tensor in private:
                for _, storage in role_storages:
                    if tensor in storage:
                        layouts[storage[tensor]] = layout
            elif tensor in self.storages:
                depths = [self.checked.pipelines[name].depth for name in self.buffer_pipelines[tensor]]
                layouts[self.storages[tensor]] = layout.expand(depths)
            else:
                layouts[tensor] = layout
        if layouts:
            attributes["layout_map"] = layouts
        return _region(self.region, bodies, attributes)


@tvm_ffi.register_global_func("tl.cuda.native.MaterializeWSSchedule")
def materialize_ws_schedule(function):
    target = function.attrs.get("target")
    requested = {}

    def schedules(statement):
        if isinstance(statement, ir.SeqStmt):
            return [schedule for child in statement.seq for schedule in schedules(child)]
        if isinstance(statement, ir.RegionStmt) and statement.op.name in ("tirx.scope", "tl.scope"):
            if "tl.ws_schedule" in statement.attrs:
                return [statement.attrs["tl.ws_schedule"]]
            return schedules(statement.body)
        return []

    def visit(statement, thread=None, dimensions=()):
        if isinstance(statement, ir.SeqStmt):
            return ir.SeqStmt([visit(child, thread, dimensions) for child in statement.seq], loc=statement.loc)
        if isinstance(statement, ir.RegionStmt):
            arguments = list(statement.args)
            current = thread
            if statement.op.name == "tirx.launch_thread":
                tag = str(arguments[0].value)
                if tag.startswith("threadIdx."):
                    dimensions = (*dimensions, (tag, arguments[1]))
                if tag == "threadIdx.x":
                    current = statement.body_params[0]
            found = schedules(statement) if current is not None else []
            if "tl.ws_schedule" in statement.attrs or found:
                if current is None or target is None or target.kind.name != "cuda":
                    raise ValueError("Warp-specialization schedules require a CUDA thread launch and target")
                if any(tag != "threadIdx.x" and not sym.Analyzer().can_prove(extent == 1) for tag, extent in dimensions):
                    raise ValueError("Warp-specialization schedules support one thread dimension")
                if len(found) > 1:
                    raise ValueError("One kernel scope cannot contain multiple warp-specialization schedules")
                if "tl.ws_schedule" not in statement.attrs:
                    statement = _region(statement, statement.body, {**_attrs(statement.attrs), "tl.ws_schedule": found[0]})
                materializer = _Materializer(statement, current, target)
                requested[_identity(current)] = materializer.checked.schedule.num_warps * 32
                return materializer.materialize()
            body = visit(statement.body, current, dimensions)
            if (
                current is not None
                and statement.op.name == "tirx.launch_thread"
                and str(arguments[0].value) == "threadIdx.x"
                and _identity(current) in requested
            ):
                arguments[1] = ir.const(requested[_identity(current)], arguments[1].ty)
            return _region(statement, body, arguments=arguments)
        if isinstance(statement, ir.For):
            return ir.For(
                statement.var,
                statement.begin,
                statement.end,
                statement.step,
                statement.kind,
                visit(statement.body, thread, dimensions),
                statement.annotations,
                loc=statement.loc,
            )
        if isinstance(statement, ir.If):
            return ir.If(
                statement.condition,
                visit(statement.then_case, thread, dimensions),
                None if statement.else_case is None else visit(statement.else_case, thread, dimensions),
                loc=statement.loc,
            )
        return statement

    body = visit(function.body)
    attributes = _attrs(function.attrs)
    if requested:
        attributes["tl_tiled_ws_applied"] = True
    return tirx.Function(list(function.params), body, ret_type=function.ret_type, attrs=ir.DictAttrs(attributes), loc=function.loc)


def _role_based(region, body, worker_threads, target):
    from .ws_auto import RoleScheduler

    return RoleScheduler(region, body, worker_threads, target).schedule()


@tvm_ffi.register_global_func("tl.cuda.native.RoleBasedSchedule")
def role_based_schedule(region, body, worker_threads, target):
    try:
        return _role_based(region, body, worker_threads, target)
    except (ValueError, TypeError) as error:
        warnings.warn(f"Automatic warp specialization declined the kernel: {error}", RuntimeWarning, stacklevel=2)
        return None
