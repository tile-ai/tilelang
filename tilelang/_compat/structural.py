"""Binding-aware traversal of TileLang metadata without changing native hooks."""

import tvm_ffi
from tvm_tirx import ir, tirx


def _callbacks(callbacks):
    if callbacks is None:
        return []
    if callable(callbacks):
        return [(tvm_ffi.Object, callbacks)]
    if isinstance(callbacks, tuple) and len(callbacks) == 2 and callable(callbacks[1]):
        return [callbacks]
    return list(callbacks)


def _callback(node, callbacks):
    candidates = []
    for position, (classes, function) in enumerate(callbacks):
        classes = classes if isinstance(classes, tuple) else (classes,)
        for cls in classes:
            if isinstance(node, cls):
                distance = type(node).__mro__.index(cls)
                candidates.append((distance, position, function))
    return min(candidates, key=lambda item: item[:2])[2] if candidates else None


def _same(first, second):
    if isinstance(first, tvm_ffi.Object):
        return isinstance(second, tvm_ffi.Object) and first.same_as(second)
    return first == second


def _call_attrs(node):
    return tvm_ffi.get_global_func("tl.ir.AttrsMayContainExpr")(node)


class _Engine:
    def __init__(self, native):
        self.native = native

    def __getattr__(self, name):
        return getattr(self.native, name)

    def _definitions(self, value, *, pattern=False, mutate=False):
        kind = tvm_ffi.DefRegionKind.DEF_PATTERN if pattern else tvm_ffi.DefRegionKind.DEF_SIMPLE
        action = self.native.mutate if mutate else self.native.visit
        return self.native.with_def_region_kind(kind, lambda: action(value))

    def _visit_metadata(self, value):
        if isinstance(value, (dict, tvm_ffi.Map)):
            children = [element for pair in value.items() for element in pair]
        elif isinstance(value, (tuple, list, tvm_ffi.Array)):
            children = value
        else:
            return self.native.visit(value)
        for child in children:
            result = self._visit_metadata(child)
            if result is not None:
                return result
        return None

    def default_visit(self, node):
        if isinstance(node, ir.DictAttrs):
            return self._visit_metadata(node.dict)
        if isinstance(node, ir.For):
            result = self._definitions(node.var)
            if result is not None:
                return result
            for child in (node.begin, node.end, node.step, node.body):
                result = self.native.visit(child)
                if result is not None:
                    return result
            return self._visit_metadata(node.annotations)
        elif isinstance(node, ir.RegionStmt):
            result = self.native.visit(node.args)
            if result is not None:
                return result
            result = self._definitions(node.body_params)
            if result is not None:
                return result
            for child in (node.body, node.attrs):
                result = self.native.visit(child)
                if result is not None:
                    return result
            return self._definitions(node.result_vars)
        elif isinstance(node, tirx.Function):
            result = self._definitions(node.params, pattern=True)
            if result is not None:
                return result
            children = [node.ret_type, node.body, node.attrs]
        elif isinstance(node, ir.Call):
            children = [node.ty]
            if not isinstance(node.op, ir.Op):
                children.append(node.op)
            children.append(node.args)
            if _call_attrs(node):
                children.append(node.attrs)
            if node.ty_args:
                children.append(node.ty_args)
        else:
            return self.native.default_visit(node)
        for child in children:
            result = self.native.visit(child)
            if result is not None:
                return result
        return None

    def _mutate_metadata(self, value):
        if isinstance(value, (dict, tvm_ffi.Map)):
            entries = [(self._mutate_metadata(key), self._mutate_metadata(element)) for key, element in value.items()]
            if all(
                _same(old_key, key) and _same(old_value, element) for (old_key, old_value), (key, element) in zip(value.items(), entries)
            ):
                return value
            return tvm_ffi.convert(dict(entries))
        if isinstance(value, (tuple, list, tvm_ffi.Array)):
            entries = [self._mutate_metadata(element) for element in value]
            return value if all(_same(old, new) for old, new in zip(value, entries)) else tvm_ffi.convert(entries)
        return self.native.mutate(value)

    def default_mutate(self, node):
        if isinstance(node, ir.DictAttrs):
            values = self._mutate_metadata(node.dict)
            return node if _same(values, node.dict) else ir.DictAttrs(values)
        if isinstance(node, ir.For):
            variable = self._definitions(node.var, mutate=True)
            begin, end, step, body = [self.native.mutate(value) for value in (node.begin, node.end, node.step, node.body)]
            annotations = self._mutate_metadata(node.annotations)
            old = (node.var, node.begin, node.end, node.step, node.body, node.annotations)
            new = (variable, begin, end, step, body, annotations)
            return (
                node
                if all(_same(first, second) for first, second in zip(old, new))
                else ir.For(variable, begin, end, step, node.kind, body, annotations, node.loc)
            )
        if isinstance(node, ir.RegionStmt):
            args = self.native.mutate(node.args)
            parameters = self._definitions(node.body_params, mutate=True)
            body = self.native.mutate(node.body)
            attrs = self.native.mutate(node.attrs)
            results = self._definitions(node.result_vars, mutate=True)
            old = (node.args, node.body_params, node.body, node.attrs, node.result_vars)
            new = (args, parameters, body, attrs, results)
            return (
                node
                if all(_same(first, second) for first, second in zip(old, new))
                else ir.RegionStmt(node.op, args, parameters, attrs, body, results, node.loc)
            )
        if isinstance(node, tirx.Function):
            parameters = self._definitions(node.params, pattern=True, mutate=True)
            result_type, body, attrs = [self.native.mutate(value) for value in (node.ret_type, node.body, node.attrs)]
            old = (node.params, node.ret_type, node.body, node.attrs)
            new = (parameters, result_type, body, attrs)
            return (
                node
                if all(_same(first, second) for first, second in zip(old, new))
                else tirx.Function(parameters, body, result_type, attrs, node.loc)
            )
        if isinstance(node, ir.Call):
            result_type = self.native.mutate(node.ty)
            operation = node.op if isinstance(node.op, ir.Op) else self.native.mutate(node.op)
            arguments = self.native.mutate(node.args)
            attrs = self.native.mutate(node.attrs) if _call_attrs(node) else node.attrs
            type_arguments = self.native.mutate(node.ty_args) if node.ty_args else node.ty_args
            old = (node.ty, node.op, node.args, node.attrs, node.ty_args)
            new = (result_type, operation, arguments, attrs, type_arguments)
            return (
                node
                if all(_same(first, second) for first, second in zip(old, new))
                else ir.Call(operation, arguments, attrs, type_arguments, ty=result_type, loc=node.loc)
            )
        return self.native.default_mutate(node)


def structural_visit(root, callbacks):
    registered = _callbacks(callbacks)

    def dispatch(node, native):
        engine = _Engine(native)
        callback = _callback(node, registered)
        return callback(node, engine) if callback else engine.default_visit(node)

    return tvm_ffi.structural_visit(root, (tvm_ffi.Object, dispatch))


def structural_mutate(root, callbacks):
    registered = _callbacks(callbacks)

    def dispatch(node, native):
        engine = _Engine(native)
        callback = _callback(node, registered)
        return callback(node, engine) if callback else engine.default_mutate(node)

    return tvm_ffi.structural_mutate(root, (tvm_ffi.Object, dispatch))


def structural_walk(root, callbacks, *, order="post"):
    if order not in ("pre", "post"):
        raise ValueError(f"Unknown structural walk order: {order!r}")

    def adapt(callback):
        def visit(node, engine):
            if order == "pre":
                result = callback(node)
                return result if result is not None else engine.default_visit(node)
            result = engine.default_visit(node)
            return result if result is not None else callback(node)

        return visit

    return structural_visit(root, [(classes, adapt(callback)) for classes, callback in _callbacks(callbacks)])


def structural_map(root, callbacks, *, order="post"):
    if order not in ("pre", "post"):
        raise ValueError(f"Unknown structural map order: {order!r}")

    def adapt(callback):
        def mutate(node, engine):
            return engine.default_mutate(callback(node)) if order == "pre" else callback(engine.default_mutate(node))

        return mutate

    return structural_mutate(root, [(classes, adapt(callback)) for classes, callback in _callbacks(callbacks)])
