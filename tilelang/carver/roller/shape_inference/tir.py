from collections.abc import Mapping
from ..._native_analysis import StageRef, TensorView, post_order_visit
from ..._workload import substitute
from ..._constraints import const_bound_context
from tilelang._analyzer import create_analyzer
from tvm_ffi import structural_equal
from tvm_tirx import sym, ir
from tvm_tirx.ir import prim


class Statement:
    def __init__(self, block_analyzer, block: StageRef):
        self.block_analyzer = block_analyzer
        self.block = block
        # assume one tir block only has one output buffer
        self.dep_name = block_analyzer.get_output_buffers(block)[0].name
        self.dependent_region = _extract_dependent_region(block_analyzer, block)

        self.reverse_bound_inference = {}

    def make_reverse(self, input_name: str, input_iter: list[ir.Expr]):
        if len(self.block_analyzer.get_reduce_axis(self.block)) > 0:
            return None
        if len(self.dependent_region[input_name]) != 1:
            return None
        indices = self.dependent_region[input_name][0]
        iter_map_range = {_iter.var: _iter.dom for _iter in self.block_analyzer.get_spatial_axis(self.block)}
        iter_map_result = sym.detect_iter_map(
            indices,
            iter_map_range,
            check_level=sym.iter_affine_map.IterMapLevel.Surjective,
            simplify_trivial_iterators=False,
        )
        if len(iter_map_result.errors) > 0:
            return None
        results = sym.iter_affine_map.inverse_affine_iter_map(iter_map_result.indices, input_iter)
        output_indices = []
        for _iter in self.block_analyzer.get_spatial_axis(self.block):
            if _iter.var in results:
                output_indices.append(results[_iter.var])
            else:
                # not Bijective mapping case
                output_indices.append(ir.Var("undefined", ir.PrimType("int64")) % int(_iter.dom.extent))
        return output_indices


def _merge_two_bounds(x: sym.ConstIntBound, y: sym.ConstIntBound):
    return sym.ConstIntBound(min(x.min_value, y.min_value), max(x.max_value, y.max_value))


class TensorDepNode:
    """
    For tensor dependency analysis.
    """

    def __init__(self, name):
        self.name = name
        self._next = []
        self._prev = []

    def add_next(self, node):
        self._next.append(node)
        self.deduplicate(self._next)

    def add_prev(self, node):
        self._prev.append(node)
        self.deduplicate(self._prev)

    def deduplicate(self, lst):
        seen = set()
        lst[:] = [n for n in lst if not (n in seen or seen.add(n))]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class DependencyAnalysis:
    def __init__(self, deps):
        self.deps = deps
        # issue: duplicate name when we have two same ops.
        self.name2dep = self._construct_unique_name2dep(deps)
        self.mapping = {}  # name -> TensorDepNode

    def _construct_unique_name2dep(self, deps):
        """
        This is a workaround for the issue that we have two same ops' fuse case.
        See https://github.com/apache/tvm/issues/16433
        """
        _names: set = set()
        name2dep: Mapping = {}
        for dep in deps:
            output_buffer = dep.block_analyzer.get_output_buffers(dep.block)[0]
            base_name = output_buffer.name
            if base_name not in _names:
                _names.add(base_name)
            else:
                i = 1
                while f"{base_name}_{i}" in _names:
                    i += 1
                base_name = f"{base_name}_{i}"
                _names.add(base_name)
            name2dep[base_name] = dep
        return name2dep

    def get_or_create_node(self, name):
        if name not in self.mapping:
            self.mapping[name] = TensorDepNode(name)
        return self.mapping[name]

    def traverse_dependencies(self, compute):
        if isinstance(compute, Statement):
            node = self.get_or_create_node(compute.block_analyzer.get_output_buffers(compute.block)[0].name)
            # Loop through input tensors
            for input_buffer in compute.block_analyzer.get_input_buffers(compute.block):
                # Get the input node
                input_node = self.traverse_dependencies(input_buffer)
                input_node.add_next(node)
                node.add_prev(input_node)
        elif isinstance(compute, TensorView):
            node = self.get_or_create_node(compute.name)
        return node

    def analyze(self):
        # Starting point for traversal
        for _, compute in self.name2dep.items():
            self.traverse_dependencies(compute)

    def print_dependencies(self):
        for name, node in self.mapping.items():
            print(f"{name} depends on {', '.join([prev.name for prev in node._prev])}")

    def find_path_from_source(self, start_name, target_name):
        """
        Finds the path (if it exists) from a starting node (source) to a target node.
        Returns the path as a list of nodes.
        """
        visited = set()
        path = []
        if self._find_path_recursive(self.mapping[start_name], target_name, visited, path):
            return path
        return []

    def _find_path_recursive(self, current_node, target_name, visited, path):
        """
        Recursive helper function for find_path_from_source.
        """
        if current_node.name == target_name:
            path.append(current_node)
            return True

        if current_node.name in visited:
            return False

        visited.add(current_node.name)
        path.append(current_node)

        for next_node in current_node._next:
            if self._find_path_recursive(next_node, target_name, visited, path):
                return True

        path.pop()
        return False


class InputShapeInference:
    def __init__(self, deps: list[Statement]):
        self.deps = deps
        self.target_mapping = {}
        self.buffer_mapping = {}
        self.reduce_axes = []
        for dep in self.deps:
            for ax in dep.block_analyzer.get_reduce_axis(dep.block):
                self.reduce_axes.append(ax)
        self.dep_analysis = DependencyAnalysis(self.deps)
        self.dep_analysis.analyze()

    def construct_dependency_target(self, targets: tuple[str]):
        if targets in self.target_mapping:
            return self.target_mapping[targets]
        # should be buffer name instead of block name
        name2dep = {dep.block_analyzer.get_output_buffers(dep.block)[0].name: dep for dep in self.deps}
        mapping = {}
        input_vars = []
        for target in targets:
            vars = [iter.var for iter in name2dep[target].block_analyzer.get_spatial_axis(name2dep[target].block)]
            input_vars.append(vars)
            mapping[target] = [vars]
        ana = create_analyzer()

        for dep in self.deps:
            for name in dep.dependent_region:
                if name not in mapping:
                    continue
                dep_name = dep.dep_name
                indices = mapping[name][0]
                output_indices = dep.make_reverse(name, indices)
                if dep_name in targets:
                    continue
                if dep_name not in mapping:
                    mapping[dep_name] = [output_indices]
                elif not region_exist_in_list(output_indices, mapping[dep_name]):
                    mapping[dep_name].append(output_indices)

        for dep in reversed(self.deps):
            indices_list = mapping[dep.dep_name]
            ax_vars = [iter.var for iter in dep.block_analyzer.get_spatial_axis(dep.block)]
            for input_name, regions in dep.dependent_region.items():
                if input_name in targets:
                    continue
                if input_name not in mapping:
                    mapping[input_name] = []
                for indices in indices_list:
                    for region in regions:
                        vmap = {k: (prim.cast(k.ty.dtype, v) if v.ty.dtype != k.ty.dtype else v) for k, v in zip(ax_vars, indices)}
                        region = [ana.simplify(substitute(ax, vmap)) for ax in region]
                        if not region_exist_in_list(region, mapping[input_name]):
                            mapping[input_name].append(region)
        buffers = []
        for dep in self.deps:
            for buffer in dep.block_analyzer.get_buffers(dep.block):
                buffers.append(buffer)

        for buffer in buffers:
            self.buffer_mapping[buffer.name] = buffer

        self.target_mapping[targets] = input_vars, mapping
        return input_vars, mapping

    def infer(self, shape: dict[str, list[sym.ConstIntBound]], rstep: dict[str, int] = None, targets=None):
        if rstep is None:
            rstep = {}
        compute_targets = tuple(shape.keys())
        input_vars, mapping = self.construct_dependency_target(compute_targets)
        ana = create_analyzer()
        results = {}
        intermediate_bind = {}
        # The old update(..., override=True) chose the last bound for each
        # identity. Select those bounds before creating immutable premises;
        # appending old and new constraints would intersect them instead.
        bounds_by_var = {}
        for vars, bounds in zip(input_vars, shape.values()):
            for var, bound in zip(vars, bounds):
                bounds_by_var[var] = bound
        for ax in self.reduce_axes:
            # Reduction bounds retain the native domain's explicit begin.
            if ax.var.name in rstep:
                bound = sym.ConstIntBound(int(ax.dom.begin), int(ax.dom.begin + min(ax.dom.extent, rstep[ax.var.name]) - 1))
            else:
                bound = sym.ConstIntBound(int(ax.dom.begin), int(ax.dom.begin + ax.dom.extent - 1))
            bounds_by_var[ax.var] = bound
        ana.sync_constraint_context(const_bound_context(bounds_by_var))

        for name, regions in mapping.items():
            if targets is not None and name not in targets:
                continue
            if compute_targets[0:1] == compute_targets:
                (compute_target,) = compute_targets
                path = self.dep_analysis.find_path_from_source(name, compute_target)
                if len(path) > 2:
                    intermediate_nodes = path[1:-1]
                    for node in intermediate_nodes:
                        iters = mapping[node.name]
                        if len(iters) != len(regions) or len(iters) != 1:
                            continue
                        if len(*iters) != len(*regions):
                            break
                        regions = iters
                        intermediate_bind[name] = compute_target

                for region in regions:
                    bound = [ana.const_int_bound(indice) for indice in region]
                    if name in results:  # simply merge two bounds
                        bound = [_merge_two_bounds(x, y) for x, y in zip(results[name], bound)]
                    results[name] = bound
            else:
                for region in regions:
                    bound = [ana.const_int_bound(indice) for indice in region]
                    if name in results:  # simply merge two bounds
                        bound = [_merge_two_bounds(x, y) for x, y in zip(results[name], bound)]
                    results[name] = bound

        for name, bounds in results.items():
            results[name] = [c.max_value - c.min_value + 1 for c in bounds]
        return results, intermediate_bind

    def get_input_exprs(self, output_exprs):
        input_vars, mapping = self.construct_dependency_target(tuple(output_exprs.keys()))
        ana = create_analyzer()
        context = sym.ConstraintContext()
        for variable in dict.fromkeys(ax.var for ax in self.reduce_axes):
            context.append(sym.BindingConstraint(variable, ir.const(0, variable.ty.dtype)))
        ana.sync_constraint_context(context)
        vmap = {}
        for vars, exprs in zip(input_vars, output_exprs.values()):
            for var, expr in zip(vars, exprs):
                if expr.ty.dtype != var.ty.dtype:
                    expr = prim.cast(var.ty.dtype, expr)
                vmap[var] = expr
        result = {}

        for name, regions in mapping.items():
            region = regions[0]
            result[name] = [ana.simplify(substitute(index, vmap)) for index in region]
        return result


def region_exist_in_list(a, list) -> bool:
    def expr_is_same(a, b) -> bool:
        if isinstance(a, ir.IntImm) and isinstance(b, ir.IntImm):
            return a.value == b.value
        return structural_equal(a, b)

    def region_is_same(a, b) -> bool:
        return len(a) == len(b) and all(expr_is_same(indice_a, indice_b) for indice_a, indice_b in zip(a, b))

    return any([region_is_same(a, x) for x in list])


def walk_indice(expr):
    if not ir.is_prim_expr(expr):
        return None
    opaque = []
    post_order_visit(expr, lambda node: opaque.append(node) if isinstance(node, (ir.TensorLoad, ir.Call)) else None)
    return None if opaque else expr


def _extract_dependent_region(block_analyzer, block: StageRef) -> dict[str, list[ir.Expr]]:
    input_buffers = block_analyzer.get_input_buffers(block)
    dependent_region = {buffer.name: [] for buffer in input_buffers}

    def fvisit(x):
        if not isinstance(x, ir.TensorLoad):
            return
        if block_analyzer.sch.tensor_view(x.source).name not in dependent_region:
            return
        index = []
        for indice, shape_limit in zip(x.indices, x.source.ty.shape):
            expr = walk_indice(indice)
            if expr is None:
                expr = ir.Var("undefined", ir.PrimType("int64")) % shape_limit
            if isinstance(expr, ir.IntImm) and expr.value == 0 and isinstance(shape_limit, ir.IntImm) and int(shape_limit) == 1:
                """for tensor ir zero dim smplification case.
                for ax0, ax1, ax2 in T.grid(T.int64(1024), T.int64(1024), T.int64(1024)):
                    with T.block("T_dense"):
                        v0, v1, v2 = T.axis.remap("SSR", [ax0, ax1, ax2])
                        T.reads(A_reindex[T.int64(0), v0, v2], B_reindex[T.int64(0), v1, v2])
                        T.writes(T_dense_reindex[T.int64(0), v0, v1])
                        with T.init():
                            T_dense_reindex[T.int64(0), v0, v1] = T.float16(0)
                        T_dense_reindex[T.int64(0), v0, v1] = T_dense_reindex[T.int64(0), v0, v1] + A_reindex[T.int64(0), v0, v2] * B_reindex[T.int64(0), v1, v2]
                For example, the T_dense_reindex has three dims, however there're only two spatial loops.
                """
                continue
            index.append(expr)
        if not region_exist_in_list(index, dependent_region[block_analyzer.sch.tensor_view(x.source).name]):
            dependent_region[block_analyzer.sch.tensor_view(x.source).name].append(index)

    stmt = block_analyzer.sch.get(block)
    post_order_visit(stmt.spec.body, fvisit=fvisit)
    return dependent_region


def get_analyzer_by_tir(block_analyzer, args) -> InputShapeInference:
    deps = [Statement(block_analyzer, block) for block in args]

    return InputShapeInference(deps)
