"""TileLang Relax optimization pipeline."""

from itertools import product
import logging
import math
from tilelang import tvm as tvm
from tvm import s_tir, relax, tirx as tir
from tvm.s_tir import dlight as dl
from tilelang.schedule.templates import default_schedule_rules
from tilelang.graph.schedule import canonicalize_scheduled_ir

logger = logging.getLogger(__name__)

_MAX_FUSED_SHARED_MEMORY_BYTES = 256 * 1024


class _FusionTilePlan:
    def __init__(
        self,
        node_tiles: dict["CallTirNode", list[int]],
        grid_size: int,
        block_orders: dict["CallTirNode", tir.PrimExpr],
    ):
        self.node_tiles = node_tiles
        self.grid_size = grid_size
        self.block_orders = block_orders


def _node_key(node: "CallTirNode") -> int:
    return id(node.outputs[0].var) if node.outputs else id(node)


def _node_output_name(node: "CallTirNode") -> str:
    return _var_name_hint(node.outputs[0].var) if node.outputs else node.name


def _remap_tile_plan(
    tile_plan: _FusionTilePlan,
    nodes: list["CallTirNode"],
) -> _FusionTilePlan:
    node_by_output = {_node_output_name(node): node for node in nodes}
    node_tiles = {
        node_by_output[_node_output_name(old_node)]: tile
        for old_node, tile in tile_plan.node_tiles.items()
        if _node_output_name(old_node) in node_by_output
    }
    block_orders = {
        node_by_output[_node_output_name(old_node)]: block_order
        for old_node, block_order in tile_plan.block_orders.items()
        if _node_output_name(old_node) in node_by_output
    }
    return _FusionTilePlan(node_tiles, tile_plan.grid_size, block_orders)


class CallTirNode:
    def __init__(self, name, inputs, outputs, call_tir):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.call_tir = call_tir
        self.prim_func: tir.PrimFunc | None = None
        self.tir_info: _TIRNodeInfo | None = None

    def attach_prim_func(self, func: tir.PrimFunc):
        self.prim_func = func
        self.tir_info = _analyze_tir_node(func)
        for idx, edge in enumerate(self.inputs):
            if idx < len(self.tir_info.input_buffers):
                edge.bind_buffer(self.tir_info.input_buffers[idx])
        for idx, edge in enumerate(self.outputs):
            if idx < len(self.tir_info.output_buffers):
                edge.bind_buffer(self.tir_info.output_buffers[idx])
        return self

    def get_space_dim(self) -> list[int]:
        return self.tir_info.output_shape() if self.tir_info is not None else []

    def candidate_axis_tiles(self) -> list[list[int]]:
        shape = self.get_space_dim()
        constraints = self.tir_info.output_axis_tile_constraints() if self.tir_info is not None else [None] * len(shape)
        axis_tiles = []
        for extent, constraint in zip(shape, constraints):
            if constraint is None:
                axis_tiles.append(self._candidate_axis_tiles(extent))
            else:
                axis_tiles.append([max(1, min(int(constraint), int(extent)))])
        return axis_tiles

    def is_valid_tile(self, tile: list[int]) -> bool:
        if self.tir_info is None:
            return True
        return self.tir_info.is_valid_output_tile(tile)

    def propogate_inputs(self, tile: list[int]) -> list[list[int]]:
        if self.tir_info is None:
            return []
        return self.tir_info.propagate_inputs(tile)

    def grid_size(self, tile: list[int]) -> int:
        shape = self.get_space_dim()
        return self._tile_grid_size(shape, tile)

    @staticmethod
    def _all_factors(n: int) -> list[int]:
        factors = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.append(i)
                if i * i != n:
                    factors.append(n // i)
        return sorted(factors)

    @classmethod
    def _candidate_axis_tiles(cls, extent: int) -> list[int]:
        preferred = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        candidates = {x for x in preferred if x <= extent and extent % x == 0}
        candidates.update(cls._all_factors(extent))
        return sorted(candidates)

    @staticmethod
    def _tile_grid_size(shape: list[int], tile: list[int]) -> int:
        if len(shape) != len(tile):
            return 0
        result = 1
        for extent, tile_extent in zip(shape, tile):
            if tile_extent <= 0:
                return 0
            result *= (extent + tile_extent - 1) // tile_extent
        return int(result)

    def block_infer(
        self,
        tile_map: dict["CallTirNode", list[int]],
        block_expr: tir.PrimExpr,
        connecting_edges: list[tuple],
    ) -> list[tir.PrimExpr | None] | None:
        if self.tir_info is None or self not in tile_map:
            return None
        input_origins = self.tir_info.propagate_input_origins(tile_map[self], block_expr)
        if input_origins is None:
            return None
        result: list[tir.PrimExpr | None] = []
        for edge, input_origin in zip(self.inputs, input_origins):
            producer = _find_edge_producer(edge, connecting_edges)
            if producer is None:
                result.append(None)
                continue
            if input_origin is None or producer not in tile_map:
                return None
            result.append(
                _encode_block_expr(
                    producer.get_space_dim(),
                    tile_map[producer],
                    input_origin,
                )
            )
        return result


class VarEdge:
    def __init__(self, name, var):
        self.name = name
        self.var = var
        self.buffer: tir.Buffer | None = None
        self.shape = self._shape_from_relax_var(var)
        self.dtype = self._dtype_from_relax_var(var)

    def bind_buffer(self, buffer: tir.Buffer):
        self.buffer = buffer
        self.shape = [_as_const_int(dim) or int(dim) for dim in buffer.shape]
        self.dtype = buffer.dtype
        return self

    @staticmethod
    def _shape_from_relax_var(var) -> list[int] | None:
        sinfo = getattr(var, "struct_info_", None)
        if sinfo is None:
            sinfo = getattr(var, "struct_info", None)
        shape = getattr(sinfo, "shape", None)
        values = getattr(shape, "values", None)
        if values is None:
            return None
        result = []
        for value in values:
            int_value = _as_const_int(value)
            if int_value is None:
                return None
            result.append(int_value)
        return result

    @staticmethod
    def _dtype_from_relax_var(var) -> str | None:
        sinfo = getattr(var, "struct_info_", None)
        if sinfo is None:
            sinfo = getattr(var, "struct_info", None)
        dtype = getattr(sinfo, "dtype", None)
        return None if dtype is None else str(dtype)


def _same_var(lhs, rhs) -> bool:
    if lhs is rhs:
        return True
    if lhs is None or rhs is None:
        return False
    return hasattr(lhs, "same_as") and lhs.same_as(rhs)


def _var_name_hint(var) -> str:
    name = getattr(var, "name_hint", None)
    if name is None:
        name = getattr(var, "name", None)
    return name if name is not None else "var"


class VarEdgeMap:
    """Map Relax vars to graph edges without relying on non-unique name_hint."""

    def __init__(self):
        self._items = []
        self._used_names = set()
        self._name_counts = {}

    def clear(self):
        self._items.clear()
        self._used_names.clear()
        self._name_counts.clear()

    def _unique_name(self, var) -> str:
        base = _var_name_hint(var)
        count = self._name_counts.get(base, 0)
        while True:
            name = base if count == 0 else f"{base}_{count}"
            count += 1
            if name not in self._used_names:
                self._name_counts[base] = count
                self._used_names.add(name)
                return name

    def get(self, var):
        for key, edge in self._items:
            if _same_var(key, var):
                return edge
        return None

    def add(self, var) -> VarEdge:
        edge = self.get(var)
        if edge is not None:
            return edge
        edge = VarEdge(self._unique_name(var), var)
        self._items.append((var, edge))
        return edge


class GraphManager:
    """Build and own the Relax call_tir dependency graph."""

    def __init__(self, mod: tvm.IRModule):
        self.mod = mod
        self.edges = VarEdgeMap()
        self.nodes: list[CallTirNode] = []
        self.connecting_edges: list[tuple] = []
        self.topo_sorted_nodes: list[CallTirNode] = []
        self.main_binding_by_var = []
        self.graph_output_vars = []
        self._build()

    def _build(self) -> None:
        main_func = self.mod["main"]
        main_bindings = main_func.body.blocks[0].bindings
        self.graph_output_vars = self._collect_output_vars(main_func.body.body)
        for binding in main_bindings:
            output_edge = self.edges.add(binding.var)
            self.main_binding_by_var.append((binding.var, binding))
            if not _is_call_tir(binding.value):
                continue
            call_tir = binding.value
            call_tir_name = call_tir.args[0].name_hint
            call_tir_args = _call_tir_arg_fields(call_tir)
            inputs = []
            for arg in call_tir_args:
                if isinstance(arg, tvm.tirx.Var):
                    inputs.append(self.edges.add(arg))
                else:
                    inputs.append(VarEdge(_var_name_hint(arg), arg))
            node = CallTirNode(call_tir_name, inputs, [output_edge], call_tir)
            node.attach_prim_func(self.mod[call_tir_name])
            self.nodes.append(node)

        for node in self.nodes:
            for output in node.outputs:
                for other_node in self.nodes:
                    if any(_same_var(output.var, input_edge.var) for input_edge in other_node.inputs):
                        self.connecting_edges.append((node, output, other_node))
        self.graph_output_vars = self._expand_output_aliases(self.graph_output_vars)
        self.topo_sorted_nodes = self.topo_sort(self.nodes, self.connecting_edges)

    def _collect_output_vars(self, expr) -> list[relax.Var]:
        outputs = []

        def append(var):
            if not any(_same_var(var, existing) for existing in outputs):
                outputs.append(var)

        def visit(value):
            if isinstance(value, relax.Var):
                append(value)
                return
            if isinstance(value, relax.Tuple):
                for field in value.fields:
                    visit(field)
                return
            if isinstance(value, relax.TupleGetItem):
                visit(value.tuple_value)

        visit(expr)
        return outputs

    def _find_binding_value(self, var):
        for binding_var, binding in self.main_binding_by_var:
            if _same_var(binding_var, var):
                return getattr(binding, "value", None)
        return None

    def _expand_output_aliases(self, vars_to_expand: list[relax.Var]) -> list[relax.Var]:
        outputs = []
        queue = list(vars_to_expand)

        def append(var):
            if not any(_same_var(var, existing) for existing in outputs):
                outputs.append(var)

        while queue:
            var = queue.pop(0)
            append(var)
            value = self._find_binding_value(var)
            if isinstance(value, relax.Tuple):
                for field in value.fields:
                    if isinstance(field, relax.Var):
                        queue.append(field)
            elif isinstance(value, relax.TupleGetItem) and isinstance(value.tuple_value, relax.Var):
                queue.append(value.tuple_value)
            elif isinstance(value, relax.Var):
                queue.append(value)
        return outputs

    def has_fusion_edges(self) -> bool:
        return bool(self.connecting_edges)

    def is_graph_output(self, edge: VarEdge) -> bool:
        return any(_same_var(edge.var, output_var) for output_var in self.graph_output_vars)

    def consumers_for(self, node: CallTirNode, output_edge: VarEdge) -> list[CallTirNode]:
        return [dst_node for src_node, edge, dst_node in self.connecting_edges if src_node is node and _same_var(edge.var, output_edge.var)]

    def edges_for(self, nodes_to_fuse: list[CallTirNode]) -> list[tuple]:
        node_set = set(nodes_to_fuse)
        return [edge for edge in self.connecting_edges if edge[0] in node_set and edge[2] in node_set]

    def topo_sort(
        self,
        nodes_to_sort: list[CallTirNode],
        connecting_edges: list[tuple] | None = None,
    ) -> list[CallTirNode]:
        if connecting_edges is None:
            connecting_edges = self.edges_for(nodes_to_sort)
        topo_sorted_nodes = []
        in_degree = {node: 0 for node in nodes_to_sort}
        adj_list = {node: [] for node in nodes_to_sort}
        for src_node, _, dst_node in connecting_edges:
            if src_node not in adj_list or dst_node not in in_degree:
                continue
            adj_list[src_node].append(dst_node)
            in_degree[dst_node] += 1
        zero_in_degree_stack = [node for node in nodes_to_sort if in_degree[node] == 0]
        while zero_in_degree_stack:
            curr_node = zero_in_degree_stack.pop()
            topo_sorted_nodes.append(curr_node)
            for neighbor in adj_list[curr_node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree_stack.append(neighbor)
        if len(topo_sorted_nodes) != len(nodes_to_sort):
            raise RuntimeError("Topological sort failed: Cycle detected in the CallTir graph!")
        return topo_sorted_nodes


def _buffer_same(lhs: tir.Buffer, rhs: tir.Buffer) -> bool:
    return lhs.same_as(rhs)


def _same_prim_expr(lhs: tir.PrimExpr, rhs: tir.PrimExpr) -> bool:
    return tvm.ir.structural_equal(lhs, rhs) or str(lhs) == str(rhs)


def _int_imm_like(expr, value: int) -> tir.IntImm:
    dtype = getattr(expr, "dtype", "int32")
    if not isinstance(dtype, str) or not (dtype.startswith("int") or dtype.startswith("uint")):
        dtype = "int32"
    return tir.IntImm(dtype, value)


def _as_const_int(expr):
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    try:
        return int(expr)
    except Exception:
        return None


def _buffer_shape(buffer: tir.Buffer) -> list[int]:
    shape = []
    for dim in buffer.shape:
        value = _as_const_int(dim)
        if value is None:
            raise RuntimeError(f"Dynamic buffer shape is not supported in fusion tile search: {buffer}")
        shape.append(value)
    return shape


def _decode_block_origin(
    shape: list[int],
    tile: list[int],
    block_expr: tir.PrimExpr,
) -> list[tir.PrimExpr] | None:
    if len(shape) != len(tile):
        return None
    origins_reversed = []
    for extent, tile_extent in zip(reversed(shape), reversed(tile)):
        tile_extent = int(tile_extent)
        num_block = (int(extent) + tile_extent - 1) // tile_extent
        if num_block <= 0:
            return None
        num_block_expr = _int_imm_like(block_expr, num_block)
        tile_extent_expr = _int_imm_like(block_expr, tile_extent)
        origins_reversed.append((block_expr % num_block_expr) * tile_extent_expr)
        block_expr = block_expr // num_block_expr
    return list(reversed(origins_reversed))


def _encode_block_expr(
    shape: list[int],
    tile: list[int],
    origins: list[tir.PrimExpr],
) -> tir.PrimExpr:
    analyzer = tvm.arith.Analyzer()
    zero = _int_imm_like(origins[0], 0) if origins else tir.IntImm("int64", 0)
    block_expr: tir.PrimExpr = zero
    for extent, tile_extent, origin in zip(shape, tile, origins):
        tile_extent = int(tile_extent)
        num_block = (int(extent) + tile_extent - 1) // tile_extent
        origin = analyzer.simplify(origin)
        num_block_expr = _int_imm_like(origin, num_block)
        tile_extent_expr = _int_imm_like(origin, tile_extent)
        block_expr = analyzer.simplify(block_expr * num_block_expr + tir.Max(origin // tile_extent_expr, _int_imm_like(origin, 0)))
    return analyzer.simplify(block_expr)


def _lookup_buffer_tile(buffer: tir.Buffer, tile_map: dict[tir.Buffer, list[int]]):
    for key, tile in tile_map.items():
        if _buffer_same(key, buffer):
            return tile
    return None


def _set_buffer_tile(buffer: tir.Buffer, tile: list[int], tile_map: dict[tir.Buffer, list[int]]):
    for key, existing in list(tile_map.items()):
        if _buffer_same(key, buffer):
            tile_map[key] = [max(a, b) for a, b in zip(existing, tile)] if len(existing) == len(tile) else list(tile)
            return
    tile_map[buffer] = list(tile)


def _lookup_buffer_origin(buffer: tir.Buffer, origin_map):
    for key, origin in origin_map.items():
        if _buffer_same(key, buffer):
            return origin
    return None


def _set_buffer_origin(buffer: tir.Buffer, origin: list[tir.PrimExpr], origin_map) -> bool:
    analyzer = tvm.arith.Analyzer()
    for key, existing in list(origin_map.items()):
        if not _buffer_same(key, buffer):
            continue
        if len(existing) != len(origin):
            return False
        simplified = [analyzer.simplify(value) for value in origin]
        if not all(_same_prim_expr(analyzer.simplify(lhs), analyzer.simplify(rhs)) for lhs, rhs in zip(existing, simplified)):
            return False
        origin_map[key] = simplified
        return True
    origin_map[buffer] = [analyzer.simplify(value) for value in origin]
    return True


class _TIRSBlockInfo:
    def __init__(self, block: tir.SBlock):
        self.block = block
        self.name = block.name_hint
        self.reads = list(block.reads)
        self.writes = list(block.writes)
        self.spatial_vars = [iter_var.var for iter_var in block.iter_vars if iter_var.iter_type == tir.IterVar.DataPar]
        self.reduce_vars = [iter_var.var for iter_var in block.iter_vars if iter_var.iter_type == tir.IterVar.CommReduce]
        self.spatial_extents = {
            iter_var.var: _as_const_int(iter_var.dom.extent) for iter_var in block.iter_vars if iter_var.iter_type == tir.IterVar.DataPar
        }
        self.reduce_extents = {
            iter_var.var: _as_const_int(iter_var.dom.extent) for iter_var in block.iter_vars if iter_var.iter_type == tir.IterVar.CommReduce
        }

    def writes_buffer(self, buffer: tir.Buffer) -> bool:
        return any(_buffer_same(region.buffer, buffer) for region in self.writes)

    def _vars_in_region_dim(self, region_range) -> list[tir.Var]:
        vars_seen = []
        for expr in (region_range.min, region_range.extent):
            for var in _vars_in_expr(expr):
                if not _tir_var_in(var, vars_seen):
                    vars_seen.append(var)
        return vars_seen

    def _spatial_tile_for_write(self, buffer: tir.Buffer, needed_tile: list[int]) -> dict[tir.Var, int]:
        result = {}
        write_region = None
        for region in self.writes:
            if _buffer_same(region.buffer, buffer):
                write_region = region
                break
        if write_region is not None:
            for dim, region_range in enumerate(write_region.region):
                if dim >= len(needed_tile):
                    break
                vars_seen = self._vars_in_region_dim(region_range)
                spatial_vars = [var for var in vars_seen if _tir_var_in(var, self.spatial_vars)]
                if len(spatial_vars) == 1:
                    result[spatial_vars[0]] = needed_tile[dim]
        for idx, var in enumerate(self.spatial_vars):
            if var not in result:
                extent = self.spatial_extents.get(var)
                fallback = needed_tile[idx] if idx < len(needed_tile) else extent
                result[var] = int(fallback if fallback is not None else 1)
        return result

    def _var_tile(self, var: tir.Var, spatial_tile: dict[tir.Var, int]) -> int | None:
        for key, value in spatial_tile.items():
            if _same_var(key, var):
                return value
        for key, value in self.reduce_extents.items():
            if _same_var(key, var):
                return int(value if value is not None else 1)
        return None

    def read_tile(self, region: tir.BufferRegion, spatial_tile: dict[tir.Var, int]) -> list[int]:
        shape = _buffer_shape(region.buffer)
        tile = []
        for dim, region_range in enumerate(region.region):
            dim_shape = shape[dim] if dim < len(shape) else 1
            extent = _as_const_int(region_range.extent)
            vars_seen = self._vars_in_region_dim(region_range)
            var_tiles = [self._var_tile(var, spatial_tile) for var in vars_seen]
            var_tiles = [value for value in var_tiles if value is not None]
            if var_tiles:
                value = max(var_tiles)
            elif extent is not None:
                value = extent
            else:
                value = dim_shape
            tile.append(max(1, min(int(value), int(dim_shape))))
        return tile

    def propagate_reads(self, write_buffer: tir.Buffer, needed_tile: list[int]) -> list[tuple[tir.Buffer, list[int]]]:
        spatial_tile = self._spatial_tile_for_write(write_buffer, needed_tile)
        return [(region.buffer, self.read_tile(region, spatial_tile)) for region in self.reads]

    def _region_for_buffer(self, regions, buffer: tir.Buffer) -> tir.BufferRegion | None:
        for region in regions:
            if _buffer_same(region.buffer, buffer):
                return region
        return None

    def _spatial_origin_for_write(
        self,
        buffer: tir.Buffer,
        write_origin: list[tir.PrimExpr],
    ) -> dict[tir.Var, tir.PrimExpr] | None:
        result: dict[tir.Var, tir.PrimExpr] = {}
        write_region = self._region_for_buffer(self.writes, buffer)
        if write_region is None:
            return None
        for dim, region_range in enumerate(write_region.region):
            if dim >= len(write_origin):
                break
            vars_seen = self._vars_in_region_dim(region_range)
            spatial_vars = [var for var in vars_seen if _tir_var_in(var, self.spatial_vars)]
            if len(spatial_vars) == 1:
                result[spatial_vars[0]] = write_origin[dim]
            elif len(spatial_vars) > 1:
                return None
        for var in self.spatial_vars:
            if var not in result:
                result[var] = _int_imm_like(var, 0)
        return result

    def _var_origin(
        self,
        var: tir.Var,
        spatial_origin: dict[tir.Var, tir.PrimExpr],
    ) -> tir.PrimExpr | None:
        for key, value in spatial_origin.items():
            if _same_var(key, var):
                return value
        for key in self.reduce_extents:
            if _same_var(key, var):
                return _int_imm_like(var, 0)
        return None

    def read_origin(
        self,
        region: tir.BufferRegion,
        spatial_origin: dict[tir.Var, tir.PrimExpr],
    ) -> list[tir.PrimExpr] | None:
        analyzer = tvm.arith.Analyzer()
        origin = []
        for region_range in region.region:
            dim_origin = region_range.min
            substitutions = {}
            for var in _vars_in_expr(dim_origin):
                var_origin = self._var_origin(var, spatial_origin)
                if var_origin is None:
                    return None
                substitutions[var] = var_origin
            if substitutions:
                dim_origin = tir.stmt_functor.substitute(dim_origin, substitutions)
            origin.append(analyzer.simplify(dim_origin))
        return origin

    def propagate_read_origins(
        self,
        write_buffer: tir.Buffer,
        write_origin: list[tir.PrimExpr],
    ) -> list[tuple[tir.Buffer, list[tir.PrimExpr]]] | None:
        spatial_origin = self._spatial_origin_for_write(write_buffer, write_origin)
        if spatial_origin is None:
            return None
        results = []
        for region in self.reads:
            read_origin = self.read_origin(region, spatial_origin)
            if read_origin is None:
                return None
            results.append((region.buffer, read_origin))
        return results


class _TIRNodeInfo:
    def __init__(self, func: tir.PrimFunc):
        self.func = func
        self.param_buffers = [func.buffer_map[param] for param in func.params if param in func.buffer_map]
        if len(self.param_buffers) < 1:
            raise RuntimeError("Expected at least one buffer parameter in call_tir PrimFunc")
        self.input_buffers = self.param_buffers[:-1]
        self.output_buffers = [self.param_buffers[-1]]
        self.blocks = self._collect_blocks()

    def _collect_blocks(self) -> list[_TIRSBlockInfo]:
        blocks = []

        def visitor(node):
            if not isinstance(node, tir.SBlockRealize):
                return
            block = node.block
            if block.name_hint in ("root", "tilelang_root") or len(block.iter_vars) == 0:
                return
            if len(block.writes) == 0:
                return
            blocks.append(_TIRSBlockInfo(block))

        tir.stmt_functor.post_order_visit(self.func.body, visitor)
        return blocks

    def output_shape(self) -> list[int]:
        return _buffer_shape(self.output_buffers[0])

    def _reduction_blocks(self) -> list[_TIRSBlockInfo]:
        return [block for block in self.blocks if block.reduce_vars]

    def _max_nonunit_reduction_spatial_rank(self) -> int:
        rank = 0
        for block in self._reduction_blocks():
            nonunit_rank = 0
            for var in block.spatial_vars:
                extent = block.spatial_extents.get(var)
                if extent is None or int(extent) != 1:
                    nonunit_rank += 1
            rank = max(rank, nonunit_rank)
        return rank

    def output_axis_tile_constraints(self) -> list[int | None]:
        """Return fixed per-axis output tile constraints for fragile reductions.

        Row-wise reductions with a broadcast epilogue (softmax, RMSNorm, etc.)
        need the broadcast/reduced axis to stay whole inside one CTA.  The
        current GeneralReduction fixed-config path is also only reliable for
        one reduction row per CTA, so constrain the leading reduction-spatial
        axes to 1 and the trailing broadcast axes to their full extent.
        """
        shape = self.output_shape()
        constraints: list[int | None] = [None] * len(shape)
        if not self._reduction_blocks():
            return constraints

        spatial_rank = self._max_nonunit_reduction_spatial_rank()
        if spatial_rank <= 0 or len(shape) <= spatial_rank:
            return constraints

        for axis in range(spatial_rank):
            constraints[axis] = 1
        for axis in range(spatial_rank, len(shape)):
            constraints[axis] = shape[axis]
        return constraints

    def is_valid_output_tile(self, tile: list[int]) -> bool:
        if len(tile) != len(self.output_shape()):
            return False
        constraints = self.output_axis_tile_constraints()
        for tile_extent, constraint in zip(tile, constraints):
            if constraint is not None and int(tile_extent) != int(constraint):
                return False
        return True

    def propagate_inputs(self, output_tile: list[int]) -> list[list[int]]:
        needed: dict[tir.Buffer, list[int]] = {}
        _set_buffer_tile(self.output_buffers[0], output_tile, needed)
        for block in reversed(self.blocks):
            written_buffers = [region.buffer for region in block.writes]
            for write_buffer in written_buffers:
                write_tile = _lookup_buffer_tile(write_buffer, needed)
                if write_tile is None:
                    continue
                for read_buffer, read_tile in block.propagate_reads(write_buffer, write_tile):
                    _set_buffer_tile(read_buffer, read_tile, needed)
        results = []
        for buffer in self.input_buffers:
            tile = _lookup_buffer_tile(buffer, needed)
            results.append(tile if tile is not None else _buffer_shape(buffer))
        return results

    def propagate_input_origins(
        self,
        output_tile: list[int],
        block_expr: tir.PrimExpr,
    ) -> list[list[tir.PrimExpr] | None] | None:
        output_origin = _decode_block_origin(self.output_shape(), output_tile, block_expr)
        if output_origin is None:
            return None
        needed = {}
        if not _set_buffer_origin(self.output_buffers[0], output_origin, needed):
            return None
        for block in reversed(self.blocks):
            written_buffers = [region.buffer for region in block.writes]
            for write_buffer in written_buffers:
                write_origin = _lookup_buffer_origin(write_buffer, needed)
                if write_origin is None:
                    continue
                propagated = block.propagate_read_origins(write_buffer, write_origin)
                if propagated is None:
                    return None
                for read_buffer, read_origin in propagated:
                    if not _set_buffer_origin(read_buffer, read_origin, needed):
                        return None
        results: list[list[tir.PrimExpr] | None] = []
        for buffer in self.input_buffers:
            results.append(_lookup_buffer_origin(buffer, needed))
        return results


def _analyze_tir_node(func: tir.PrimFunc) -> _TIRNodeInfo:
    return _TIRNodeInfo(func)


def _find_edge_producer(edge: VarEdge, connecting_edges: list[tuple]) -> CallTirNode | None:
    for src_node, output_edge, _ in connecting_edges:
        if _same_var(edge.var, output_edge.var):
            return src_node
    return None


def _tir_var_in(var, candidates) -> bool:
    return any(_same_var(var, candidate) for candidate in candidates)


def _vars_in_expr(expr) -> list[tir.Var]:
    vars_seen = []

    def visitor(node):
        if isinstance(node, tir.Var) and not _tir_var_in(node, vars_seen):
            vars_seen.append(node)

    tir.stmt_functor.post_order_visit(expr, visitor)
    return vars_seen


class _SharedIntermediatePlan:
    def __init__(self, base_buffer, shared_buffer, origins, local_mins, name_hint=None):
        self.base_buffer = base_buffer
        self.shared_buffer = shared_buffer
        self.origins = list(origins)
        self.local_mins = list(local_mins)
        self.name_hint = name_hint if name_hint is not None else shared_buffer.name
        self.analyzer = tvm.arith.Analyzer()

    def localize_indices(self, indices):
        localized = []
        for index, origin, local_min in zip(indices, self.origins, self.local_mins):
            localized.append(self.analyzer.simplify(index - origin - local_min))
        return localized


@tir.functor.mutator
class _PrimitiveTIRBufferSubstitutor(tir.PyStmtExprMutator):
    """Python equivalent of the primitive-function lowering buffer substitutor."""

    def __init__(self, buffer_remap, prefix=None, protected_buffers=None, var_remap=None):
        super().__init__()
        self.buffer_remap = dict(buffer_remap)
        self.var_remap = {src.data: dst.data for src, dst in buffer_remap.items()}
        self.var_remap.update(var_remap or {})
        self.prefix = prefix
        self.protected_buffers = list(protected_buffers or [])
        self.defined_var_remap = {}

    def _new_var(self, var):
        if self.prefix is None:
            return var
        if var not in self.defined_var_remap:
            self.defined_var_remap[var] = tir.Var(f"{self.prefix}_{var.name}", var.dtype)
        return self.defined_var_remap[var]

    def _map_var(self, var):
        if var in self.var_remap:
            return self.var_remap[var]
        if var in self.defined_var_remap:
            return self.defined_var_remap[var]
        return var

    def _is_protected_buffer(self, buffer):
        return any(buffer.same_as(protected) for protected in self.protected_buffers)

    def _make_prefixed_buffer(self, buffer):
        strides = [self.visit_expr(stride) for stride in buffer.strides] if buffer.strides else None
        return tir.decl_buffer(
            shape=[self.visit_expr(dim) for dim in buffer.shape],
            dtype=buffer.dtype,
            name=f"{self.prefix}_{buffer.name}",
            strides=strides,
            elem_offset=self.visit_expr(buffer.elem_offset),
            scope=buffer.scope(),
            data_alignment=buffer.data_alignment,
            offset_factor=buffer.offset_factor,
            axis_separators=buffer.axis_separators,
            span=getattr(buffer, "span", None),
        )

    def _map_buffer(self, buffer):
        if buffer in self.buffer_remap:
            return self.buffer_remap[buffer]
        if buffer.data in self.var_remap:
            new_buffer = tir.decl_buffer(
                shape=[self.visit_expr(dim) for dim in buffer.shape],
                dtype=buffer.dtype,
                name=buffer.name,
                data=self.var_remap[buffer.data],
                strides=[self.visit_expr(stride) for stride in buffer.strides] if buffer.strides else None,
                elem_offset=self.visit_expr(buffer.elem_offset),
                scope=buffer.scope(),
                data_alignment=buffer.data_alignment,
                offset_factor=buffer.offset_factor,
                axis_separators=buffer.axis_separators,
                span=getattr(buffer, "span", None),
            )
            self.buffer_remap[buffer] = new_buffer
            return new_buffer
        if self.prefix is not None and not self._is_protected_buffer(buffer):
            new_buffer = self._make_prefixed_buffer(buffer)
            self.buffer_remap[buffer] = new_buffer
            self.var_remap[buffer.data] = new_buffer.data
            return new_buffer
        return buffer

    def _map_region(self, region):
        return [
            tvm.ir.Range.from_min_extent(
                self.visit_expr(item.min),
                self.visit_expr(item.extent),
            )
            for item in region
        ]

    def _map_buffer_region(self, buffer_region):
        return tir.BufferRegion(
            self._map_buffer(buffer_region.buffer),
            self._map_region(buffer_region.region),
        )

    def _map_match_buffer(self, match_buffer):
        return tir.MatchBufferRegion(
            self._map_buffer(match_buffer.buffer),
            self._map_buffer_region(match_buffer.source),
        )

    def visit_var_(self, op):
        return self._map_var(op)

    def visit_buffer_load_(self, op):
        return tir.BufferLoad(
            self._map_buffer(op.buffer),
            [self.visit_expr(idx) for idx in op.indices],
            self.visit_expr(op.predicate) if getattr(op, "predicate", None) else None,
            getattr(op, "span", None),
        )

    def visit_buffer_store_(self, op):
        return tir.BufferStore(
            self._map_buffer(op.buffer),
            self.visit_expr(op.value),
            [self.visit_expr(idx) for idx in op.indices],
            self.visit_expr(op.predicate) if getattr(op, "predicate", None) else None,
            getattr(op, "span", None),
        )

    def visit_call_(self, op):
        return tir.Call(
            op.dtype,
            op.op,
            [self.visit_expr(arg) for arg in op.args],
            getattr(op, "annotations", None),
            getattr(op, "span", None),
        )

    def visit_for_(self, op):
        old_var = op.loop_var
        new_var = self._new_var(old_var)
        return tir.For(
            new_var,
            self.visit_expr(op.min),
            self.visit_expr(op.extent),
            op.kind,
            self.visit_stmt(op.body),
            getattr(op, "thread_binding", None),
            getattr(op, "annotations", None),
            getattr(op, "step", None),
            getattr(op, "span", None),
        )

    def visit_sblock_(self, op):
        return tir.SBlock(
            op.iter_vars,
            [self._map_buffer_region(region) for region in op.reads],
            [self._map_buffer_region(region) for region in op.writes],
            op.name_hint if self.prefix is None else f"{self.prefix}_{op.name_hint}",
            self.visit_stmt(op.body),
            self.visit_stmt(op.init) if op.init is not None else None,
            [self._map_buffer(buffer) for buffer in op.alloc_buffers],
            [self._map_match_buffer(match_buffer) for match_buffer in op.match_buffers],
            getattr(op, "annotations", None),
            getattr(op, "span", None),
        )


def _is_call_tir(call) -> bool:
    return isinstance(call, relax.Call) and hasattr(call.op, "name") and call.op.name == "relax.call_tir"


def _call_tir_arg_fields(call: relax.Call) -> list:
    args = call.args[1]
    if isinstance(args, relax.Tuple):
        return list(args.fields)
    return [args]


def _call_tir_out_sinfos(call: relax.Call):
    return list(call.sinfo_args)


def _call_tir_tir_vars(call: relax.Call):
    if len(call.args) > 2:
        return call.args[2]
    return None


class GraphFuser:
    """Fuse a scheduled call_tir subgraph into one TIR PrimFunc."""

    def __init__(
        self,
        graph: GraphManager,
        mod: tvm.IRModule,
        tile_plan: _FusionTilePlan,
    ):
        self.graph = graph
        self.mod = mod
        self.tile_plan = tile_plan

    def _get_param_buffer(self, func: tir.PrimFunc, param_idx: int) -> tir.Buffer:
        if param_idx >= len(func.params):
            raise RuntimeError(f"PrimFunc has {len(func.params)} params, cannot access param {param_idx}")
        param = func.params[param_idx]
        if param not in func.buffer_map:
            raise RuntimeError(f"PrimFunc param {param} is not backed by a buffer")
        return func.buffer_map[param]

    def _thread_launch_signature(self, func: tir.PrimFunc) -> list[tuple[str, tir.PrimExpr]]:
        signature = []

        def visitor(node):
            if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
                thread_tag = node.node.thread_tag if hasattr(node.node, "thread_tag") else str(node.node)
                signature.append((thread_tag, node.value))
                return False
            if isinstance(node, tir.For) and node.thread_binding is not None:
                signature.append((node.thread_binding.thread_tag, node.extent))
            return True

        tir.stmt_functor.pre_order_visit(func.body, visitor)
        return signature

    def _thread_launch_vars(self, func: tir.PrimFunc) -> list[tuple[str, tir.Var]]:
        launch_vars = []

        def visitor(node):
            if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
                thread_tag = node.node.thread_tag if hasattr(node.node, "thread_tag") else str(node.node)
                var = node.node.var if hasattr(node.node, "var") else None
                if isinstance(var, tir.Var):
                    launch_vars.append((thread_tag, var))
                return False
            if isinstance(node, tir.For) and node.thread_binding is not None:
                launch_vars.append((node.thread_binding.thread_tag, node.loop_var))
            return True

        tir.stmt_functor.pre_order_visit(func.body, visitor)
        return launch_vars

    def _check_same_launch_threads(self, funcs: list[tir.PrimFunc]) -> None:
        if not funcs:
            return
        base = self._thread_launch_signature(funcs[0])
        if not base:
            raise RuntimeError("Failed to find launch_thread/thread_extent in the first TIR")
        for idx, func in enumerate(funcs[1:], start=1):
            curr = self._thread_launch_signature(func)
            if len(curr) != len(base):
                raise RuntimeError(f"Cannot fuse TIR #{idx}: launch_thread count mismatch ({len(curr)} vs {len(base)})")
            for (base_tag, base_extent), (curr_tag, curr_extent) in zip(base, curr):
                if base_tag != curr_tag or not _same_prim_expr(base_extent, curr_extent):
                    raise RuntimeError(
                        f"Cannot fuse TIRs with different launch_thread bindings: {base_tag}={base_extent} vs {curr_tag}={curr_extent}"
                    )

    def _find_tilelang_root(self, func: tir.PrimFunc) -> tir.SBlockRealize:
        roots = []

        def visitor(node):
            if isinstance(node, tir.SBlockRealize) and node.block.name_hint == "tilelang_root":
                roots.append(node)

        tir.stmt_functor.post_order_visit(func.body, visitor)
        if len(roots) != 1:
            raise RuntimeError(f"Expected exactly one tilelang_root block, but found {len(roots)}")
        return roots[0]

    def _as_seq(self, stmt: tir.Stmt) -> list[tir.Stmt]:
        if isinstance(stmt, tir.SeqStmt):
            return list(stmt.seq)
        return [stmt]

    def _var_in(self, var, candidates) -> bool:
        return any(_same_var(var, candidate) for candidate in candidates)

    def _is_tileop_call(self, expr, op_name: str) -> bool:
        return isinstance(expr, tir.Call) and hasattr(expr.op, "name") and expr.op.name == op_name

    def _lookup_buffer_remap(self, buffer: tir.Buffer, buffer_remap):
        for src, dst in buffer_remap.items():
            if _buffer_same(buffer, src):
                return dst
        return None

    def _buffer_in(self, buffer: tir.Buffer, buffers) -> bool:
        return any(_buffer_same(buffer, candidate) for candidate in buffers)

    def _unique_buffers(self, buffers) -> list[tir.Buffer]:
        unique = []
        for buffer in buffers:
            if not self._buffer_in(buffer, unique):
                unique.append(buffer)
        return unique

    def _sanitize_buffer_name(self, name: str) -> str:
        return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)

    def _make_unique_global_var(self, base_name: str) -> tvm.ir.GlobalVar:
        base_name = self._sanitize_buffer_name(base_name)
        if not base_name or base_name[0].isdigit():
            base_name = f"fused_{base_name}"
        base_name = base_name[:160]
        existing_names = {gv.name_hint for gv, _ in self.mod.functions.items()}
        name = base_name
        suffix = 0
        while name in existing_names:
            suffix += 1
            trimmed = base_name[: max(1, 150 - len(str(suffix)))]
            name = f"{trimmed}_{suffix}"
        return tvm.ir.GlobalVar(name)

    def _lookup_var_domain(self, var, domains):
        for key, value in domains.items():
            if _same_var(var, key):
                return value
        return None

    def _thread_launch_bindings(
        self,
        func: tir.PrimFunc,
    ) -> list[tuple[str, tir.Var, tir.PrimExpr]]:
        launch_bindings = []

        def visitor(node):
            if isinstance(node, tir.AttrStmt) and node.attr_key == "thread_extent":
                thread_tag = node.node.thread_tag if hasattr(node.node, "thread_tag") else str(node.node)
                var = node.node.var if hasattr(node.node, "var") else None
                if isinstance(var, tir.Var):
                    launch_bindings.append((thread_tag, var, node.value))
                return False
            if isinstance(node, tir.For) and node.thread_binding is not None:
                launch_bindings.append((node.thread_binding.thread_tag, node.loop_var, node.extent))
            return True

        tir.stmt_functor.pre_order_visit(func.body, visitor)
        return launch_bindings

    def _replace_block_idx_symbol(self, expr: tir.PrimExpr, block_var: tir.Var) -> tir.PrimExpr:
        substitutions = {}
        for var in _vars_in_expr(expr):
            if getattr(var, "name", None) == "block_idx":
                substitutions[var] = block_var
        if substitutions:
            expr = tir.stmt_functor.substitute(expr, substitutions)
        return tvm.arith.Analyzer().simplify(expr)

    def _collect_loop_domains(self, stmt: tir.Stmt):
        analyzer = tvm.arith.Analyzer()
        domains = {}

        def visitor(node):
            if isinstance(node, tir.For):
                min_value = node.min
                max_value = analyzer.simplify(node.min + node.extent - _int_imm_like(node.extent, 1))
                domains[node.loop_var] = tvm.arith.IntervalSet(min_value, max_value)

        tir.stmt_functor.post_order_visit(stmt, visitor)
        return domains

    def _collect_buffer_access_regions(self, stmt: tir.Stmt, target_buffer: tir.Buffer):
        regions = []

        def append_point(indices):
            extents = [_int_imm_like(index, 1) for index in indices]
            regions.append((list(indices), extents))

        def visitor(node):
            if (
                isinstance(node, tir.BufferStore)
                and _buffer_same(node.buffer, target_buffer)
                or isinstance(node, tir.BufferLoad)
                and _buffer_same(node.buffer, target_buffer)
            ):
                append_point(node.indices)
            elif self._is_tileop_call(node, "tl.tileop.region") and node.args:
                buffer_load = node.args[0]
                if isinstance(buffer_load, tir.BufferLoad) and _buffer_same(buffer_load.buffer, target_buffer):
                    indices = list(buffer_load.indices)
                    extents = list(node.args[2:])
                    if len(extents) == len(indices):
                        regions.append((indices, extents))

        tir.stmt_functor.post_order_visit(stmt, visitor)
        return regions

    def _make_shared_intermediate_buffer(
        self,
        base_buffer: tir.Buffer,
        name_hint: str,
        shape=None,
    ) -> tir.Buffer:
        elem_offset = tir.IntImm(
            base_buffer.elem_offset.dtype if hasattr(base_buffer.elem_offset, "dtype") else "int32",
            0,
        )
        actual_shape = list(shape if shape is not None else base_buffer.shape)
        kwargs = {
            "shape": actual_shape,
            "dtype": base_buffer.dtype,
            "name": self._sanitize_buffer_name(name_hint),
            "elem_offset": elem_offset,
            "scope": "shared.dyn",
            "data_alignment": base_buffer.data_alignment,
            "offset_factor": base_buffer.offset_factor,
            "axis_separators": base_buffer.axis_separators,
            "span": getattr(base_buffer, "span", None),
        }
        if shape is None and base_buffer.strides:
            kwargs["strides"] = list(base_buffer.strides)
        return tir.decl_buffer(**kwargs)

    def _estimate_bounds(self, expr, domains, analyzer):
        try:
            int_set = analyzer.int_set(expr, domains)
        except Exception:
            return None
        if int_set.is_everything() or int_set.is_nothing():
            return None
        return analyzer.simplify(int_set.min_value), analyzer.simplify(int_set.max_value)

    def _block_origin(self, expr, block_vars, domains, analyzer):
        substitutions = {}
        for var in _vars_in_expr(expr):
            if _tir_var_in(var, block_vars):
                continue
            domain = self._lookup_var_domain(var, domains)
            substitutions[var] = domain.min_value if domain is not None else _int_imm_like(var, 0)
        if substitutions:
            expr = tir.stmt_functor.substitute(expr, substitutions)
        return analyzer.simplify(expr)

    def _make_shared_intermediate_plan(
        self,
        func: tir.PrimFunc,
        root: tir.SBlockRealize,
        base_buffer: tir.Buffer,
        name_hint: str,
    ) -> _SharedIntermediatePlan:
        analyzer = tvm.arith.Analyzer()
        launch_bindings = self._thread_launch_bindings(func)
        block_vars = [var for thread_tag, var, _ in launch_bindings if thread_tag.startswith("blockIdx")]
        domains = self._collect_loop_domains(root.block.body)
        for _, var, extent in launch_bindings:
            domains[var] = tvm.arith.IntervalSet(
                _int_imm_like(extent, 0),
                analyzer.simplify(extent - _int_imm_like(extent, 1)),
            )

        access_regions = self._collect_buffer_access_regions(root.block.body, base_buffer)
        rank = len(base_buffer.shape)
        zero_origins = [_int_imm_like(dim, 0) for dim in base_buffer.shape]
        if not access_regions:
            shared_buffer = self._make_shared_intermediate_buffer(base_buffer, name_hint)
            return _SharedIntermediatePlan(base_buffer, shared_buffer, zero_origins, zero_origins, name_hint)

        first_indices, _ = access_regions[0]
        if len(first_indices) != rank:
            shared_buffer = self._make_shared_intermediate_buffer(base_buffer, name_hint)
            return _SharedIntermediatePlan(base_buffer, shared_buffer, zero_origins, zero_origins, name_hint)

        origins = [self._block_origin(index, block_vars, domains, analyzer) for index in first_indices]
        local_mins = []
        shared_shape = []

        for dim, base_extent in enumerate(base_buffer.shape):
            dim_min = None
            dim_max = None
            failed = False
            for indices, extents in access_regions:
                if len(indices) != rank or len(extents) != rank:
                    failed = True
                    break
                one = _int_imm_like(extents[dim], 1)
                region_min = analyzer.simplify(indices[dim] - origins[dim])
                region_max = analyzer.simplify(indices[dim] + extents[dim] - one - origins[dim])
                min_bounds = self._estimate_bounds(region_min, domains, analyzer)
                max_bounds = self._estimate_bounds(region_max, domains, analyzer)
                if min_bounds is None or max_bounds is None:
                    failed = True
                    break
                candidate_min = min_bounds[0]
                candidate_max = max_bounds[1]
                dim_min = candidate_min if dim_min is None else analyzer.simplify(tir.Min(dim_min, candidate_min))
                dim_max = candidate_max if dim_max is None else analyzer.simplify(tir.Max(dim_max, candidate_max))

            if failed or dim_min is None or dim_max is None:
                origins[dim] = _int_imm_like(base_extent, 0)
                local_mins.append(_int_imm_like(base_extent, 0))
                shared_shape.append(base_extent)
                continue

            dim_extent = analyzer.simplify(dim_max - dim_min + _int_imm_like(dim_max, 1))
            const_dim_extent = _as_const_int(dim_extent)
            const_base_extent = _as_const_int(base_extent)
            if const_dim_extent is not None and (
                const_dim_extent <= 0 or (const_base_extent is not None and const_dim_extent > const_base_extent)
            ):
                origins[dim] = _int_imm_like(base_extent, 0)
                local_mins.append(_int_imm_like(base_extent, 0))
                shared_shape.append(base_extent)
                continue

            local_mins.append(dim_min)
            shared_shape.append(dim_extent)

        shared_buffer = self._make_shared_intermediate_buffer(base_buffer, name_hint, shared_shape)
        return _SharedIntermediatePlan(base_buffer, shared_buffer, origins, local_mins, name_hint)

    def _merge_shared_intermediate_layout(self, plans: list[_SharedIntermediatePlan]) -> None:
        if not plans:
            return
        analyzer = tvm.arith.Analyzer()
        rank = len(plans[0].base_buffer.shape)
        common_mins = []
        common_shape = []
        for dim in range(rank):
            dim_min = None
            dim_max = None
            for plan in plans:
                local_min = plan.local_mins[dim]
                local_extent = plan.shared_buffer.shape[dim]
                local_max = analyzer.simplify(local_min + local_extent - _int_imm_like(local_extent, 1))
                dim_min = local_min if dim_min is None else analyzer.simplify(tir.Min(dim_min, local_min))
                dim_max = local_max if dim_max is None else analyzer.simplify(tir.Max(dim_max, local_max))
            dim_extent = analyzer.simplify(dim_max - dim_min + _int_imm_like(dim_max, 1))
            common_mins.append(dim_min)
            common_shape.append(dim_extent)

        shared_buffer = self._make_shared_intermediate_buffer(
            plans[0].base_buffer,
            plans[0].name_hint,
            common_shape,
        )
        for plan in plans:
            plan.shared_buffer = shared_buffer
            plan.local_mins = list(common_mins)

    def _region_call_buffer(self, expr) -> tir.Buffer:
        if not self._is_tileop_call(expr, "tl.tileop.region") or not expr.args:
            return None
        buffer_load = expr.args[0]
        if not isinstance(buffer_load, tir.BufferLoad):
            return None
        return buffer_load.buffer

    def _region_call_details(self, expr):
        if not self._is_tileop_call(expr, "tl.tileop.region") or len(expr.args) < 2:
            return None
        buffer_load = expr.args[0]
        if not isinstance(buffer_load, tir.BufferLoad):
            return None
        return (
            buffer_load.buffer,
            list(buffer_load.indices),
            expr.args[1],
            list(expr.args[2:]),
        )

    def _copy_call_region_details(self, copy_call: tir.Call):
        if not self._is_tileop_call(copy_call, "tl.tileop.copy") or len(copy_call.args) < 2:
            return None
        src_region = self._region_call_details(copy_call.args[0])
        dst_region = self._region_call_details(copy_call.args[1])
        if src_region is None or dst_region is None:
            return None
        return src_region, dst_region

    def _full_buffer_region(self, buffer: tir.Buffer, access_type: str) -> tir.Call:
        access_mask = {"r": 1, "w": 2, "rw": 3}[access_type]
        zero_indices = [tir.IntImm("int32", 0) for _ in buffer.shape]
        return tir.Call(
            "handle",
            tir.op.Op.get("tl.tileop.region"),
            [
                tir.BufferLoad(buffer, zero_indices),
                tir.IntImm("int32", access_mask),
                *buffer.shape,
            ],
        )

    def _make_buffer_copy(self, src_buffer: tir.Buffer, dst_buffer: tir.Buffer) -> tir.Evaluate:
        return tir.Evaluate(
            tir.Call(
                "handle",
                tir.op.Op.get("tl.tileop.copy"),
                [
                    self._full_buffer_region(src_buffer, "r"),
                    self._full_buffer_region(dst_buffer, "w"),
                ],
            )
        )

    def _is_shared_buffer(self, buffer: tir.Buffer) -> bool:
        return buffer.scope().startswith("shared")

    def _is_region_full_buffer(self, buffer: tir.Buffer, indices, extents) -> bool:
        if len(indices) != len(buffer.shape) or len(extents) != len(buffer.shape):
            return False
        zero = tir.IntImm("int32", 0)
        return all(_same_prim_expr(index, zero) for index in indices) and all(
            _same_prim_expr(extent, shape) for extent, shape in zip(extents, buffer.shape)
        )

    def _same_buffer_shape_dtype(self, lhs: tir.Buffer, rhs: tir.Buffer) -> bool:
        return (
            lhs.dtype == rhs.dtype
            and len(lhs.shape) == len(rhs.shape)
            and all(_same_prim_expr(l_dim, r_dim) for l_dim, r_dim in zip(lhs.shape, rhs.shape))
        )

    def _access_mask_writes(self, access_mask) -> bool:
        value = _as_const_int(access_mask)
        return value is not None and (value & 2) != 0

    def _collect_redundant_shared_copy_aliases(self, stmts, alias_sources):
        write_counts = {}
        candidate_copies = {}

        def bump_write(buffer):
            if buffer is None:
                return
            existing = None
            for key in write_counts:
                if _buffer_same(key, buffer):
                    existing = key
                    break
            if existing is None:
                write_counts[buffer] = 1
            else:
                write_counts[existing] += 1

        def append_candidate(dst_buffer, src_buffer):
            existing = None
            for key in candidate_copies:
                if _buffer_same(key, dst_buffer):
                    existing = key
                    break
            if existing is None:
                candidate_copies[dst_buffer] = [src_buffer]
            else:
                candidate_copies[existing].append(src_buffer)

        def visitor(node):
            if isinstance(node, tir.BufferStore):
                bump_write(node.buffer)
                return
            if isinstance(node, tir.Call):
                region = self._region_call_details(node)
                if region is not None:
                    buffer, _, access_mask, _ = region
                    if self._access_mask_writes(access_mask):
                        bump_write(buffer)
                copy_details = self._copy_call_region_details(node)
                if copy_details is None:
                    return
                src_region, dst_region = copy_details
                src_buffer, src_indices, _, src_extents = src_region
                dst_buffer, dst_indices, _, dst_extents = dst_region
                if (
                    self._buffer_in(src_buffer, alias_sources)
                    and not self._buffer_in(dst_buffer, alias_sources)
                    and self._is_shared_buffer(src_buffer)
                    and self._is_shared_buffer(dst_buffer)
                    and src_buffer.scope() == dst_buffer.scope()
                    and self._same_buffer_shape_dtype(src_buffer, dst_buffer)
                    and self._is_region_full_buffer(src_buffer, src_indices, src_extents)
                    and self._is_region_full_buffer(dst_buffer, dst_indices, dst_extents)
                ):
                    append_candidate(dst_buffer, src_buffer)

        for stmt in stmts:
            tir.stmt_functor.post_order_visit(stmt, visitor)

        aliases = {}
        for dst_buffer, src_buffers in candidate_copies.items():
            if not src_buffers:
                continue
            first_src = src_buffers[0]
            if not all(_buffer_same(src_buffer, first_src) for src_buffer in src_buffers):
                continue
            write_count = 0
            for buffer, count in write_counts.items():
                if _buffer_same(buffer, dst_buffer):
                    write_count = count
                    break
            if write_count == len(src_buffers):
                aliases[dst_buffer] = first_src
        return aliases

    def _merge_buffer_aliases(self, *alias_maps):
        aliases = {}
        for alias_map in alias_maps:
            for dst_buffer, src_buffer in alias_map.items():
                aliases[dst_buffer] = self._lookup_buffer_remap(src_buffer, aliases) or src_buffer
        return aliases

    def _is_alias_copy(self, stmt: tir.Stmt, aliases) -> bool:
        if not isinstance(stmt, tir.Evaluate):
            return False
        details = self._copy_call_region_details(stmt.value)
        if details is None:
            return False
        src_region, dst_region = details
        src_buffer = src_region[0]
        dst_buffer = dst_region[0]
        alias_src = self._lookup_buffer_remap(dst_buffer, aliases)
        return (
            alias_src is not None
            and _buffer_same(src_buffer, alias_src)
            and self._is_shared_buffer(src_buffer)
            and self._is_shared_buffer(dst_buffer)
        )

    def _remove_alias_copies_from_stmt(self, stmt: tir.Stmt, aliases):
        if self._is_alias_copy(stmt, aliases):
            return None
        if isinstance(stmt, tir.SeqStmt):
            new_seq = [
                new_item for item in stmt.seq for new_item in [self._remove_alias_copies_from_stmt(item, aliases)] if new_item is not None
            ]
            if not new_seq:
                return None
            if len(new_seq) == 1:
                return new_seq[0]
            return tir.SeqStmt(new_seq, getattr(stmt, "span", None))
        if isinstance(stmt, tir.For):
            new_body = self._remove_alias_copies_from_stmt(stmt.body, aliases)
            if new_body is None:
                new_body = tir.Evaluate(tir.IntImm("int32", 0))
            elif isinstance(stmt.body, tir.SeqStmt) and not isinstance(new_body, tir.SeqStmt):
                annotations = getattr(stmt, "annotations", None)
                if annotations is not None and "num_stages" in annotations:
                    new_body = stmt.body
            return tir.For(
                stmt.loop_var,
                stmt.min,
                stmt.extent,
                stmt.kind,
                new_body,
                getattr(stmt, "thread_binding", None),
                getattr(stmt, "annotations", None),
                getattr(stmt, "step", None),
                getattr(stmt, "span", None),
            )
        if isinstance(stmt, tir.AttrStmt):
            new_body = self._remove_alias_copies_from_stmt(stmt.body, aliases)
            if new_body is None:
                new_body = tir.Evaluate(tir.IntImm("int32", 0))
            return tir.AttrStmt(stmt.node, stmt.attr_key, stmt.value, new_body, getattr(stmt, "span", None))
        if isinstance(stmt, tir.IfThenElse):
            then_case = self._remove_alias_copies_from_stmt(stmt.then_case, aliases)
            else_case = self._remove_alias_copies_from_stmt(stmt.else_case, aliases) if stmt.else_case is not None else None
            if then_case is None:
                then_case = tir.Evaluate(tir.IntImm("int32", 0))
            return tir.IfThenElse(
                stmt.condition,
                then_case,
                else_case,
                getattr(stmt, "span", None),
            )
        if isinstance(stmt, tir.Bind):
            # tirx.Bind is a flat statement (no body); nothing to recurse into.
            return stmt
        if isinstance(stmt, tir.SBlockRealize):
            block = stmt.block
            new_body = self._remove_alias_copies_from_stmt(block.body, aliases)
            if new_body is None:
                new_body = tir.Evaluate(tir.IntImm("int32", 0))
            new_block = tir.SBlock(
                block.iter_vars,
                block.reads,
                block.writes,
                block.name_hint,
                new_body,
                block.init,
                block.alloc_buffers,
                block.match_buffers,
                block.annotations,
                getattr(block, "span", None),
            )
            return tir.SBlockRealize(stmt.iter_values, stmt.predicate, new_block, getattr(stmt, "span", None))
        if isinstance(stmt, tir.SBlock):
            new_body = self._remove_alias_copies_from_stmt(stmt.body, aliases)
            if new_body is None:
                new_body = tir.Evaluate(tir.IntImm("int32", 0))
            return tir.SBlock(
                stmt.iter_vars,
                stmt.reads,
                stmt.writes,
                stmt.name_hint,
                new_body,
                stmt.init,
                stmt.alloc_buffers,
                stmt.match_buffers,
                stmt.annotations,
                getattr(stmt, "span", None),
            )
        return stmt

    def _remove_redundant_shared_to_shared_copies(
        self,
        stmts: list[tir.Stmt],
        dependency_aliases,
        alias_sources,
    ):
        cache_aliases = self._collect_redundant_shared_copy_aliases(stmts, alias_sources)
        aliases = self._merge_buffer_aliases(dependency_aliases, cache_aliases)
        if not aliases:
            return stmts, aliases
        new_stmts = []
        for stmt in stmts:
            new_stmt = self._remove_alias_copies_from_stmt(stmt, aliases)
            if new_stmt is not None:
                new_stmts.append(new_stmt)
        return new_stmts, aliases

    def _remap_buffer_obj(self, remap):
        return remap.shared_buffer if isinstance(remap, _SharedIntermediatePlan) else remap

    def _remap_buffer_indices(self, remap, indices):
        if isinstance(remap, _SharedIntermediatePlan):
            return remap.localize_indices(indices)
        return list(indices)

    def _replace_region_buffer(self, region: tir.Call, remap) -> tir.Call:
        buffer_load = region.args[0]
        new_load = tir.BufferLoad(
            self._remap_buffer_obj(remap),
            self._remap_buffer_indices(remap, list(buffer_load.indices)),
            getattr(buffer_load, "predicate", None),
            getattr(buffer_load, "span", None),
        )
        return tir.Call(
            region.dtype,
            region.op,
            [new_load, *list(region.args[1:])],
            getattr(region, "annotations", None),
            getattr(region, "span", None),
        )

    def _redirect_copy_regions(
        self,
        copy_call: tir.Call,
        promoted_output_remap,
        promoted_input_remap,
    ) -> tir.Call:
        args = list(copy_call.args)
        if len(args) >= 1:
            src_buffer = self._region_call_buffer(args[0])
            new_src_remap = self._lookup_buffer_remap(src_buffer, promoted_input_remap) if src_buffer is not None else None
            if new_src_remap is not None:
                args[0] = self._replace_region_buffer(args[0], new_src_remap)
        if len(args) >= 2:
            dst_buffer = self._region_call_buffer(args[1])
            new_dst_remap = self._lookup_buffer_remap(dst_buffer, promoted_output_remap) if dst_buffer is not None else None
            if new_dst_remap is not None:
                args[1] = self._replace_region_buffer(args[1], new_dst_remap)
        return tir.Call(
            copy_call.dtype,
            copy_call.op,
            args,
            getattr(copy_call, "annotations", None),
            getattr(copy_call, "span", None),
        )

    def _redirect_promoted_transfers(
        self,
        stmt: tir.Stmt,
        promoted_output_remap,
        promoted_input_remap,
    ) -> tir.Stmt:
        def transform(s: tir.Stmt) -> tir.Stmt:
            if isinstance(s, tir.BufferStore):
                remap = self._lookup_buffer_remap(s.buffer, promoted_output_remap)
                if remap is not None:
                    return tir.BufferStore(
                        self._remap_buffer_obj(remap),
                        s.value,
                        self._remap_buffer_indices(remap, list(s.indices)),
                        getattr(s, "predicate", None),
                        getattr(s, "span", None),
                    )
                return s
            if isinstance(s, tir.Evaluate) and self._is_tileop_call(s.value, "tl.tileop.copy"):
                return tir.Evaluate(
                    self._redirect_copy_regions(s.value, promoted_output_remap, promoted_input_remap),
                    getattr(s, "span", None),
                )
            if isinstance(s, tir.SeqStmt):
                new_seq = [transform(item) for item in s.seq]
                if not new_seq:
                    return tir.Evaluate(tir.IntImm("int32", 0))
                if len(new_seq) == 1:
                    return new_seq[0]
                return tir.SeqStmt(new_seq, getattr(s, "span", None))
            if isinstance(s, tir.For):
                return tir.For(
                    s.loop_var,
                    s.min,
                    s.extent,
                    s.kind,
                    transform(s.body),
                    getattr(s, "thread_binding", None),
                    getattr(s, "annotations", None),
                    getattr(s, "step", None),
                    getattr(s, "span", None),
                )
            if isinstance(s, tir.AttrStmt):
                return tir.AttrStmt(s.node, s.attr_key, s.value, transform(s.body), getattr(s, "span", None))
            if isinstance(s, tir.IfThenElse):
                return tir.IfThenElse(
                    s.condition,
                    transform(s.then_case),
                    transform(s.else_case) if s.else_case is not None else None,
                    getattr(s, "span", None),
                )
            if isinstance(s, tir.Bind):
                # tirx.Bind is a flat statement (no body); nothing to recurse into.
                return s
            if isinstance(s, tir.SBlockRealize):
                block = s.block
                new_block = tir.SBlock(
                    block.iter_vars,
                    block.reads,
                    block.writes,
                    block.name_hint,
                    transform(block.body),
                    block.init,
                    block.alloc_buffers,
                    block.match_buffers,
                    block.annotations,
                    getattr(block, "span", None),
                )
                return tir.SBlockRealize(s.iter_values, s.predicate, new_block, getattr(s, "span", None))
            if isinstance(s, tir.SBlock):
                return tir.SBlock(
                    s.iter_vars,
                    s.reads,
                    s.writes,
                    s.name_hint,
                    transform(s.body),
                    s.init,
                    s.alloc_buffers,
                    s.match_buffers,
                    s.annotations,
                    getattr(s, "span", None),
                )
            return s

        return transform(stmt)

    def _substitute_block(
        self,
        block: tir.SBlock,
        buffer_remap,
        prefix=None,
        protected_buffers=None,
        var_remap=None,
    ) -> tir.SBlock:
        new_block = _PrimitiveTIRBufferSubstitutor(buffer_remap, prefix, protected_buffers, var_remap).visit_stmt(block)
        if not isinstance(new_block, tir.SBlock):
            raise RuntimeError("Expected block substitution to return a tir.SBlock")
        return new_block

    def _replace_tilelang_root_body(self, func: tir.PrimFunc, new_block: tir.SBlock) -> tir.Stmt:
        def post_order(stmt):
            if isinstance(stmt, tir.SBlockRealize) and stmt.block.name_hint == "tilelang_root":
                return tir.SBlockRealize(
                    stmt.iter_values,
                    stmt.predicate,
                    new_block,
                    getattr(stmt, "span", None),
                )
            return stmt

        return tir.stmt_functor.ir_transform(func.body, None, post_order)

    def _set_tir_global_var_struct_info(
        self,
        gv: tvm.ir.GlobalVar,
        input_args: list,
        output_sinfos,
        ret_sinfo,
    ) -> None:
        param_sinfos = []
        for arg in input_args:
            sinfo = getattr(arg, "struct_info_", None)
            if sinfo is None:
                sinfo = getattr(arg, "struct_info", None)
            if sinfo is None:
                raise RuntimeError(f"Cannot infer struct_info for fused call_tir arg {arg}")
            param_sinfos.append(sinfo)
        param_sinfos.extend(output_sinfos)
        relax.expr._update_struct_info(
            gv,
            relax.FuncStructInfo(param_sinfos, ret_sinfo, False),
        )

    def _block_with_bindings(self, block, bindings):
        if isinstance(block, relax.DataflowBlock):
            return relax.DataflowBlock(bindings, getattr(block, "span", None))
        return relax.BindingBlock(bindings, getattr(block, "span", None))

    def _rewrite_main_fused_call(
        self,
        main_func: relax.Function,
        fused_gv: tvm.ir.GlobalVar,
        fused_binding_vars: list[relax.Var],
        sink_output_vars: list[relax.Var],
        fused_relax_args: list,
        fused_output_sinfos,
        fused_tir_vars,
    ) -> relax.Function:
        if not isinstance(main_func.body, relax.SeqExpr):
            raise RuntimeError("Expected main function body to be a Relax SeqExpr")
        if not sink_output_vars:
            raise RuntimeError("Cannot rewrite main: fused group has no sink output")
        if not fused_output_sinfos:
            raise RuntimeError("Cannot rewrite main: fused call has no output sinfo")

        inserted = False
        new_blocks = []
        for block in main_func.body.blocks:
            new_bindings = []
            for binding in block.bindings:
                if not isinstance(binding, relax.VarBinding):
                    new_bindings.append(binding)
                    continue
                value = binding.value
                if not self._var_in(binding.var, fused_binding_vars):
                    new_bindings.append(binding)
                    continue
                if not _is_call_tir(value):
                    new_bindings.append(binding)
                    continue
                if self._var_in(binding.var, sink_output_vars):
                    if inserted:
                        raise RuntimeError("Cannot rewrite main: multiple fused sink insertion points")
                    if len(fused_output_sinfos) != 1:
                        raise RuntimeError("Main rewrite currently expects one fused output")
                    out_sinfo = fused_output_sinfos[0] if len(fused_output_sinfos) == 1 else fused_output_sinfos
                    fused_call = relax.call_tir(
                        fused_gv,
                        relax.Tuple(fused_relax_args),
                        out_sinfo=out_sinfo,
                        tir_vars=fused_tir_vars,
                    )
                    new_bindings.append(
                        relax.VarBinding(
                            binding.var,
                            fused_call,
                            getattr(binding, "span", None),
                        )
                    )
                    inserted = True
            new_blocks.append(self._block_with_bindings(block, new_bindings))

        if not inserted:
            raise RuntimeError("Cannot rewrite main: failed to place fused call_tir")

        new_body = relax.SeqExpr(
            new_blocks,
            main_func.body.body,
            getattr(main_func.body, "span", None),
        )
        return relax.Function(
            list(main_func.params),
            new_body,
            main_func.ret_struct_info,
            getattr(main_func, "is_pure", True),
            main_func.attrs,
            getattr(main_func, "span", None),
        )

    def fuse(self, nodes_to_fuse: list[CallTirNode]) -> tvm.IRModule:
        topo_sorted_nodes = self.graph.topo_sort(nodes_to_fuse)
        connecting_edges = self.graph.edges_for(topo_sorted_nodes)
        if not connecting_edges:
            return self.mod

        ready_to_fuse: list[tvm.tirx.function.PrimFunc] = [s_tir.renew_defs(self.mod[node.name]) for node in topo_sorted_nodes]

        self._check_same_launch_threads(ready_to_fuse)
        roots = [self._find_tilelang_root(func) for func in ready_to_fuse]

        node2func = {node: i for i, node in enumerate(topo_sorted_nodes)}
        promote_input: dict[int, set[int]] = {}
        edge_infos = []
        output_shared_buffers = {}
        input_shared_buffers = {}
        for src_node, edge, dst_node in connecting_edges:
            src_idx = node2func[src_node]
            dst_idx = node2func[dst_node]
            target_edge_var = edge.var
            target_edge_idx = None
            for edge_idx, input_edge in enumerate(dst_node.inputs):
                if _same_var(input_edge.var, target_edge_var):
                    target_edge_idx = edge_idx
                    break
            if target_edge_idx is None:
                raise RuntimeError(f"Failed to find target edge {edge.name} in dst_node {dst_node.name}")
            promote_input.setdefault(dst_idx, set()).add(target_edge_idx)

            src_output = self._get_param_buffer(ready_to_fuse[src_idx], len(ready_to_fuse[src_idx].params) - 1)
            dst_input = self._get_param_buffer(ready_to_fuse[dst_idx], target_edge_idx)
            output_key = (src_idx, len(ready_to_fuse[src_idx].params) - 1)
            input_key = (dst_idx, target_edge_idx)
            if output_key not in output_shared_buffers:
                output_shared_buffers[output_key] = self._make_shared_intermediate_plan(
                    ready_to_fuse[src_idx],
                    roots[src_idx],
                    src_output,
                    f"{src_node.name}_{src_output.name}_fuse_out_shared",
                )
            if input_key not in input_shared_buffers:
                input_shared_buffers[input_key] = self._make_shared_intermediate_plan(
                    ready_to_fuse[dst_idx],
                    roots[dst_idx],
                    dst_input,
                    f"{dst_node.name}_{dst_input.name}_fuse_in{target_edge_idx}_shared",
                )
            edge_infos.append(
                (
                    src_idx,
                    target_edge_idx,
                    dst_idx,
                    output_key,
                    input_key,
                )
            )

        edge_infos_by_output = {}
        for _, _, _, output_key, input_key in edge_infos:
            edge_infos_by_output.setdefault(output_key, []).append(input_key)
        for output_key, input_keys in edge_infos_by_output.items():
            self._merge_shared_intermediate_layout(
                [output_shared_buffers[output_key]] + [input_shared_buffers[input_key] for input_key in input_keys]
            )

        edge_infos = [
            (
                src_idx,
                target_edge_idx,
                dst_idx,
                output_shared_buffers[output_key].shared_buffer,
                input_shared_buffers[input_key].shared_buffer,
            )
            for src_idx, target_edge_idx, dst_idx, output_key, input_key in edge_infos
        ]

        base_launch_vars = dict(self._thread_launch_vars(ready_to_fuse[0]))
        shared_endpoint_buffers = self._unique_buffers(
            [plan.shared_buffer for plan in output_shared_buffers.values()] + [plan.shared_buffer for plan in input_shared_buffers.values()]
        )
        rewritten_roots = []
        for idx, (func, root) in enumerate(zip(ready_to_fuse, roots)):
            promoted_output_remap = {}
            output_key = (idx, len(func.params) - 1)
            if output_key in output_shared_buffers:
                promoted_output_remap[self._get_param_buffer(func, len(func.params) - 1)] = output_shared_buffers[output_key]
            promoted_input_remap = {}
            for input_idx in sorted(promote_input.get(idx, set())):
                input_key = (idx, input_idx)
                if input_key in input_shared_buffers:
                    promoted_input_remap[self._get_param_buffer(func, input_idx)] = input_shared_buffers[input_key]
            new_body = self._redirect_promoted_transfers(
                root.block.body,
                promoted_output_remap,
                promoted_input_remap,
            )
            block = tir.SBlock(
                root.block.iter_vars,
                root.block.reads,
                root.block.writes,
                root.block.name_hint,
                new_body,
                root.block.init,
                root.block.alloc_buffers,
                root.block.match_buffers,
                root.block.annotations,
                getattr(root.block, "span", None),
            )
            prefix = None if idx == 0 else f"fused{idx}"
            protected_buffers = list(func.buffer_map.values()) + shared_endpoint_buffers
            launch_var_remap = {}
            if idx != 0:
                for thread_tag, var in self._thread_launch_vars(func):
                    if thread_tag in base_launch_vars:
                        launch_var_remap[var] = base_launch_vars[thread_tag]
            node_block_order = self.tile_plan.block_orders.get(topo_sorted_nodes[idx])
            if node_block_order is not None:
                base_block_var = base_launch_vars.get("blockIdx.x")
                if base_block_var is None:
                    raise RuntimeError("Cannot apply fused block order without blockIdx.x")
                for thread_tag, var in self._thread_launch_vars(func):
                    if thread_tag == "blockIdx.x":
                        launch_var_remap[var] = self._replace_block_idx_symbol(
                            node_block_order,
                            base_block_var,
                        )
            rewritten_roots.append(self._substitute_block(block, {}, prefix, protected_buffers, launch_var_remap))

        outgoing_dependency_copies = {}
        dependency_aliases = {}
        for src_idx, _, _, src_shared_buffer, dst_shared_buffer in edge_infos:
            if not _buffer_same(src_shared_buffer, dst_shared_buffer):
                outgoing_dependency_copies.setdefault(src_idx, []).append(self._make_buffer_copy(src_shared_buffer, dst_shared_buffer))
                dependency_aliases[dst_shared_buffer] = src_shared_buffer

        fused_stmts = []
        fused_alloc_buffers = []
        fused_match_buffers = []
        for idx, block in enumerate(rewritten_roots):
            fused_stmts.extend(self._as_seq(block.body))
            fused_stmts.extend(outgoing_dependency_copies.get(idx, []))
            fused_alloc_buffers.extend(list(block.alloc_buffers))
            fused_match_buffers.extend(list(block.match_buffers))

        fused_stmts, shared_buffer_aliases = self._remove_redundant_shared_to_shared_copies(
            fused_stmts,
            dependency_aliases,
            shared_endpoint_buffers,
        )
        aliased_shared_buffers = list(shared_buffer_aliases.keys())
        fused_alloc_buffers = [buffer for buffer in fused_alloc_buffers if not self._buffer_in(buffer, aliased_shared_buffers)]
        fused_alloc_buffers.extend([buffer for buffer in shared_endpoint_buffers if not self._buffer_in(buffer, aliased_shared_buffers)])
        fused_alloc_buffers = self._unique_buffers(fused_alloc_buffers)

        first_root = roots[0]
        first_block = first_root.block
        fused_root_block = tir.SBlock(
            first_block.iter_vars,
            first_block.reads,
            first_block.writes,
            first_block.name_hint,
            tir.SeqStmt(fused_stmts),
            first_block.init,
            fused_alloc_buffers,
            fused_match_buffers,
            first_block.annotations,
            getattr(first_block, "span", None),
        )
        if shared_buffer_aliases:
            fused_root_block = self._substitute_block(fused_root_block, shared_buffer_aliases)

        skip_params = {idx: set() for idx in range(len(ready_to_fuse))}
        connecting_tensor_vars = [edge.var for _, edge, _ in connecting_edges]
        fused_binding_vars = [output.var for node in topo_sorted_nodes for output in node.outputs]
        sink_output_vars = [
            output.var for node in topo_sorted_nodes for output in node.outputs if not self._var_in(output.var, connecting_tensor_vars)
        ]
        for src_node, _, dst_node in connecting_edges:
            src_idx = node2func[src_node]
            dst_idx = node2func[dst_node]
            skip_params[src_idx].add(len(ready_to_fuse[src_idx].params) - 1)
            skip_params[dst_idx].update(promote_input.get(dst_idx, set()))

        fused_params = []
        fused_relax_args = []
        fused_buffer_map = {}
        for idx, func in enumerate(ready_to_fuse):
            call_tir_args = _call_tir_arg_fields(topo_sorted_nodes[idx].call_tir)
            for param_idx, param in enumerate(func.params):
                if param_idx in skip_params[idx]:
                    continue
                fused_params.append(param)
                if param_idx < len(call_tir_args):
                    fused_relax_args.append(call_tir_args[param_idx])
                if param in func.buffer_map:
                    fused_buffer_map[param] = func.buffer_map[param]

        fused_body = self._replace_tilelang_root_body(ready_to_fuse[0], fused_root_block)
        fused_func = tir.PrimFunc(
            fused_params,
            fused_body,
            ready_to_fuse[0].ret_type,
            fused_buffer_map,
            ready_to_fuse[0].attrs,
            getattr(ready_to_fuse[0], "span", None),
        )

        fused_output_name = "_".join(_var_name_hint(output.var) for node in topo_sorted_nodes for output in node.outputs)
        fused_gv = self._make_unique_global_var(f"{topo_sorted_nodes[0].name}_fused_{fused_output_name}")
        new_funcs = dict(self.mod.functions.items())
        new_funcs[fused_gv] = fused_func

        if len(sink_output_vars) != 1:
            sink_output_names = [_var_name_hint(var) for var in sink_output_vars]
            raise RuntimeError(f"Expected exactly one fused Relax output, found {sink_output_names}")
        main_func = self.mod["main"]
        sink_binding = None
        for var, binding in self.graph.main_binding_by_var:
            if _same_var(var, sink_output_vars[0]):
                sink_binding = binding
                break
        if sink_binding is None:
            raise RuntimeError(f"Failed to find sink binding for fused Relax output {_var_name_hint(sink_output_vars[0])}")
        if not isinstance(sink_binding.value, relax.Call):
            raise RuntimeError("Expected fused sink binding to be a call_tir")
        fused_output_sinfos = _call_tir_out_sinfos(sink_binding.value)
        self._set_tir_global_var_struct_info(
            fused_gv,
            fused_relax_args,
            fused_output_sinfos,
            sink_binding.value.args[0].struct_info.ret,
        )
        main_gv = self.mod.get_global_var("main")
        new_funcs[main_gv] = self._rewrite_main_fused_call(
            main_func,
            fused_gv,
            fused_binding_vars,
            sink_output_vars,
            fused_relax_args,
            fused_output_sinfos,
            _call_tir_tir_vars(sink_binding.value),
        )
        return tvm.IRModule(new_funcs, attrs=self.mod.attrs)


class _TuneResult:
    def __init__(
        self,
        baseline_mod: tvm.IRModule,
        fused_mod: tvm.IRModule,
        baseline_latency: float | None,
        fused_latency: float | None,
        tile_plan: _FusionTilePlan | None = None,
    ):
        self.baseline_mod = baseline_mod
        self.fused_mod = fused_mod
        self.baseline_latency = baseline_latency
        self.fused_latency = fused_latency
        self.tile_plan = tile_plan


class _FusionGroup:
    def __init__(
        self,
        nodes: list[CallTirNode],
        group_id: int,
        result: _TuneResult | None,
        gain: float | None,
    ):
        self.nodes = nodes
        self.group_id = group_id
        self.result = result
        self.gain = gain


def _get_nodes_dependency(
    graph: GraphManager,
    nodes: list[CallTirNode],
    processed: set[CallTirNode],
) -> list[CallTirNode]:
    """Collect unfused producer dependencies needed to fuse the given nodes."""
    queue = list(nodes)
    deps: set[CallTirNode] = set()
    while queue:
        node = queue.pop(0)
        if node in processed or node in deps:
            continue
        deps.add(node)
        for edge in node.inputs:
            producer = _find_edge_producer(edge, graph.connecting_edges)
            if producer is None or producer in processed or producer in deps:
                continue
            queue.append(producer)
    return list(deps)


class Tunner:
    """Pick fixed tiles, schedule, fuse TIR, and benchmark candidates."""

    def __init__(self, graph: GraphManager, target, use_cuda_graph, device_index=0):
        self.graph = graph
        self.target = target
        self.use_cuda_graph = use_cuda_graph
        self.device_index = device_index
        self._rules = None
        self._baseline_mod = None
        self._baseline_latency = None
        self._tune_cache = {}

    def _schedule_rules(self):
        if self._rules is None:
            self._rules = default_schedule_rules(self.target)
        return self._rules

    def _schedule_mod_with_tile_plans(
        self,
        tile_plans: list[_FusionTilePlan],
    ) -> tvm.IRModule:
        def _copy_ir_module(mod: tvm.IRModule) -> tvm.IRModule:
            funcs = {gv: func for gv, func in mod.functions_items()}
            global_infos = getattr(mod, "global_infos", None)

            if global_infos is not None:
                try:
                    return tvm.IRModule(funcs, attrs=mod.attrs, global_infos=global_infos)
                except TypeError:
                    pass

            return tvm.IRModule(funcs, attrs=mod.attrs)

        mod = _copy_ir_module(self.graph.mod)
        selected_tiles: dict[CallTirNode, list[int]] = {}
        for tile_plan in tile_plans:
            selected_tiles.update(tile_plan.node_tiles)
        rules = self._schedule_rules()
        with self.target:
            if selected_tiles:
                mod = self._apply_fixed_config_schedules(mod, rules, selected_tiles)
            mod = dl.ApplyDefaultSchedule(*rules)(mod)
            return canonicalize_scheduled_ir(mod)

    def baseline(self) -> tuple[tvm.IRModule, float | None]:
        if self._baseline_mod is None:
            self._baseline_mod = self._schedule_mod_with_tile_plans([])
        if self._baseline_latency is None:
            self._baseline_latency = self._bench_fused_vm(self._baseline_mod)
        return self._baseline_mod, self._baseline_latency

    def _dtype_nbytes(self, dtype: str | None) -> int:
        if dtype is None:
            return 4
        dtype = str(dtype)
        if dtype.startswith("float") or dtype.startswith("int") or dtype.startswith("uint"):
            digits = "".join(ch for ch in dtype if ch.isdigit())
            if digits:
                return max(1, int(digits) // 8)
        if dtype in ("bool",):
            return 1
        return 4

    def _merge_tile(self, lhs: list[int] | None, rhs: list[int]) -> list[int]:
        if lhs is None:
            return list(rhs)
        if len(lhs) != len(rhs):
            return list(rhs)
        return [max(a, b) for a, b in zip(lhs, rhs)]

    def _external_traffic_score(
        self,
        nodes_to_score: list[CallTirNode],
        node_tiles: dict[CallTirNode, list[int]],
    ) -> int:
        score = 0
        for node in nodes_to_score:
            tile = node_tiles[node]
            input_tiles = node.propogate_inputs(tile)
            for edge, input_tile in zip(node.inputs, input_tiles):
                if edge.buffer is not None and _buffer_same(edge.buffer, node.tir_info.output_buffers[0]):
                    continue
                if edge.shape is None:
                    continue
                dtype_nbytes = self._dtype_nbytes(edge.dtype)
                score += int(math.prod(input_tile)) * dtype_nbytes
            for output_edge in node.outputs:
                if output_edge.shape is not None:
                    score += int(math.prod(tile)) * self._dtype_nbytes(output_edge.dtype)
        return score

    def _node_output_tile_bytes(
        self,
        node: CallTirNode,
        node_tiles: dict[CallTirNode, list[int]],
    ) -> int:
        tile = node_tiles[node]
        dtype = node.outputs[0].dtype if node.outputs else None
        return int(math.prod(tile)) * self._dtype_nbytes(dtype)

    def _connected_intermediate_shared_bytes(
        self,
        connecting_edges: list[tuple],
        node_tiles: dict[CallTirNode, list[int]],
    ) -> int:
        bytes_by_src = {}
        for src_node, _, _ in connecting_edges:
            if src_node not in node_tiles:
                continue
            bytes_by_src.setdefault(src_node, self._node_output_tile_bytes(src_node, node_tiles))
        return sum(bytes_by_src.values())

    def _matmul_like_shared_bytes(self, node: CallTirNode, node_tile: list[int]) -> int:
        if node.tir_info is None:
            return 0
        if len(node_tile) < 2:
            return 0
        block_m = int(node_tile[-2])
        block_n = int(node_tile[-1])
        total = 0
        for block in node.tir_info.blocks:
            if len(block.reduce_vars) != 1 or len(block.reads) < 2 or len(block.writes) != 1:
                continue
            reduce_extent = block.reduce_extents.get(block.reduce_vars[0])
            if reduce_extent is None:
                continue
            input_bits = max(tvm.DataType(region.buffer.dtype).bits for region in block.reads)
            element_bytes = max(1, input_bits // 8)
            block_k = min(64 if input_bits <= 16 else 32, int(reduce_extent))
            stages = 4 if input_bits <= 16 else 2
            total += (block_m + block_n) * block_k * element_bytes * stages
        return int(total)

    def _estimated_fused_shared_memory_bytes(
        self,
        topo_sorted_nodes: list[CallTirNode],
        connecting_edges: list[tuple],
        node_tiles: dict[CallTirNode, list[int]],
    ) -> int:
        total = self._connected_intermediate_shared_bytes(connecting_edges, node_tiles)
        for node in topo_sorted_nodes:
            total += self._matmul_like_shared_bytes(node, node_tiles[node])
        return int(total)

    def _propagate_candidate_tiles(
        self,
        topo_sorted_nodes: list[CallTirNode],
        connecting_edges: list[tuple],
        sink_node: CallTirNode,
        sink_tile: list[int],
    ) -> dict[CallTirNode, list[int]] | None:
        node_tiles: dict[CallTirNode, list[int]] = {sink_node: list(sink_tile)}
        if not sink_node.is_valid_tile(sink_tile):
            return None
        for node in reversed(topo_sorted_nodes):
            if node not in node_tiles:
                continue
            input_tiles = node.propogate_inputs(node_tiles[node])
            for edge, input_tile in zip(node.inputs, input_tiles):
                producer = _find_edge_producer(edge, connecting_edges)
                if producer is None:
                    continue
                producer_shape = producer.get_space_dim()
                if len(producer_shape) != len(input_tile):
                    return None
                bounded_tile = [
                    max(1, min(int(tile_extent), int(shape_extent))) for tile_extent, shape_extent in zip(input_tile, producer_shape)
                ]
                if producer in node_tiles:
                    node_tiles[producer] = self._merge_tile(node_tiles[producer], bounded_tile)
                else:
                    node_tiles[producer] = bounded_tile
        if any(node not in node_tiles for node in topo_sorted_nodes):
            return None
        if any(not node.is_valid_tile(tile) for node, tile in node_tiles.items()):
            return None
        return node_tiles

    def _find_sink_nodes(
        self,
        topo_sorted_nodes: list[CallTirNode],
        connecting_edges: list[tuple],
    ) -> list[CallTirNode]:
        producers = {src_node for src_node, _, _ in connecting_edges}
        return [node for node in topo_sorted_nodes if node not in producers]

    def _assign_block_order(
        self,
        topo_sorted_nodes: list[CallTirNode],
        connecting_edges: list[tuple],
        node_tiles: dict[CallTirNode, list[int]],
    ) -> tuple[int, dict[CallTirNode, tir.PrimExpr]] | None:
        expected = None
        for node, tile in node_tiles.items():
            grid = node.grid_size(tile)
            if grid <= 0:
                return None
            if expected is None:
                expected = grid
            elif expected != grid:
                return None
        if expected is None:
            return None

        sink_nodes = self._find_sink_nodes(topo_sorted_nodes, connecting_edges)
        block_idx = tvm.te.var("block_idx", "int64")
        analyzer = tvm.arith.Analyzer()
        analyzer.update(block_idx, tvm.arith.ConstIntBound(0, expected - 1))
        expr_map: dict[CallTirNode, tir.PrimExpr] = {node: block_idx for node in sink_nodes}
        result: dict[CallTirNode, tir.PrimExpr] = {}
        for node in reversed(topo_sorted_nodes):
            if node not in expr_map:
                return None
            expr = analyzer.simplify(expr_map[node])
            if not (expr.same_as(block_idx) or self._is_const_prim_expr(expr)):
                result[node] = expr
            deps = node.block_infer(node_tiles, expr, connecting_edges)
            if deps is None:
                return None
            for edge, dep in zip(node.inputs, deps):
                if dep is None:
                    continue
                producer = _find_edge_producer(edge, connecting_edges)
                if producer is None:
                    continue
                dep = analyzer.simplify(dep)
                if producer in expr_map:
                    if not _same_prim_expr(dep, analyzer.simplify(expr_map[producer])):
                        return None
                else:
                    expr_map[producer] = dep
        return expected, result

    def _is_const_prim_expr(self, expr) -> bool:
        const_expr = getattr(getattr(tir, "expr", None), "ConstExpr", None)
        if const_expr is not None and isinstance(expr, const_expr):
            return True
        return isinstance(expr, (tir.IntImm, tir.FloatImm))

    def _search_tile_size(
        self,
        topo_sorted_nodes: list[CallTirNode],
        connecting_edges: list[tuple],
    ) -> _FusionTilePlan:
        sink_nodes = self._find_sink_nodes(topo_sorted_nodes, connecting_edges)
        if len(sink_nodes) != 1:
            raise RuntimeError(f"Tile search currently expects one fused sink node, got {[node.name for node in sink_nodes]}")
        sink_node = sink_nodes[0]
        output_shape = sink_node.get_space_dim()
        if not output_shape:
            raise RuntimeError("Cannot infer sink output shape for fusion tile search")
        axis_candidates = sink_node.candidate_axis_tiles()
        best = None
        max_tile_elems = 65536
        max_candidates = 4096
        checked = 0
        for sink_tile_tuple in product(*axis_candidates):
            checked += 1
            if checked > max_candidates:
                break
            sink_tile = list(sink_tile_tuple)
            if math.prod(sink_tile) > max_tile_elems:
                continue
            node_tiles = self._propagate_candidate_tiles(topo_sorted_nodes, connecting_edges, sink_node, sink_tile)
            if node_tiles is None:
                continue
            block_order_result = self._assign_block_order(
                topo_sorted_nodes,
                connecting_edges,
                node_tiles,
            )
            if block_order_result is None:
                continue
            grid, block_orders = block_order_result
            shared_bytes = self._estimated_fused_shared_memory_bytes(
                topo_sorted_nodes,
                connecting_edges,
                node_tiles,
            )
            if shared_bytes >= _MAX_FUSED_SHARED_MEMORY_BYTES:
                continue
            traffic = self._external_traffic_score(topo_sorted_nodes, node_tiles)
            tile_work = sum(int(math.prod(tile)) for tile in node_tiles.values())
            non_identity_orders = len(block_orders)
            score = (
                (traffic + 1) * max(grid, 1),
                non_identity_orders,
                grid,
                tile_work,
                shared_bytes,
            )
            if best is None or score < best[0]:
                best = (score, _FusionTilePlan(node_tiles, grid, block_orders))
        if best is None:
            raise RuntimeError("Failed to find aligned tile sizes for fused call_tir graph")
        return best[1]

    def _make_torch_tensor_arg(self, shape, dtype, device):
        import torch
        from tilelang.graph.utils import tvm_dtype_to_torch

        torch_dtype = tvm_dtype_to_torch(str(dtype))
        shape = tuple(int(dim) for dim in shape)
        if torch_dtype.is_floating_point or torch_dtype.is_complex:
            return torch.randn(shape, dtype=torch_dtype, device=device)
        if torch_dtype == torch.bool:
            return torch.randint(0, 2, shape, dtype=torch_dtype, device=device)
        return torch.randint(0, 8, shape, dtype=torch_dtype, device=device)

    def _shape_from_struct_info(self, sinfo) -> list[int] | None:
        shape = getattr(sinfo, "shape", None)
        values = getattr(shape, "values", None)
        if values is None:
            return None
        result = []
        for value in values:
            int_value = _as_const_int(value)
            if int_value is None:
                return None
            result.append(int_value)
        return result

    def _make_benchmark_args_from_main(self, mod: tvm.IRModule) -> list | None:
        import torch

        if "cuda" not in str(self.target):
            logger.debug("Skipping fused VM benchmark for non-CUDA target %s", self.target)
            return None
        device = torch.device("cuda", self.device_index)
        if not torch.cuda.is_available():
            logger.debug("Skipping fused VM benchmark because CUDA is not available")
            return None

        main_func = mod["main"]
        args = []
        for param in main_func.params:
            sinfo = getattr(param, "struct_info_", None)
            if sinfo is None:
                sinfo = getattr(param, "struct_info", None)
            if not isinstance(sinfo, relax.TensorStructInfo):
                logger.debug(
                    "Skipping fused VM benchmark: main param %s is not a TensorStructInfo",
                    _var_name_hint(param),
                )
                return None
            shape = self._shape_from_struct_info(sinfo)
            dtype = getattr(sinfo, "dtype", None)
            if shape is None or dtype is None:
                logger.debug(
                    "Skipping fused VM benchmark: cannot infer static shape/dtype for %s",
                    _var_name_hint(param),
                )
                return None
            args.append(self._make_torch_tensor_arg(shape, dtype, device))
        return args

    def _bench_fused_vm(self, mod: tvm.IRModule) -> float | None:
        import torch

        args = self._make_benchmark_args_from_main(mod)
        if args is None:
            return None

        from tilelang.graph.vm_build import _compile_tir_for_vm, _apply_vm_lowering, VMRunner
        from tilelang import profiler

        lowering_passes = [
            relax.transform.RewriteDataflowReshape(),
            relax.transform.ToNonDataflow(),
            relax.transform.RemovePurityChecking(),
            relax.transform.CallTIRRewrite(),
            relax.transform.StaticPlanBlockMemory(),
        ]
        if self.use_cuda_graph:
            lowering_passes.append(relax.transform.RewriteCUDAGraph())
        lowering_passes.append(relax.transform.LowerAllocTensor())

        bench_mod = tvm.transform.Sequential(lowering_passes)(mod)
        bench_mod = _apply_vm_lowering(bench_mod, self.target)
        builder = relax.ExecBuilder()
        bench_mod = tvm.get_global_func("relax.VMCodeGen")(builder, bench_mod)
        tir_funcs = {gv: func for gv, func in bench_mod.functions.items() if isinstance(func, tir.PrimFunc)}
        lib = None
        if tir_funcs:
            tir_mod = tvm.IRModule(tir_funcs, attrs=bench_mod.attrs)
            lib = _compile_tir_for_vm(tir_mod, self.target)
        vm_exe = tvm.get_global_func("relax.VMLink")(
            builder,
            self.target,
            lib,
            [],
            {},
        )
        vm_exe = relax.vm_build.VMExecutable(vm_exe)
        vm_runner = VMRunner(
            vm_exe,
            device=tvm.cuda(self.device_index),
            func_name="main",
            clone_output=False,
            torch_device=torch.device("cuda", self.device_index),
        )
        return profiler.do_bench(lambda: vm_runner(*args))

    def _apply_rules_with_config(self, func, rules, config=None):
        for rule in rules:
            if config is not None and hasattr(rule, "apply_config"):
                space = rule.apply_config(func, self.target, config, False)
            else:
                space = rule.apply(func, self.target, False)
            if space is None:
                continue
            if isinstance(space, s_tir.Schedule):
                space = [space]
            return space
        return None

    def _apply_fixed_config_schedules(
        self,
        mod: tvm.IRModule,
        rules,
        selected_tiles: dict[CallTirNode, list[int]],
    ) -> tvm.IRModule:
        updated_functions = {}
        tile_by_name = {node.name: tile for node, tile in selected_tiles.items()}
        for g_var, func in mod.functions_items():
            if not isinstance(func, tir.PrimFunc):
                continue
            if func.attrs is not None and func.attrs.get("tirx.is_scheduled", False):
                continue
            config = tile_by_name.get(g_var.name_hint)
            space = self._apply_rules_with_config(func, rules, config)
            if space is None:
                continue
            if len(space) != 1:
                raise RuntimeError(f"Expected one schedule for {g_var.name_hint}, got {len(space)}")
            updated_functions[g_var] = (
                space[0]
                .mod["main"]
                .with_attr(
                    "tirx.is_scheduled",
                    True,
                )
            )
        for g_var, func in updated_functions.items():
            mod[g_var] = func
        return mod

    def tune(self, nodes_to_fuse: list[CallTirNode]) -> _TuneResult | None:
        topo_sorted_nodes = self.graph.topo_sort(nodes_to_fuse)
        cache_key = tuple(_node_key(node) for node in topo_sorted_nodes)
        if cache_key in self._tune_cache:
            return self._tune_cache[cache_key]
        connecting_edges = self.graph.edges_for(topo_sorted_nodes)
        if not connecting_edges:
            baseline_mod, baseline_latency = self.baseline()
            result = _TuneResult(
                baseline_mod,
                baseline_mod,
                baseline_latency,
                baseline_latency,
                None,
            )
            self._tune_cache[cache_key] = result
            return result
        try:
            tile_plan = self._search_tile_size(topo_sorted_nodes, connecting_edges)
            scheduled_mod = self._schedule_mod_with_tile_plans([tile_plan])
            scheduled_graph = GraphManager(scheduled_mod)
            scheduled_nodes = [
                node
                for node in scheduled_graph.topo_sorted_nodes
                if any(_node_output_name(node) == _node_output_name(old_node) for old_node in topo_sorted_nodes)
            ]
            if len(scheduled_nodes) != len(topo_sorted_nodes):
                raise RuntimeError(
                    f"Failed to remap fusion candidate nodes after scheduling: {[_node_output_name(node) for node in topo_sorted_nodes]}"
                )
            scheduled_tile_plan = _remap_tile_plan(tile_plan, scheduled_nodes)
            fused_mod = GraphFuser(
                scheduled_graph,
                scheduled_mod,
                scheduled_tile_plan,
            ).fuse(scheduled_nodes)
            baseline_mod, baseline_latency = self.baseline()
            fused_latency = self._bench_fused_vm(fused_mod)
            result = _TuneResult(
                baseline_mod,
                fused_mod,
                baseline_latency,
                fused_latency,
                tile_plan,
            )
        except Exception:
            logger.debug(
                "Fusion candidate failed: %s",
                [node.name for node in topo_sorted_nodes],
                exc_info=True,
            )
            result = None
        self._tune_cache[cache_key] = result
        return result


class Engine:
    """Greedy fusion-group search over the call_tir dependency graph."""

    def __init__(self, graph: GraphManager, tunner: Tunner):
        self.graph = graph
        self.tunner = tunner
        self.node2group: dict[CallTirNode, int] = {}
        self.node_topo_id = {node: idx for idx, node in enumerate(graph.topo_sorted_nodes)}

    def run(self) -> list[_FusionGroup]:
        fusion_groups = []
        for node in self.graph.topo_sorted_nodes:
            if node in self.node2group:
                continue
            group = self._build_fusion_group(node)
            fusion_groups.append(group)
            logger.info(
                "Fusion group created: %s %s gain=%s",
                group.group_id,
                [node.name for node in group.nodes],
                group.gain,
            )
        return fusion_groups

    def _next_group_id(self) -> int:
        return 0 if not self.node2group else max(self.node2group.values()) + 1

    def _topo_sort_group(self, nodes: list[CallTirNode]) -> list[CallTirNode]:
        return sorted(nodes, key=lambda node: self.node_topo_id[node])

    def _group_output_edges(self, nodes: list[CallTirNode]) -> tuple[set[tuple], set[tuple]]:
        node_set = set(nodes)
        internal_outputs = set()
        external_outputs = set()
        for node in nodes:
            for output_edge in node.outputs:
                edge_key = (node, output_edge.name)
                consumers = self.graph.consumers_for(node, output_edge)
                if any(consumer in node_set for consumer in consumers):
                    internal_outputs.add(edge_key)
                if self.graph.is_graph_output(output_edge) or any(consumer not in node_set for consumer in consumers):
                    external_outputs.add(edge_key)
        return internal_outputs, external_outputs

    def _is_valid_group(self, nodes: list[CallTirNode]) -> bool:
        connecting_edges = self.graph.edges_for(nodes)
        if not connecting_edges:
            return False
        sink_nodes = self.tunner._find_sink_nodes(nodes, connecting_edges)
        if len(sink_nodes) != 1:
            return False
        internal_outputs, external_outputs = self._group_output_edges(nodes)
        return not internal_outputs.intersection(external_outputs)

    def _candidate_consumers(self, node: CallTirNode) -> list[CallTirNode] | None:
        candidates = []
        for output_edge in node.outputs:
            if self.graph.is_graph_output(output_edge):
                return None
            for consumer in self.graph.consumers_for(node, output_edge):
                if consumer in self.node2group or consumer in candidates:
                    continue
                candidates.append(consumer)
        return candidates

    def _build_fusion_group(self, top_node: CallTirNode) -> _FusionGroup:
        cur_group = [top_node]
        cur_group_id = self._next_group_id()
        cur_gain = 0.0
        best_result = None
        self.node2group[top_node] = cur_group_id
        queue = [top_node]

        while queue:
            node = queue.pop(0)
            candidates = self._candidate_consumers(node)
            if not candidates:
                continue

            fusing_nodes = _get_nodes_dependency(
                self.graph,
                candidates,
                set(self.node2group.keys()),
            )
            if len(fusing_nodes) == 0 or len(fusing_nodes) > 10:
                continue

            new_group = self._topo_sort_group(fusing_nodes + cur_group)
            if not self._is_valid_group(new_group):
                continue

            result = self.tunner.tune(new_group)
            if result is None:
                continue
            gain = self.compute_gain(result)
            if gain is not None and gain < cur_gain:
                continue

            cur_gain = gain if gain is not None else cur_gain
            cur_group = new_group
            best_result = result
            for fused_node in fusing_nodes:
                self.node2group[fused_node] = cur_group_id
                queue.append(fused_node)

        if best_result is None:
            result = self.tunner.tune(cur_group)
            best_result = result
            cur_gain = 0.0
        return _FusionGroup(cur_group, cur_group_id, best_result, cur_gain)

    def compute_gain(self, result: _TuneResult) -> float | None:
        if result.baseline_latency is None or result.fused_latency is None:
            return None
        return result.baseline_latency - result.fused_latency


def _apply_fusion_groups(
    graph: GraphManager,
    tunner: Tunner,
    fusion_groups: list[_FusionGroup],
) -> tvm.IRModule:
    accepted_groups = [
        group for group in fusion_groups if len(group.nodes) > 1 and group.result is not None and group.result.tile_plan is not None
    ]
    if not accepted_groups:
        return tunner.baseline()[0]

    scheduled_mod = tunner._schedule_mod_with_tile_plans([group.result.tile_plan for group in accepted_groups])
    fused_mod = scheduled_mod
    current_graph = GraphManager(fused_mod)
    for group in accepted_groups:
        current_nodes = [
            node
            for node in current_graph.topo_sorted_nodes
            if any(_node_output_name(node) == _node_output_name(old_node) for old_node in group.nodes)
        ]
        if len(current_nodes) != len(group.nodes):
            logger.debug(
                "Skipping stale fusion group after earlier rewrites: %s",
                [node.name for node in group.nodes],
            )
            continue
        current_tile_plan = _remap_tile_plan(group.result.tile_plan, current_nodes)
        fused_mod = GraphFuser(current_graph, fused_mod, current_tile_plan).fuse(current_nodes)
        current_graph = GraphManager(fused_mod)
    return fused_mod


def fuse_all(mod: tvm.IRModule, target, use_cuda_graph, device_index=0) -> tvm.IRModule:
    graph = GraphManager(mod)
    if not graph.has_fusion_edges():
        # No fusion edges, but TIR kernels still need scheduling so they
        # bear ``tirx.is_scheduled`` before device codegen.  Create a
        # lightweight Tunner that only applies the default schedule rules
        # without running the fusion engine or benchmarking.
        tunner = Tunner(graph, target, use_cuda_graph, device_index)
        return tunner._schedule_mod_with_tile_plans([])

    tunner = Tunner(graph, target, use_cuda_graph, device_index)
    fusion_groups = Engine(graph, tunner).run()
    fused_mod = _apply_fusion_groups(graph, tunner, fusion_groups)
    baseline_mod, baseline_latency = tunner.baseline()
    fused_latency = tunner._bench_fused_vm(fused_mod)
    if baseline_latency is not None and fused_latency is not None and baseline_latency <= fused_latency:
        logger.info(
            "Graph fusion rejected: baseline %.6f ms <= fused %.6f ms",
            baseline_latency,
            fused_latency,
        )
        return baseline_mod
    return fused_mod
