"""PrimFunc Wrapper and Block information Analaysis"""

from __future__ import annotations

import tvm
from tvm import tir
from tvm.tir import IterVar, PrimFunc
from typing import Any
from tvm.tir.schedule.schedule import BlockRV
import numpy as np
import functools
from ..analysis import BlockInfo, get_reduction_blocks
from .. import analysis
from .. import normalize_prim_func
from .shape_inference import get_analyzer_by_tir
from dataclasses import dataclass


def pre_order_traverse(block_analyzer, blocks, func):
    visited = set()

    def _traverse(block):
        if block in visited:
            return
        visited.add(block)
        for dep_blocks in block_analyzer.get_consumer_blocks(block):
            _traverse(dep_blocks)
        func(block)

    for block in blocks:
        _traverse(block)


class BlockAnalyzer:
    def __init__(self, sch) -> None:
        self.sch: tir.Schedule = sch
        self.block_infos: list[BlockInfo] = normalize_prim_func(self.sch)

    def get_block_name(self, block: BlockRV) -> str:
        return self.sch.get(block).name_hint

    def get_block_info(self, block: BlockRV) -> BlockInfo:
        for block_info in self.block_infos:
            if self.get_block_name(block) == block_info.name:
                return block_info
        return None

    def get_spatial_axis(self, block: BlockRV) -> list[IterVar]:
        block_info = self.get_block_info(block)
        axis = []
        for iter in block_info.iters:
            if iter.kind == "S":
                axis.append(iter)
        return axis

    def get_reduce_axis(self, block: BlockRV) -> list[IterVar]:
        block_info = self.get_block_info(block)
        raxis = []
        for iter in block_info.iters:
            if iter.kind == "R":
                raxis.append(iter)
        return raxis

    def get_input_buffers(self, block: BlockRV) -> list[tir.Buffer]:
        buffers = []
        for read in self.sch.get(block).reads:
            buffers.append(read.buffer)
        return buffers

    def get_output_buffers(self, block: BlockRV) -> list[tir.Buffer]:
        buffers = []
        for write in self.sch.get(block).writes:
            buffers.append(write.buffer)
        return buffers

    def get_buffers(self, block: BlockRV) -> list[tir.Buffer]:
        return self.get_input_buffers(block) + self.get_output_buffers(block)

    def get_producer_blocks(self, block: BlockRV) -> list[BlockRV]:
        return self.sch.get_producers(block)

    def get_consumer_blocks(self, block: BlockRV) -> list[BlockRV]:
        return self.sch.get_consumers(block)


@dataclass
class Edge:
    src_node: Node
    dst_node: Node
    src_id: int
    dst_id: int


class Node:
    def __init__(self, tags: dict | None = None, name: str = "Node") -> None:
        self.name = name
        if tags is None:
            tags = {}
        self._out_edges = []
        self._in_edges = []
        self._shapes = []
        self._dtypes = []
        self._tag: dict = {}
        self.update_tags(tags)

    def update_tags(self, tags: dict) -> None:
        for tag in tags:
            self.add_tag(tag, tags[tag])

    def set_tag(self, k: str, v: Any = True) -> None:
        self.add_tag(k, v)

    def add_tag(self, k: str, v: Any = True) -> None:
        self._tag[k] = v

    def get_tag(self, k: str) -> Any:
        if k not in self._tag:
            return None
        return self._tag[k]

    def is_placeholder(self):
        return False

    def is_output(self):
        return False

    @property
    def inputs(self) -> list[Edge]:
        return self._in_edges

    @property
    def outputs(self) -> list[Edge]:
        return self._out_edges

    def set_inputs(self, i: int, edge: Edge):
        assert i < len(self._in_edges)
        self._in_edges[i] = edge

    def set_outputs(self, i: int, edge: Edge):
        assert i < len(self._out_edges)
        self._out_edges[i] = edge

    def get_dtype(self, id=0) -> tvm.DataType:
        return self._dtypes[id]

    def set_dtype(self, dtype: tvm.DataType, id=0) -> None:
        assert isinstance(dtype, tvm.DataType), type(dtype)
        if dtype == tvm.DataType("bool"):
            dtype = tvm.DataType("int8")
        if len(self._dtypes) <= id:
            self._dtypes.extend([None for _ in range(id - len(self._dtypes) + 1)])
        elif self._dtypes[id] is not None:
            assert self._dtypes[id] == dtype, (self._dtypes, dtype)
        self._dtypes[id] = dtype

    def get_shape(self, id: int = 0) -> list[int]:
        return self._shapes[id]

    def set_shape(self, shape: list[int], id=0, overwrite=False) -> None:
        if len(self._shapes) <= id:
            self._shapes.extend([None for _ in range(id - len(self._shapes) + 1)])
        # elif self._shapes[id] is not None and not overwrite:
        #     assert self._shapes[id] == list(map(int, shape)), (self._shapes, list(map(int, shape)))
        self._shapes[id] = list(map(int, shape))

    def num_outputs(self) -> int:
        if len(self.outputs) == 0:
            return 0
        return max([e.src_id for e in self.outputs]) + 1

    def get_ir(self) -> str:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return "<Node, " + self.name + ">"


class PlaceHolderNode(Node):
    def __init__(self, name=""):
        super().__init__(name="PlaceHolder_" + name)

    def is_placeholder(self):
        return True

    def get_ir(self) -> str:
        return "placeholder"


class PrimFuncNode(Node):
    def __init__(self, prim_func: PrimFunc, tags: dict | None = None, name: str = "PrimFuncNode") -> None:
        super().__init__(tags, name=name)
        self.prim_func = self._specialize_func(prim_func)
        self.sch: tir.Schedule = tir.Schedule(self.prim_func)
        self.block_analyzer: BlockAnalyzer = BlockAnalyzer(self.sch)
        self.schedule_stages: list[BlockRV] = []
        self.blocks: list[BlockRV] = []
        self.output_blocks: list[BlockRV] = None
        self.reduction_block: BlockRV = None
        self.raxis = []
        self.input_buffers = []
        self.output_buffers = []
        self.buffers = []
        self.args = []
        self._analysis_funcinfo()
        self._assign_placeholder_node()
        self.ana = get_analyzer_by_tir(self.block_analyzer, self.blocks)

        # set input shapes and dtypes
        for edge, buffer in zip(self.inputs, self.input_buffers):
            edge.src_node.set_shape(buffer.shape, edge.src_id)
            edge.src_node.set_dtype(tvm.DataType(buffer.dtype), edge.src_id)
        for output_id, buffer in enumerate(self.output_buffers):
            self.set_shape(buffer.shape, output_id)
            self.set_dtype(tvm.DataType(buffer.dtype), output_id)

    def _assign_placeholder_node(self):
        inputs: list[Node] = []
        for buffer in self.input_buffers:
            inputs.append(PlaceHolderNode(buffer.name))

        for dst_id, n in enumerate(inputs):
            if isinstance(n, Node):
                n = (n, 0)
            assert len(n) == 2
            src_node, src_id = n[0], n[1]
            edge = Edge(src_node, self, src_id, dst_id)
            self._in_edges.append(edge)
            src_node._out_edges.append(edge)

    def _specialize_func(self, func: PrimFunc):
        # Specialize the function to make it more friendly for analysis.
        # set attrs
        for k, v in func.attrs.items():
            self.set_tag(k, v)
        if self.get_tag("is_speclized"):
            return func
        opt_shapes = self.get_tag("opt_shapes")
        if opt_shapes:
            for name, shape in opt_shapes.items():
                var = analysis.find_var_from_func(func, name)
                if var is not None:
                    func = func.specialize({var: shape.astype(var.dtype)})
        return func

    def _analysis_funcinfo(self):
        root_block = analysis.get_root_block(self.sch)
        blocks = self.sch.get_child_blocks(root_block)
        self.blocks = blocks

        self.output_blocks = self.sch.get_output_blocks(root_block)
        reduction_blocks = get_reduction_blocks(self.sch, blocks)
        if reduction_blocks is None:
            self.reduction_block = None
            self.schedule_stages.append(*self.output_blocks)
        else:
            # analysis on the last reduction block
            self.reduction_block = reduction_blocks[-1]
            # set raxis
            reduce_block_info = self.block_analyzer.get_block_info(self.reduction_block)
            for iter in reduce_block_info.iters:
                if iter.kind == "R":
                    self.raxis.append(iter)
            self.schedule_stages.append(self.reduction_block)

        # collect output buffers
        for output_block in self.output_blocks:
            for write in self.sch.get(output_block).writes:
                if write not in self.output_buffers:
                    self.output_buffers.append(write.buffer)

        for param in self.prim_func.params:
            if param not in self.prim_func.buffer_map:
                # in case of dynamic symbolic may in params
                continue
            buffer = self.prim_func.buffer_map[param]
            if buffer not in self.output_buffers:
                self.input_buffers.append(buffer)

        self.args = self.input_buffers + self.output_buffers
        self.buffers = [buffer for buffer in self.prim_func.buffer_map.values()]

        # set dtype
        self.set_dtype(tvm.DataType(self.output_buffers[0].dtype))

    def get_opt_shape(self, name) -> int:
        opt_shapes = self.get_tag("opt_shapes")
        if opt_shapes is None:
            return None
        return opt_shapes[name]

    def extent_wrapper(self, value) -> int:
        if isinstance(value, tvm.tir.Var):
            return self.get_opt_shape(value.name)
        elif isinstance(value, tvm.tir.IntImm):
            return int(value)
        else:
            return value

    @functools.lru_cache
    def get_space_dim(self) -> list[int]:
        dim_size = []
        if self.reduction_block:
            block_info = self.block_analyzer.get_block_info(self.reduction_block)
            for iter in block_info.iters:
                if iter.kind == "S":
                    if isinstance(iter.dom.extent, tvm.tir.IntImm):
                        dim_size.append(int(iter.dom.extent))
                    else:
                        assert isinstance(iter.dom.extent, tvm.tir.Var)
                        dim_size.append(self.get_opt_shape(iter.dom.extent.name))
        else:
            # assume outer stage has the same shape
            loops = self.sch.get_loops(self.schedule_stages[0])
            for loop in loops:
                dim_size.append(int(self.sch.get(loop).extent))
        return [int(x) for x in dim_size]

    def set_dtype(self, dtype: tvm.DataType, id=0) -> None:
        assert isinstance(dtype, tvm.DataType), type(dtype)
        if dtype == tvm.DataType("bool"):
            dtype = tvm.DataType("int8")
        if len(self._dtypes) <= id:
            self._dtypes.extend([None for _ in range(id - len(self._dtypes) + 1)])
        elif self._dtypes[id] is not None:
            assert self._dtypes[id] == dtype, (self._dtypes, dtype)
        self._dtypes[id] = dtype

    def get_buffer_dtype(self, buffer: tir.Buffer) -> tvm.DataType:
        return tvm.DataType(buffer.dtype)

    def propagate(self, tile, rstep: dict | None = None, targets=None):
        if rstep is None:
            rstep = {}
        shape = {
            self.block_analyzer.get_output_buffers(block)[0].name: [tvm.arith.ConstIntBound(0, val - 1) for val in tile]
            for block in self.schedule_stages
        }
        return self.ana.infer(shape, rstep, targets)

    def propagate_inputs(self, tile, rstep: dict | None = None) -> list[list[int]]:
        if rstep is None:
            rstep = {}
        read_idx_offset = len(self.input_buffers)
        targets = [t.name for t in self.args[:read_idx_offset]]
        shapes, intermediate_bind = self.propagate(tile, rstep, targets)
        results = []
        for i, arg in enumerate(self.args[:read_idx_offset]):
            if arg.name in intermediate_bind:
                results.append(shapes[arg.name])
                continue
            # should not exceed original shape
            trimmed_shape = [self.extent_wrapper(i) for i in list(map(min, zip(shapes[arg.name], self.input_buffers[i].shape)))]
            results.append(trimmed_shape)
        return results

    # Propagate inputs only on reduction block
    def propagate_inputs_on_reduction(self, tile, rstep: dict | None = None) -> list[list[int]]:
        if rstep is None:
            rstep = {}
        reduction_block = self.reduction_block
        args = self.block_analyzer.get_input_buffers(reduction_block)
        targets = [t.name for t in args]
        shapes, intermediate_bind = self.propagate(tile, rstep, targets)
        results = []
        for i, arg in enumerate(args):
            if arg.name in intermediate_bind:
                results.append(shapes[arg.name])
                continue
            # should not exceed original shape
            propagate_shape = shapes[arg.name]
            buffer_shape = args[i].shape
            if len(buffer_shape) > len(propagate_shape):
                buffer_shape = buffer_shape[-len(propagate_shape) :]
            trimmed_shape = [self.extent_wrapper(j) for j in list(map(min, zip(propagate_shape, buffer_shape)))]
            results.append(trimmed_shape)
        return results

    def propagate_outputs(self, tile, rstep: dict | None = None) -> list[list[int]]:
        if rstep is None:
            rstep = {}
        read_idx_offset = len(self.input_buffers)
        targets = [t.name for t in self.args[read_idx_offset:]]
        shapes, _ = self.propagate(tile, rstep, targets)
        results = []
        for i, arg in enumerate(self.args[read_idx_offset:]):
            # should not exceed original shape
            trimmed_shape = list(map(min, zip(shapes[arg.name], self.input_buffers[i].shape)))
            results.append(trimmed_shape)
        return results

    def propagate_reduction_inputs(self, shape, rstep: dict | None = None) -> dict[str, list[int]]:
        if rstep is None:
            rstep = {}
        if self.reduction_block is None:
            return {}
        targets = [b.name for b in self.block_analyzer.get_input_buffers(self.reduction_block)]
        results, _ = self.propagate(shape, rstep, targets)
        return results

    def get_reduce_inputs_dtype(self):
        if self.reduction_block is None:
            return {}
        return {b.name: tvm.DataType(b.dtype) for b in self.block_analyzer.get_input_buffers(self.reduction_block)}

    @functools.lru_cache
    def infer_tensorcore_axis(self) -> tuple[int]:
        # axis is fixed for one expression, so only inference and cached
        assert self.get_tag("tensorcore_config")

        C_ax_m, C_ax_n = self.get_tag("tensorcore_config")
        wmma_m, wmma_n, wmma_k = [16, 16, 16]  # just for testing, any number is ok

        output_buffer_shape = self.block_analyzer.sch.get(self.reduction_block).writes[0].buffer.shape
        valid_region = []
        for region in output_buffer_shape:
            if region.value == 1:
                continue
            valid_region.append(region)

        num_nvalid_regions = len(output_buffer_shape) - len(valid_region)
        self.set_tag("num_nvalid_regions", num_nvalid_regions)

        def get_cl_shapes(c_ax_m, c_ax_n, num_nvalid_regions):
            spatial_dim = self.get_space_dim()
            assert len(valid_region) == len(spatial_dim), f" {valid_region} mismatch with {spatial_dim}"
            cl_shapes = [1] * len(spatial_dim)
            cl_shapes[c_ax_m - num_nvalid_regions] = wmma_m
            cl_shapes[c_ax_n - num_nvalid_regions] = wmma_n
            return cl_shapes

        CL_shape = get_cl_shapes(C_ax_m, C_ax_n, num_nvalid_regions)
        self.set_tag("tensorcore_config", [s - num_nvalid_regions for s in [C_ax_m, C_ax_n]])
        shapes = self.propagate_reduction_inputs(CL_shape, {x.var.name: 1 for x in self.raxis})
        A_deps, B_deps = shapes.values()
        A_ax_m = A_deps.index(wmma_m)
        B_ax_n = B_deps.index(wmma_n)

        CL_shape = [1] * len(self.get_space_dim())
        shapes = self.propagate_reduction_inputs(CL_shape, {x.var.name: wmma_k for x in self.raxis})
        A_deps, B_deps = shapes.values()
        A_ax_k = len(A_deps) - 1 - A_deps[::-1].index(wmma_k)
        B_ax_k = len(B_deps) - 1 - B_deps[::-1].index(wmma_k)
        tc_axis = (A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n)
        return tc_axis

    def footprint(self, shape, rstep, stride_map: dict | None = None) -> int:
        if stride_map is None:
            stride_map = {}
        result = 0
        shapes, _ = self.propagate(shape, rstep)

        def is_broadcast_pattern(buffer, output_buffer):
            return (
                buffer in self.args
                and len(shapes[output_buffer.name]) > len(shapes[buffer.name])
                and np.prod(shapes[output_buffer.name]) > np.prod(shapes[buffer.name])
            )

        def is_after_reduce_stage(block):
            if not self.reduction_block:
                return False
            reduce_dependent_blocks = getattr(self, "reduce_dependent_blocks", None)
            if reduce_dependent_blocks is None:
                reduce_dependent_blocks = set()
                pre_order_traverse(
                    self.block_analyzer,
                    [self.reduction_block],
                    lambda block: reduce_dependent_blocks.add(block),
                )
                self.reduce_dependent_blocks = reduce_dependent_blocks
            return block not in reduce_dependent_blocks

        # compute cached stages
        cached_tensor = []
        for block in self.blocks:
            output_buffer = self.block_analyzer.get_output_buffers(block)[0]
            for buffer in self.block_analyzer.get_input_buffers(block):
                cache = buffer.name not in cached_tensor and (
                    is_broadcast_pattern(buffer, output_buffer) or self.block_analyzer.get_block_info(block).is_reduction()
                )
                if not cache:
                    continue
                cached_tensor.append(buffer.name)
                if is_after_reduce_stage(block):
                    continue  # cache after reduce op can often reuse buffer in reduce stage

                if buffer.name in stride_map:
                    num_elem = stride_map[buffer.name].compute_elements_from_shape(shapes[buffer.name])
                else:
                    num_elem = np.prod(shapes[buffer.name])
                buffer_len = num_elem * int((tvm.DataType(buffer.dtype).bits + 7) // 8)
                buffer_len = (buffer_len + 31) // 32 * 32
                result += buffer_len
        return result, cached_tensor

    def get_input_buffers(self) -> list[tir.Buffer]:
        return self.block_analyzer.input_buffers


class OutputNode(Node):
    def __init__(self, node, id=0):
        super().__init__(name="OutputNode")
        # connect node and output node
        assert isinstance(node, PrimFuncNode), "OutputNode should connect to PrimFuncNode"

        # initialize edge and connect
        src_node, src_id = node, id
        edge = Edge(src_node, self, src_id, 0)
        self._in_edges.append(edge)
        src_node._out_edges.append(edge)

        self.set_shape(node.get_shape(id))
        self.set_dtype(node.get_dtype(id))

    def is_output(self):
        return True

    def get_ir(self) -> str:
        return "output"


def topo_order(list_of_nodes) -> list[Node]:
    input_ready_count = {node: len(node.inputs) for node in list_of_nodes}
    ready = list(filter(lambda node: input_ready_count[node] == 0, list_of_nodes))
    output_list = []
    while len(ready) > 0:
        node = ready.pop(0)
        output_list.append(node)
        for edge in node.outputs:
            dst_node = edge.dst_node
            if dst_node not in input_ready_count:
                input_ready_count[dst_node] = len(dst_node.inputs)
                list_of_nodes.append(dst_node)
            input_ready_count[dst_node] -= 1
            assert input_ready_count[dst_node] >= 0
            if input_ready_count[dst_node] == 0:
                ready.append(dst_node)
    assert len(list_of_nodes) == len(output_list)
    return output_list


def find_topo_sort_priority(output_node_list) -> list[Node]:
    import sys

    sys.setrecursionlimit(10000)

    def topo_sort_get_layer(node, topo_layer):
        if node in topo_layer:
            return
        topo_layer[node] = 0
        for edge in node.inputs:
            topo_sort_get_layer(edge.src_node, topo_layer)
            topo_layer[node] = max(topo_layer[node], topo_layer[edge.src_node] + 1)

    topo_layer = {}
    for node in output_node_list:
        topo_sort_get_layer(node, topo_layer)

    def topo_sort_dfs(node, visited, topo_order):
        if node in visited:
            return
        visited.add(node)
        ordered_input_nodes = sorted([edge.src_node for edge in node.inputs], key=lambda n: topo_layer[n], reverse=True)
        for n in ordered_input_nodes:
            topo_sort_dfs(n, visited, topo_order)
        topo_order.append(node)

    visited = set()
    topo_order = []
    for node in output_node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def find_topo_sort(output_node_list) -> list[Node]:
    def topo_sort_dfs(node, visited, topo_order):
        if node in visited:
            return
        visited.add(node)
        for edge in node.inputs:
            topo_sort_dfs(edge.src_node, visited, topo_order)
        topo_order.append(node)

    visited = set()
    topo_order = []
    for node in output_node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order
