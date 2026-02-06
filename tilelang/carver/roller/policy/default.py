"""Policy for cuda core schedule"""

from __future__ import annotations
import functools
import math
from queue import PriorityQueue
from collections.abc import Iterable

import numpy as np
import tvm

from ...arch import TileDevice
from ..bestfit import BestFit
from ..hint import Hint, Stride, TileDict
from .common import coalesced_factor, coalesced_tensor_shape, factorize, get_all_factors
from ..node import PrimFuncNode, OutputNode, find_topo_sort
from ..rasterization import NoRasterization


class DefaultPolicy:
    """
    Default Policy for fastdlight, a heuristic plan that tries to
    minimize memory traffic and maximize parallelism.for BitBLAS Schedule.
    """

    func: tvm.tir.PrimFunc
    nodes: list[PrimFuncNode] = []
    arch: TileDevice
    tags: dict

    def __init__(self, arch: TileDevice, tags: dict | None = None) -> None:
        if tags is None:
            tags = {}

        self.arch = arch
        self.tags = tags
        self.rasterization = NoRasterization()

    @classmethod
    def from_prim_func(cls, func: tvm.tir.PrimFunc, arch: TileDevice, tags: dict | None = None, name: str = "PrimFuncNode"):
        return cls(arch, tags)._init_with_prim_func(func, name)

    @classmethod
    def from_output_nodes(cls, nodes: list[OutputNode], arch: TileDevice, tags: dict | None = None):
        return cls(arch, tags)._init_with_output_nodes(nodes)

    def _init_with_prim_func(self, func: tvm.tir.PrimFunc, name: str = "PrimFuncNode") -> DefaultPolicy:
        if func is not None and isinstance(func, tvm.tir.PrimFunc):
            self.func = func
            self.prim_func_node = PrimFuncNode(self.func, tags=self.tags, name=name)
        else:
            raise NotImplementedError("Only support PrimFunc for now")
        output_nodes = [OutputNode(self.prim_func_node)]
        self._init_with_output_nodes(output_nodes)
        return self

    def _init_with_output_nodes(self, output_nodes: list[OutputNode]):
        self.ordered_nodes = list(filter(lambda n: not n.is_placeholder() and not n.is_output(), find_topo_sort(output_nodes)))
        for node in self.ordered_nodes:
            node.update_tags(self.tags)

        self.output_nodes = []
        for node in self.ordered_nodes:
            is_topo_output = True
            for edge in node.outputs:
                if not edge.dst_node.is_output():
                    is_topo_output = False
            if is_topo_output:
                self.output_nodes.append(node)
        return self

    def emit_config(self, topk: int) -> list[Hint]:
        base_tile = self.get_base_tile()
        if base_tile is None:
            return []

        rstep_map = {node: self._assign_reduce_step(node) for node in self.ordered_nodes}
        smem_tile_condidates = self.dfs_smem_tile(base_tile, rstep_map)
        results = []
        for td in smem_tile_condidates:
            if not self.check_tile_shape_isvalid(td):
                continue

            self._expand_reduce_axis(td)
            for codegen_dicts in self.assign_block_size(td):
                if isinstance(codegen_dicts, dict) and len(codegen_dicts) == 1:
                    results.append(list(codegen_dicts.values())[0])
                else:
                    results.append(codegen_dicts)
                if len(results) >= topk:
                    break
            if len(results) >= topk:
                break
        return results

    def dfs_smem_tile(self, init_tile, rstep_map) -> Iterable[TileDict]:
        _steps = [get_all_factors(n) for n in self.output_nodes[0].get_space_dim()]
        steps = [step[step.index(t) :] for step, t in zip(_steps, init_tile)]
        for i in range(len(steps)):
            added = list(
                filter(
                    lambda s: s < steps[i][-1] and s > steps[i][0] and s not in steps[i],
                    [2, 4, 8, 16, 32],
                )
            )
            steps[i].extend(added)
            steps[i] = sorted(steps[i])
        visited_tiles = {}
        queue = PriorityQueue()

        def prio(td: TileDict):
            return (td.traffic + 1) * td.num_wave

        def add_to_queue(tile):
            if tuple(tile) in visited_tiles:
                return
            td = self.compute_tile_dict(tile, rstep_map)
            visited_tiles[tuple(tile)] = td
            if td.valid:
                queue.put([prio(td), tile])

        add_to_queue(init_tile)
        while not (queue.empty() or len(visited_tiles) > 2000):
            _, tile = queue.get()
            dim_ids = [step.index(t) for step, t in zip(steps, tile)]
            for i in reversed(range(len(dim_ids))):
                if dim_ids[i] + 1 < len(steps[i]):
                    new_tile = tile.copy()
                    new_tile[i] = steps[i][dim_ids[i] + 1]
                    add_to_queue(new_tile)

        visited_tiles = filter(lambda td: td.valid, visited_tiles.values())
        sorted_tiles = sorted(visited_tiles, key=lambda td: prio(td))
        return sorted_tiles

    def get_base_tile(self):
        """
        Gets the minimum tile configuration that satisfies no redundancy in computation.

        Returns
        -------
        List[int]
            The base tile configuration, which is a list of 1s equal in length to the space dimensions
            of the primary function node.
        """
        if len(set([len(node.get_space_dim()) for node in self.output_nodes])) > 1:
            # If output dim sizes are not same, don't know how to handle them
            return None

        out_node = self.output_nodes[0]
        shape = out_node.get_space_dim()
        base_tile = [1 for _ in shape]
        wpi = self.compute_workload_per_item(base_tile)
        for dim, n in enumerate(shape):
            factors = [n]
            for factor in factors:
                if factor == base_tile[dim]:
                    continue
                tile = base_tile.copy()
                tile[dim] = factor
                new_wpi = self.compute_workload_per_item(tile)
                if new_wpi < wpi:
                    wpi, base_tile = new_wpi, tile
                else:
                    break

        return base_tile

    # handles multiple output cases
    def _get_output_tile_map(self, tile):
        """
        Handles multiple output cases by mapping output nodes to their respective tile configurations.

        Parameters
        ----------
        tile : List[int]
            The tile configuration.

        Returns
        -------
        Dict
            A dictionary mapping the primary function node to its corresponding tile configuration
            based on the output nodes' space dimensions.
        """
        tile_map = {}
        for node in self.output_nodes:
            tile_map[node] = [tile[i] * node.get_space_dim()[i] // self.output_nodes[0].get_space_dim()[i] for i in range(len(tile))]
        return tile_map

    def compute_workload_per_item(self, output_tile) -> float:
        op_tile_map = self._get_output_tile_map(output_tile)
        compute = 0
        num_item = int(np.prod(output_tile))
        for node in reversed(self.ordered_nodes):
            tile = op_tile_map[node]
            dep = node.propagate_inputs(tile)
            compute += int(np.prod(tile))
            for i, edge in enumerate(node.inputs):
                op_tile_map[edge.src_node] = dep[i]
        return float(compute / num_item)

    def score_block_size(self, n):
        """
        Scores a block size based on its efficiency and fit relative to the architecture's warp size and SM partition.

        Parameters
        ----------
        n : int
            The block size to score.

        Returns
        -------
        Tuple[float, float]
            A tuple containing two scores representing efficiency and fit, respectively.
        """
        num_wrap = (n + self.arch.warp_size - 1) // self.arch.warp_size
        r1 = max(num_wrap / self.arch.sm_partition, self.arch.sm_partition / num_wrap)
        r2 = (num_wrap * self.arch.warp_size - n) / n
        return (r1, r2)

    def get_block_size(self, n):
        """
        Determines the optimal block size for a given constraint, based on scoring various factors.

        Parameters
        ----------
        n : int
            The constraint size.

        Returns
        -------
        int
            The optimal block size chosen from the factors of n, constrained by a maximum of 1024 and
            scored by the `score_block_size` method.
        """
        factors = get_all_factors(n)
        factors = list(filter(lambda x: x <= 1024, factors))
        factor_ordered = sorted(factors, key=self.score_block_size)
        return factor_ordered[0]

    def get_node_reduce_step_candidates(self, node: PrimFuncNode):
        """
        Calculates reduction step candidates for each reduction axis in a PrimFuncNode. General idea : use factor first, since it does not require extra boundary check. for large prime number, which is rare case, use power of 2.

        Parameters
        ----------
        node : PrimFuncNode
            The node for which to calculate reduction step candidates. It contains reduction axes (raxis)
            with their domains (dom.extent).

        Returns
        -------
        Dict[str, List[int]]
            A dictionary mapping axis variable names to lists of step candidates. For each axis in the node,
            this function calculates possible step sizes. For axes with a large prime domain, it uses powers of 2
            as step candidates; for others, it uses all factors of the domain.
        """

        results = {}
        for k_iter in node.raxis:
            all_factors = get_all_factors(int(k_iter.dom.extent))
            if len(all_factors) == 2 and int(k_iter.dom.extent) > 64:
                all_factors = [1]
                while all_factors[-1] * 2 < int(k_iter.dom.extent):
                    all_factors.append(all_factors[-1] * 2)
            results[k_iter.var.name] = all_factors
        return results

    def _assign_reduce_step(self, node: PrimFuncNode):
        """
        Assigns an optimal reduction step for the given PrimFuncNode.

        Parameters
        ----------
        node : PrimFuncNode
            The node for which the reduction step is to be assigned.

        Returns
        -------
        Dict
            A dictionary mapping reduction axis variable names to their optimal reduction steps.
        """
        if node.reduction_block is None:
            return {}

        raxis = node.raxis
        tile = [1] * len(node.get_space_dim())
        all_steps = self.get_node_reduce_step_candidates(node)

        def sim(a: int, b: int):
            return (2 * a * b) / (a * a + b * b)

        def _score(rstep_id):
            rstep = {k: all_steps[k][rstep_id[k]] for k in rstep_id}
            score = 0
            shape = node.propagate_inputs(tile, rstep=rstep)
            for i, input_buffer in enumerate(node.input_buffers):
                read_transaction_elements = self.arch.transaction_size[1] // ((node.get_buffer_dtype(input_buffer).bits + 7) // 8)
                score += sim(
                    int(coalesced_factor(shape[i], input_buffer.shape)),
                    read_transaction_elements,
                )
            return score

        def _enlarge(rstep_id):
            candidates = []
            candidates.append((rstep_id, _score(rstep_id)))
            for ax in rstep_id:
                if rstep_id[ax] + 1 == len(all_steps[ax]):
                    continue
                r = rstep_id.copy()
                r[ax] += 1
                candidates.append((r, _score(r)))
            best = max(candidates, key=lambda x: x[1])
            return best

        # enlarge rstep to ensure read is coaleased
        cur_rstep_id = {ax.var.name: 0 for ax in raxis}
        cur_score = _score(cur_rstep_id)
        while True:
            if cur_score == 0:
                break
            new_rstep, new_score = _enlarge(cur_rstep_id)
            if new_score <= cur_score:
                break
            else:
                cur_rstep_id, cur_score = new_rstep, new_score
        rstep = {k: all_steps[k][cur_rstep_id[k]] for k in cur_rstep_id}
        return rstep

    def _expand_reduce_axis(self, td: TileDict):
        """
        Expands the reduction axis in the TileDict based on shared memory limits.

        Parameters
        ----------
        td : TileDict
            The TileDict object to be optimized.

        Returns
        -------
        None
            This function modifies the TileDict in place.
        """
        smem_limit = min(self.arch.max_smem_usage // td.block_per_SM, self.arch.smem_cap)
        rstep_map = td.rstep_map.copy()

        def _optimize(node, rstep):
            all_steps = self.get_node_reduce_step_candidates(node)
            for k in all_steps:
                all_steps[k] = list(filter(lambda x: x % rstep[k] == 0, all_steps[k]))

            def _score(rstep_id):
                rstep = {k.var.name: all_steps[k.var.name][rstep_id[k.var.name]] for k in node.raxis}
                score = 0
                shape = node.propagate_inputs(td.get_tile(node), rstep=rstep)
                for i, input_buffer in enumerate(node.input_buffers):
                    score += coalesced_factor(shape[i], input_buffer.shape)
                return score

            def _enlarge(rstep_id):
                candidates = []
                for ax in rstep_id:
                    if rstep_id[ax] + 1 == len(all_steps[ax]):
                        continue
                    r = rstep_id.copy()
                    r[ax] += 1
                    candidates.append((r, _score(r)))
                if len(candidates) == 0:
                    return None
                return max(candidates, key=lambda x: x[1])[0]

            cur_rstep_id = {k.var.name: all_steps[k.var.name].index(rstep[k.var.name]) for k in node.raxis}
            new_rstep_map = rstep_map.copy()
            while True:
                new_rstep_id = _enlarge(cur_rstep_id)
                if new_rstep_id is None:
                    break
                new_rstep_map[node] = {k.var.name: all_steps[k.var.name][new_rstep_id[k.var.name]] for k in node.raxis}
                old_rstep_map = td.rstep_map
                td.rstep_map = new_rstep_map
                smem_usage, _ = self._compute_shared_memory_usage(td)
                td.rstep_map = old_rstep_map
                if smem_usage > smem_limit:
                    break
                else:
                    cur_rstep_id = new_rstep_id
            rstep = {k.var.name: all_steps[k.var.name][cur_rstep_id[k.var.name]] for k in node.raxis}
            return rstep

        for node in self.ordered_nodes:
            if len(node.raxis) > 0:
                rstep = _optimize(node, rstep_map[node])
                rstep_map[node] = rstep
        td.rstep_map = rstep_map
        td.smem_cost, td.cached_tensors_map = self._compute_shared_memory_usage(td)

    def _compute_memory_traffic(self, output_tile):
        """
        Computes the memory traffic for a given output tile configuration.

        Parameters
        ----------
        output_tile : List[int]
            The output tile configuration.

        Returns
        -------
        Tuple[int, Dict]
            The total memory traffic and a map of operation tiles.
        """
        op_tile_map = self._get_output_tile_map(output_tile)
        traffic = 0
        for node in reversed(self.ordered_nodes):
            tile = op_tile_map[node]
            input_shapes = node.propagate_inputs(tile)
            output_shapes = node.propagate_outputs(tile)
            for i, edge in enumerate(node.inputs):
                op_tile_map[edge.src_node] = input_shapes[i]
                if edge.src_node.is_placeholder():
                    nbytes = (edge.src_node.get_dtype().bits + 7) // 8
                    read_transaction_elements = self.arch.transaction_size[1] // nbytes
                    traffic += coalesced_tensor_shape(input_shapes[i], edge.src_node.get_shape(), read_transaction_elements) * nbytes
            for edge in node.outputs:
                if edge.dst_node.is_output():
                    nbytes = (edge.src_node.get_dtype().bits + 7) // 8
                    write_transaction_elements = self.arch.transaction_size[0] // nbytes
                    traffic += (
                        coalesced_tensor_shape(output_shapes[edge.src_id], node.get_shape(edge.src_id), write_transaction_elements) * nbytes
                    )

        return traffic, op_tile_map

    def infer_node_smem_usage(self, td: TileDict, node: PrimFuncNode):
        """
        Infers the shared memory usage of a node given a TileDict configuration.

        Parameters
        ----------
        td : TileDict
            The TileDict object containing the tile configuration.
        node : PrimFuncNode
            The node for which to infer the shared memory usage.

        Returns
        -------
        int
            The estimated amount of shared memory used by the node.
        """
        return node.footprint(td.get_tile(node), td.get_rstep(node), td.tensor_strides_map[node])

    def _compute_shared_memory_usage(self, td: TileDict):
        """
        Computes the stride map for a given node and TileDict configuration.

        Parameters
        ----------
        node : PrimFuncNode
            The node for which to compute the stride map.
        td : TileDict
            The TileDict object containing the tile configuration.

        Returns
        -------
        Tuple[Dict, Dict]
            The output strides and tensor strides.
        """
        self._compute_stride_map(td)
        allocator = BestFit()
        block_map = {}
        processed = set()
        cached_tensors_map = {}

        def can_free(node, out_id):
            return all(not (edge.src_id == out_id and edge.dst_node not in processed) for edge in node.outputs)

        for node in self.ordered_nodes:
            node_internal_bytes, cached_tensors_map[node] = self.infer_node_smem_usage(td, node)
            block = allocator.malloc(node_internal_bytes)
            allocator.free(block)
            # free inputs
            processed.add(node)
            for edge in node.inputs:
                if not edge.src_node.is_placeholder() and can_free(edge.src_node, edge.src_id):
                    allocator.free(block_map.pop((edge.src_node, edge.src_id)))
            # alloc outputs
            for edge in node.outputs:
                if not edge.dst_node.is_output() and (node, edge.src_id) not in block_map:
                    dtype_bytes = (node.get_dtype(edge.src_id).bits + 7) // 8
                    stride = td.output_strides_map[node][len(node.inputs) + edge.src_id]
                    output_elem = stride.compute_elements_from_shape(td.get_tile(node))
                    block_map[(node, edge.src_id)] = allocator.malloc(output_elem * dtype_bytes)

        assert len(block_map) == 0
        return allocator.limit, cached_tensors_map

    def compute_node_stride_map(self, node: PrimFuncNode, td: TileDict):
        """
        Computes the stride map for a given node based on the TileDict configuration.

        Parameters
        ----------
        node : PrimFuncNode
            The node for which to compute the stride map.
        td : TileDict
            The TileDict object containing the tile configuration.

        Returns
        -------
        Tuple[Dict, Dict]
            A tuple of dictionaries containing the output strides and tensor strides.
        """
        output_strides = {int(i + len(node.input_buffers)): Stride() for i, _ in enumerate(node.output_buffers)}
        tensor_strides = {}
        return output_strides, tensor_strides

    def _compute_stride_map(self, td: TileDict):
        """
        Computes the stride map for all nodes in a TileDict.

        Parameters
        ----------
        td : TileDict
            The TileDict object for which to compute the stride maps.

        Returns
        -------
        None
            This function updates the TileDict object in place with the computed stride maps.
        """
        output_strides_map = {}
        tensor_strides_map = {}
        for node in self.ordered_nodes:
            output_strides_map[node], tensor_strides_map[node] = self.compute_node_stride_map(node, td)
        td.output_strides_map, td.tensor_strides_map = output_strides_map, tensor_strides_map

    def compute_tile_dict(self, output_tile: list[int], rstep_map) -> TileDict:
        """
        Computes and returns a TileDict object for a given output tile configuration and reduction step map.

        Parameters
        ----------
        output_tile : List[int]
            The output tile configuration.
        rstep_map : Dict
            The reduction step map.

        Returns
        -------
        TileDict
            A TileDict object containing the computed tile configuration, memory traffic, shared memory cost,
            grid size, and other related parameters.
        """
        td = TileDict(output_tile)
        td.rstep_map = rstep_map
        td.traffic, td.tile_map = self._compute_memory_traffic(output_tile)
        td.smem_cost, td.cached_tensors_map = self._compute_shared_memory_usage(td)
        if td.smem_cost > self.arch.smem_cap:
            td.valid = False
            return td
        output_shape = self.output_nodes[0].get_space_dim()
        td.grid_size = int(np.prod([(y + x - 1) // x for x, y in zip(output_tile, output_shape)]))
        # estimated reg usage
        reg_usage = int(2 * max([np.prod(td.get_tile(node)) * node.get_dtype().bits / 32 for node in self.ordered_nodes]))
        if reg_usage > self.arch.reg_cap:
            td.valid = False
            return td
        td.block_per_SM = min(
            self.arch.max_smem_usage // max(td.smem_cost, 1),
            self.arch.reg_cap // max(reg_usage, 1),
            self.arch.sm_partition,
        )
        td.num_wave = int(np.ceil(td.grid_size / int(td.block_per_SM * self.arch.compute_max_core)))
        return td

    def check_tile_shape_isvalid(self, td: TileDict) -> bool:
        """
        Checks if the tile shapes in the TileDict are valid for the nodes in this context.

        Parameters:
        - td (TileDict): The TileDict object containing tile shapes and other configurations.

        Returns:
        - bool: True if all tile shapes are valid, False otherwise.
        """
        for node in self.ordered_nodes:
            if np.prod(td.get_tile(node)) == 0:
                return False
            node_grid_size = np.prod([(y + x - 1) // x for x, y in zip(td.get_tile(node), node.get_space_dim())])
            if node_grid_size != td.grid_size:
                return False
            if hasattr(node, "reduce_op") and node.reduce_op is not None and len(node.reduce_op.axis) == len(td.output_tile):
                for i, tile_extent in enumerate(td.output_tile):
                    if node.reduce_op.axis[i].dom.extent % tile_extent:
                        return False

        return True

    def recommend_block_size(self, td: TileDict) -> list[int]:
        """
        Recommends optimal block sizes based on the TileDict configuration.

        Parameters
        ----------
        td : TileDict
            The TileDict object containing the tile configuration.

        Returns
        -------
        List[int]
            A list of recommended block sizes sorted based on their score.
        """
        node_space_sizes = [int(np.prod(td.get_tile(node))) for node in self.ordered_nodes]
        max_block_size = functools.reduce(math.gcd, node_space_sizes)

        if max_block_size < self.arch.warp_size * self.arch.sm_partition and max_block_size == min(node_space_sizes):
            node_reduce_sizes = [int(np.prod(list(td.get_rstep(node).values()))) for node in self.ordered_nodes]
            total_sizes = [x * y for x, y in zip(node_space_sizes, node_reduce_sizes)]
            max_possible_size = functools.reduce(math.gcd, total_sizes)
            possible_block_sizes = list(
                filter(
                    lambda x: x % max_block_size == 0 and x <= 1024,
                    get_all_factors(max_possible_size),
                )
            )
            possible_block_sizes = list(
                filter(  # either be a factor of space or cover fully cover the space
                    lambda x: all([x % s == 0 or s % x == 0 for s in node_space_sizes]),
                    possible_block_sizes,
                )
            )
            factor_ordered = sorted(possible_block_sizes, key=self.score_block_size)
            return factor_ordered
        else:
            possible_block_sizes = get_all_factors(max_block_size)
            possible_block_sizes = list(filter(lambda x: x <= 1024, possible_block_sizes))
        factor_ordered = sorted(possible_block_sizes, key=self.score_block_size)
        return factor_ordered

    def assign_block_size(self, td: TileDict, topk=1):
        """
        Assigns block sizes to the TileDict based on the recommended block sizes.

        Parameters
        ----------
        td : TileDict
            The TileDict object to assign block sizes to.
        topk : int, optional
            The number of top block sizes to consider.

        Yields
        -------
        Dict
            The block size assignment for the primary function node.
        """
        block_size_ordered = self.recommend_block_size(td)
        for block_size in block_size_ordered:
            result = {}
            failed = False
            for node in self.ordered_nodes:
                result[node] = self._assign_block_size(node, td, block_size)
                if result[node] is None:
                    failed = True
                    break
            if failed:
                continue
            else:
                yield result
                topk -= 1
                if topk == 0:
                    break

    def _assign_block_size(self, node: PrimFuncNode, td: TileDict, block_size: int):
        """
        Assigns a block size to a given PrimFuncNode based on the TileDict configuration and the specified block size.

        Parameters
        ----------
        node : PrimFuncNode
            The node to assign the block size to.
        td : TileDict
            The TileDict object containing the tile configuration.
        block_size : int
            The block size to be assigned.

        Returns
        -------
        Hint
            A Hint object containing the assigned block size and other related settings.
        """
        tile, rsteps = td.get_tile(node), td.get_rstep(node)
        factors = factorize(block_size)
        cur_threads = [1 for _ in tile]
        reduce_thread = {k: 1 for k in rsteps}
        ndim = len(tile)

        def _score(node, thread):  # small is better
            score = 0
            block_tile = [int(np.ceil(tile[i] / thread[i])) for i in range(ndim)]
            shape = node.propagate_inputs(block_tile)
            for i, _ in enumerate(node.input_buffers):
                score += np.prod(shape[i]) / self.arch.bandwidth[1]
            for buffer in node.output_buffers:
                score += coalesced_tensor_shape(thread, buffer.shape, 8) / self.arch.bandwidth[0]
            return score

        for factor in reversed(factors):
            score_map = {}
            for i in range(ndim):
                if cur_threads[i] >= tile[i]:
                    continue
                if (tile[i] % (cur_threads[i] * factor)) != 0:
                    continue
                cur_threads[i] *= factor
                score_map[i] = (_score(node, cur_threads), i)
                cur_threads[i] //= factor
            if len(score_map) > 0:
                # assign to space axis
                dim_order = sorted(score_map.keys(), key=lambda x: score_map[x])
                cur_threads[dim_order[0]] *= factor
            else:
                # assign to reduce axis
                target_ax = None
                for ax, ax_len in reversed(list(rsteps.items())):
                    if ax_len % (reduce_thread[ax] * factor) == 0:
                        target_ax = ax
                        break
                assert target_ax
                reduce_thread[target_ax] *= factor

        codegen_dict = Hint()
        codegen_dict.block = tile
        codegen_dict.thread = cur_threads
        codegen_dict.rstep = [rsteps[ax.var.name] for ax in node.raxis]
        codegen_dict.reduce_thread = [reduce_thread[ax.var.name] for ax in node.raxis]
        codegen_dict.cached_tensors = td.cached_tensors_map[node]
        codegen_dict.rasterization_plan = self.plan_rasterization(td)

        if node.get_dtype().bits == 16:  # set step=2 for 16bit case to ensure coalesced access
            codegen_dict._step = [1 for _ in range(ndim)]
            for i in reversed(range(ndim)):
                if codegen_dict.block[i] // codegen_dict.thread[i] % 2 == 0:
                    codegen_dict._step[i] = 2
                    break
        elif node.get_dtype().bits == 8:  # set step=4 for 8bit case to ensure coalesced access
            codegen_dict._step = [1 for _ in range(ndim)]
            for i in reversed(range(ndim)):
                if codegen_dict.block[i] // codegen_dict.thread[i] % 4 == 0:
                    codegen_dict._step[i] = 4
                    break
        # Plan vectorize
        codegen_dict.vectorize = self._plan_vectorize(node, td, block_size)
        codegen_dict.arch = self.arch
        codegen_dict.opt_shapes = node.get_tag("opt_shapes")
        return codegen_dict

    def _plan_vectorize(self, node: PrimFuncNode, td: TileDict, block_size: int):
        """
        Plans vectorization for a given PrimFuncNode based on the TileDict configuration and block size.

        Parameters
        ----------
        node : PrimFuncNode
            The node for which to plan vectorization.
        td : TileDict
            The TileDict object containing the tile configuration.
        block_size : int
            The block size used for vectorization planning.

        Returns
        -------
        Dict
            A dictionary mapping tensors to their vectorization size.
        """

        def is_cont(shape, vec):
            if len(shape) == 0:
                return vec == 1
            last = shape[-1]
            if last == 1:
                return is_cont(shape[0:-1], vec // last)
            else:
                return last % vec == 0

        def is_shape_aligned(shape, factor):
            return int(np.prod(shape)) % factor == 0

        def is_type_allowed(dtype, vec):
            return dtype.bits * vec <= 128

        vectorize_sizes = [16, 8, 4, 2]
        dtypes = node.get_reduce_inputs_dtype()
        shapes = node.propagate_reduction_inputs(td.get_tile(node), td.get_rstep(node))
        vectorize_result = {}
        for tensor, shape in shapes.items():
            for v in vectorize_sizes:
                if is_shape_aligned(shape, block_size * v) and is_cont(shape, v) and is_type_allowed(dtypes[tensor], v):
                    vectorize_result[tensor] = v
                    break
        return vectorize_result

    def plan_rasterization(self, td: TileDict):  # pylint: disable=unused-argument
        """
        Plans the rasterization for the given TileDict. This function is not implemented yet.

        Parameters
        ----------
        td : TileDict
            The TileDict object to plan rasterization for.

        Raises
        -------
        RasterRationPlan
            This function is not implemented yet.
        """
        return NoRasterization()
