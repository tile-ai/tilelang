"""A fallback schedule rule for GPU operators."""

from ... import Schedule as TileSchedule

from . import utils
from .base import GPUScheduleRule

from tvm import s_tir, tirx as tir
from tvm.target import Target
from tvm.s_tir.dlight import normalize_prim_func, try_inline


class Fallback(GPUScheduleRule):
    """
    A fallback schedule rule for all GPU operators. It will try to inline all the blocks first,
    and then apply a simple block/grid mapping to the spatial loops on top of the remaining blocks.
    """

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> s_tir.Schedule:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None
        max_threads_per_block = utils.max_threads_per_block(target)

        sch = TileSchedule(func)
        block_infos = normalize_prim_func(sch)

        if block_infos is None:
            return None

        block_infos = try_inline(sch, block_infos)
        if block_infos is None:
            return None

        param_buffers = [func.buffer_map[param] for param in func.params]
        block_records = []
        for block in block_infos:
            s_loops: list[s_tir.schedule.LoopRV] = []
            r_loops: list[s_tir.schedule.LoopRV] = []
            o_loops: list[s_tir.schedule.LoopRV] = []
            dom_kind = block.dom_kind()
            block = block.block_rv

            if any([sch.get(loop_rv).thread_binding is not None for loop_rv in sch.get_loops(block)]) or len(sch.get_loops(block)) == 0:
                continue

            for loop, iter_type in zip(sch.get_loops(block), dom_kind):
                {"S": s_loops, "R": r_loops, "O": o_loops}[iter_type].append(loop)

            if not s_loops:
                s_loops.append(sch.add_unit_loop(block))
            sch.reorder(*s_loops, *r_loops, *o_loops)
            bx, tx = sch.split(  # pylint: disable=invalid-name
                sch.fuse(*s_loops),
                factors=[None, max_threads_per_block],
            )
            sch.bind(bx, "blockIdx.x")
            sch.bind(tx, "threadIdx.x")

            block_stmt = sch.get(block)
            block_records.append(
                {
                    "name": block_stmt.name_hint,
                    "bx": bx,
                    "intermediate_buffers": [
                        write.buffer for write in block_stmt.writes if not any(buf.same_as(write.buffer) for buf in param_buffers)
                    ],
                }
            )

        for idx in reversed(range(len(block_records))):
            record = block_records[idx]
            if not record["intermediate_buffers"]:
                continue

            for consumer_record in reversed(block_records[idx + 1 :]):
                consumer_block = sch.get_sblock(consumer_record["name"])
                consumer_stmt = sch.get(consumer_block)
                if not any(any(read.buffer.same_as(buf) for buf in record["intermediate_buffers"]) for read in consumer_stmt.reads):
                    continue
                try:
                    sch.reverse_compute_at(
                        consumer_block,
                        record["bx"],
                        preserve_unit_loops=True,
                    )
                except Exception:
                    continue
                consumer_block = sch.get_sblock(consumer_record["name"])
                consumer_stmt = sch.get(consumer_block)
                consumer_loops = sch.get_loops(consumer_block)
                if consumer_stmt.init is None and consumer_loops:
                    sch.parallel(consumer_loops[-1])

            block = sch.get_sblock(record["name"])
            block_stmt = sch.get(block)
            for write_buffer_index, write_region in reversed(list(enumerate(block_stmt.writes))):
                if any(buf.same_as(write_region.buffer) for buf in param_buffers):
                    continue
                sch.cache_write_at(
                    record["bx"],
                    block,
                    write_buffer_index,
                    "shared",
                    write_back=False,
                )
                block = sch.get_sblock(record["name"])

        return sch
