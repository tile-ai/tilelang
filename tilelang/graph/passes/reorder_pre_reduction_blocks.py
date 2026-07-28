"""TIR pass that reorders blocks in fused PrimFuncs to put reduction blocks first.

After FuseTIR, blocks are concatenated in Relax binding order.  When the
converter emits a standalone injective operation (e.g. a bias dtype cast)
*before* a matmul, the resulting TIR has::

    [cast_bias, NT_matmul, add_bias, cast_output]

All schedule rules (Matmul, GeneralReduction, Fallback) expect the
reduction block to come first, or at least to be preceded only by
blocks that feed into it.  This pass moves independent injective
pre-blocks to just after the first reduction and before their earliest
consumer, producing::

    [NT_matmul, cast_bias, add_bias, cast_output]
"""

from __future__ import annotations

from collections import deque

from tvm import tirx as tir


def _is_reduction_block(block: tir.SBlock) -> bool:
    """True when the block has at least one ``CommReduce`` iter_var."""
    return any(iv.iter_type == tir.IterVar.CommReduce for iv in block.iter_vars)


def _block_writes(block: tir.SBlock) -> set[tir.buffer.Buffer]:
    """Return the set of buffers written by *block*."""
    return {r.buffer for r in block.writes}


def _block_reads(block: tir.SBlock) -> set[tir.buffer.Buffer]:
    """Return the set of buffers read by *block*."""
    return {r.buffer for r in block.reads}


def _stripped_body(stmt: tir.Stmt) -> tir.SeqStmt | None:
    """Walk through ``AllocBuffer`` / ``AttrStmt`` wrappers to the body SeqStmt."""
    while isinstance(stmt, (tir.AttrStmt, tir.Allocate)):
        stmt = stmt.body
    if isinstance(stmt, tir.SeqStmt):
        return stmt
    return None


def reorder_pre_reduction_blocks(mod):
    """Module-level pass: reorder TIR blocks so the first reduction comes first.

    Applied after FuseTIR and before fuse_all in the TileLang pipeline.
    """
    updated = {}
    for gv, func in mod.functions_items():
        if not isinstance(func, tir.PrimFunc):
            continue
        if func.attrs and func.attrs.get("tirx.is_scheduled", False):
            continue

        new_func = _reorder_one(func)
        if new_func is not None:
            updated[gv] = new_func

    if updated:
        new_mod = mod.clone()
        for gv, func in updated.items():
            new_mod[gv] = func
        return new_mod
    return mod


def _reorder_one(func: tir.PrimFunc) -> tir.PrimFunc | None:
    """Reorder blocks in one PrimFunc.  Returns a new PrimFunc or None (no-op)."""

    # Navigate: func.body → BlockRealize → root Block
    if not isinstance(func.body, tir.SBlockRealize):
        return None
    root_realize = func.body
    root_block = root_realize.block
    if not isinstance(root_block, tir.SBlock):
        return None

    body_seq = _stripped_body(root_block.body)
    if body_seq is None:
        return None

    stmts = list(body_seq.seq)
    if not stmts:
        return None

    # ── collect info about each For / BlockRealize child ──────────────
    class _Info:
        __slots__ = ("idx", "stmt", "block", "is_reduction", "writes", "reads")

        def __init__(self, idx, stmt, block):
            self.idx = idx
            self.stmt = stmt
            self.block = block
            self.is_reduction = _is_reduction_block(block)
            self.writes = _block_writes(block)
            self.reads = _block_reads(block)

    infos: list[_Info] = []
    for i, stmt in enumerate(stmts):
        block = _block_from_stmt(stmt)
        if block is not None:
            infos.append(_Info(i, stmt, block))

    if not infos:
        return None

    # ── find the first reduction ──────────────────────────────────────
    red_idx = next((info.idx for info in infos if info.is_reduction), None)
    if red_idx is None:
        return None

    # ── writer map: buffer → list of info indices that write it ───────
    writer_map: dict[tir.buffer.Buffer, list[int]] = {}
    for info in infos:
        for buf in info.writes:
            writer_map.setdefault(buf, []).append(info.idx)

    # ── transitive producers of the reduction ─────────────────────────
    producers: set[int] = set()
    queue: deque[int] = deque([red_idx])
    while queue:
        cur_idx = queue.popleft()
        for buf in infos[cur_idx].reads:
            for writer_idx in writer_map.get(buf, []):
                if writer_idx not in producers and writer_idx != red_idx:
                    producers.add(writer_idx)
                    queue.append(writer_idx)

    # ── identify independent pre-blocks ───────────────────────────────
    # An "independent pre-block" is a non-reduction block before the
    # reduction that the reduction does NOT transitively read from.
    # It will be moved to just after the reduction but before the
    # earliest consumer that depends on it.
    moves: list[tuple[int, int]] = []  # (insert_after_info_idx, info_to_move_idx)

    for info in infos:
        if info.idx >= red_idx:
            continue
        if info.is_reduction:
            continue
        if info.idx in producers:
            continue

        # Find the earliest consumer after the reduction.
        earliest_consumer: int | None = None
        for buf in info.writes:
            for other in infos:
                if other.idx <= red_idx:
                    continue
                if buf in other.reads and (earliest_consumer is None or other.idx < earliest_consumer):
                    earliest_consumer = other.idx

        if earliest_consumer is not None:
            moves.append((earliest_consumer - 1, info.idx))

    if not moves:
        return None

    # ── reorder ───────────────────────────────────────────────────────
    moves.sort(key=lambda x: x[0])

    moved_indices: set[int] = {m[1] for m in moves}
    new_pairs: list[tuple[int, tir.Stmt]] = []
    for i, stmt in enumerate(stmts):
        if i in moved_indices:
            continue
        new_pairs.append((i, stmt))

    # Insert moves at their target positions, right to left.
    for insert_after, idx_to_move in reversed(moves):
        for pos, (orig_i, _) in enumerate(new_pairs):
            if orig_i == insert_after:
                new_pairs.insert(pos + 1, (idx_to_move, stmts[idx_to_move]))
                break

    # ── rebuild ───────────────────────────────────────────────────────
    new_body = tir.SeqStmt([s for _, s in new_pairs])

    new_root = tir.SBlock(
        root_block.iter_vars,
        root_block.reads,
        root_block.writes,
        root_block.name_hint,
        new_body,
        root_block.init,
        root_block.alloc_buffers,
        root_block.match_buffers,
        root_block.annotations,
    )
    new_realize = tir.SBlockRealize(
        root_realize.iter_values,
        root_realize.predicate,
        new_root,
    )
    return tir.PrimFunc(
        func.params,
        new_realize,
        func.ret_type,
        func.buffer_map,
        func.attrs,
    )


def _block_from_stmt(stmt: tir.Stmt) -> tir.SBlock | None:
    """Extract the ``tir.SBlock`` from a ``For → BlockRealize → Block`` chain."""
    s = stmt
    while isinstance(s, (tir.AttrStmt, tir.For)):
        s = s.body
    if isinstance(s, tir.SBlockRealize):
        return s.block
    if isinstance(s, tir.SBlock):
        return s
    return None
