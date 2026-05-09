"""Unit tests for the EpochGraphBuilder dump emitted from
tl.MergeSharedMemoryAllocations under the
``tl.debug_merge_shared_memory_allocations`` flag.

These tests cover the five S1.4 sub-cases enumerated in the dev notes
(2026-05-09 Phase 4 design):

  (a) sequential ``__syncthreads`` produces back-to-back hard-sync
      epochs in the root scope.
  (b) ``ptx_wait_group`` is recognized as a cp_async_wait epoch
      boundary.
  (c) a ``For`` loop introduces a body scope plus loop_body /
      loop_back / loop_exit edges.
  (d) ``IfThenElse`` introduces parallel then/else scopes plus a
      branch_join epoch in the parent scope.
  (e) nested ``For`` produces nested for_body scopes whose parents
      chain through the outer body.

The tests assert on the textual ``[MSMA-EPOCH-GRAPH]`` dump that is
written to stderr; they do not touch the merge planner itself, which
is still the legacy 1-D model in S1.
"""

from __future__ import annotations

import re
import sys

import pytest

from tilelang import tvm as tvm
import tilelang
import tilelang.testing
from tvm.script import tir as T


def _run_with_dump(func: tvm.tir.PrimFunc, capfd) -> str:
    """Run the merge pass with the debug dump enabled and return stderr."""
    mod = tvm.IRModule({"main": func})
    with tvm.transform.PassContext(
        config={
            "tl.debug_merge_shared_memory_allocations": True,
        }
    ):
        tilelang.transform.MergeSharedMemoryAllocations()(mod)
    sys.stderr.flush()
    out, err = capfd.readouterr()
    return err + out


def _epoch_lines(dump: str) -> list[str]:
    return [ln for ln in dump.splitlines() if ln.startswith("[MSMA-EPOCH-GRAPH]")]


def _scope_lines(lines: list[str]) -> list[str]:
    return [ln for ln in lines if " scope #" in ln]


def _epoch_node_lines(lines: list[str]) -> list[str]:
    return [ln for ln in lines if re.search(r" epoch #\d+ scope=", ln)]


def _edge_lines(lines: list[str]) -> list[str]:
    return [ln for ln in lines if " edge #" in ln]


def _count_edge_kind(lines: list[str], kind: str) -> int:
    return sum(1 for ln in _edge_lines(lines) if f"kind={kind}" in ln)


def _count_epoch_in(lines: list[str], in_kind: str) -> int:
    return sum(1 for ln in _epoch_node_lines(lines) if f" in={in_kind} " in ln)


# Two non-overlapping shared buffers so the merge pass actually enters the
# planner and runs the verbose dump path. The control flow shape, not the
# alias result, is what we are testing here.


@tilelang.testing.requires_cuda
def test_epoch_graph_sequential_hard_syncs(capfd):
    """(a) Two ``__syncthreads`` calls produce three sequential epochs in
    the root scope, joined by two hard-sync seq edges."""

    @T.prim_func(private=True)
    def before(A: T.Buffer((64,), "float16")):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([64], "float16", "shared.dyn")
        Y = T.allocate([64], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 64)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((64,), "float16", data=X, scope="shared.dyn")
        Yb = T.Buffer((64,), "float16", data=Y, scope="shared.dyn")
        Xb[tx] = A[tx]
        T.tvm_storage_sync("shared")
        A[tx] = Xb[tx]
        T.tvm_storage_sync("shared")
        Yb[tx] = A[tx]
        A[tx] = Yb[tx]

    dump = _run_with_dump(before, capfd)
    lines = _epoch_lines(dump)
    assert lines, f"no [MSMA-EPOCH-GRAPH] lines emitted; full dump:\n{dump}"
    # exactly one root scope
    scopes = _scope_lines(lines)
    assert sum(1 for s in scopes if "kind=root" in s) == 1
    # two hard syncs => two `kind=hard` epoch in-edges (epochs 1 and 2)
    assert _count_epoch_in(lines, "hard") == 2, dump


@tilelang.testing.requires_cuda
def test_epoch_graph_cp_async_wait_is_weak_sync(capfd):
    """(b) ``ptx_wait_group`` produces a cp_async_wait epoch boundary."""

    @T.prim_func(private=True)
    def before(A: T.Buffer((64,), "float16")):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([64], "float16", "shared.dyn")
        Y = T.allocate([64], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 64)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((64,), "float16", data=X, scope="shared.dyn")
        Yb = T.Buffer((64,), "float16", data=Y, scope="shared.dyn")
        Xb[tx] = A[tx]
        T.ptx_wait_group(0)
        A[tx] = Xb[tx]
        Yb[tx] = A[tx]
        A[tx] = Yb[tx]

    dump = _run_with_dump(before, capfd)
    lines = _epoch_lines(dump)
    assert _count_epoch_in(lines, "cp_async_wait") == 1, dump


@tilelang.testing.requires_cuda
def test_epoch_graph_for_loop_back_and_exit(capfd):
    """(c) A ``For`` loop opens a for_body scope and contributes both a
    loop_back and a loop_exit edge."""

    @T.prim_func(private=True)
    def before(A: T.Buffer((64, 8), "float16")):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([64], "float16", "shared.dyn")
        Y = T.allocate([64], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 64)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((64,), "float16", data=X, scope="shared.dyn")
        Yb = T.Buffer((64,), "float16", data=Y, scope="shared.dyn")
        for k in range(8):
            Xb[tx] = A[tx, k]
            T.tvm_storage_sync("shared")
            A[tx, k] = Xb[tx]
        Yb[tx] = A[tx, 0]
        A[tx, 0] = Yb[tx]

    dump = _run_with_dump(before, capfd)
    lines = _epoch_lines(dump)
    assert any("kind=for_body" in s for s in _scope_lines(lines)), dump
    assert _count_edge_kind(lines, "loop_body") >= 1, dump
    assert _count_edge_kind(lines, "loop_back") >= 1, dump
    assert _count_edge_kind(lines, "loop_exit") >= 1, dump


@tilelang.testing.requires_cuda
def test_epoch_graph_if_then_else_join(capfd):
    """(d) ``IfThenElse`` creates parallel if_then / if_else scopes and a
    branch_join epoch in the parent scope."""

    @T.prim_func(private=True)
    def before(A: T.Buffer((64,), "float16"), flag: T.int32):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([64], "float16", "shared.dyn")
        Y = T.allocate([64], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 64)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((64,), "float16", data=X, scope="shared.dyn")
        Yb = T.Buffer((64,), "float16", data=Y, scope="shared.dyn")
        if flag == 0:
            Xb[tx] = A[tx]
            A[tx] = Xb[tx]
        else:
            Yb[tx] = A[tx]
            A[tx] = Yb[tx]
        Xb[tx] = A[tx]
        A[tx] = Xb[tx]

    dump = _run_with_dump(before, capfd)
    lines = _epoch_lines(dump)
    scopes = _scope_lines(lines)
    assert any("kind=if_then" in s for s in scopes), dump
    assert any("kind=if_else" in s for s in scopes), dump
    assert _count_edge_kind(lines, "branch") >= 2, dump
    # then-tail + else-tail both join.
    assert _count_edge_kind(lines, "branch_join") >= 2, dump


@tilelang.testing.requires_cuda
def test_epoch_graph_nested_for(capfd):
    """(e) Nested ``For`` produces nested for_body scopes; the inner
    scope's parent is the outer for_body."""

    @T.prim_func(private=True)
    def before(A: T.Buffer((64, 4, 4), "float16")):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([64], "float16", "shared.dyn")
        Y = T.allocate([64], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 64)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((64,), "float16", data=X, scope="shared.dyn")
        Yb = T.Buffer((64,), "float16", data=Y, scope="shared.dyn")
        for i in range(4):
            for j in range(4):
                Xb[tx] = A[tx, i, j]
                T.tvm_storage_sync("shared")
                A[tx, i, j] = Xb[tx]
        Yb[tx] = A[tx, 0, 0]
        A[tx, 0, 0] = Yb[tx]

    dump = _run_with_dump(before, capfd)
    lines = _epoch_lines(dump)
    for_body_scopes = [s for s in _scope_lines(lines) if "kind=for_body" in s]
    assert len(for_body_scopes) >= 2, dump

    outer_ids = []
    inner_parents = []
    for s in for_body_scopes:
        m = re.search(r"scope #(\d+) kind=for_body parent=(-?\d+)", s)
        assert m, s
        sid, parent = int(m.group(1)), int(m.group(2))
        if parent == 0:
            outer_ids.append(sid)
        else:
            inner_parents.append(parent)
    assert any(p in outer_ids for p in inner_parents), dump


@tilelang.testing.requires_cuda
def test_epoch_access_buffer_segregation_across_hard_sync(capfd):
    """S2 sanity: two buffers separated by a hard sync should produce
    distinct ``[MSMA-EPOCH-ACCESS]`` rows whose epoch ids differ.

    Specifically, X is written before the sync and Y is written after,
    so the X-write epoch id must be strictly less than the Y-write
    epoch id.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer((64,), "float16")):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([64], "float16", "shared.dyn")
        Y = T.allocate([64], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 64)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((64,), "float16", data=X, scope="shared.dyn")
        Yb = T.Buffer((64,), "float16", data=Y, scope="shared.dyn")
        Xb[tx] = A[tx]
        A[tx] = Xb[tx]
        T.tvm_storage_sync("shared")
        Yb[tx] = A[tx]
        A[tx] = Yb[tx]

    dump = _run_with_dump(before, capfd)
    access_lines = [ln for ln in dump.splitlines() if ln.startswith("[MSMA-EPOCH-ACCESS]")]
    assert access_lines, f"no [MSMA-EPOCH-ACCESS] lines:\n{dump}"

    def writes_for(buf):
        out = set()
        for ln in access_lines:
            m = re.search(r"epoch=(\d+) buf=(\S+) r=(\d) w=(\d)", ln)
            if m and m.group(2) == buf and int(m.group(4)) == 1:
                out.add(int(m.group(1)))
        return out

    x_writes = writes_for("Xb")
    y_writes = writes_for("Yb")
    assert x_writes, dump
    assert y_writes, dump
    # X must not be written in any epoch that also writes Y
    assert x_writes.isdisjoint(y_writes), dump
    # X's earliest write epoch must precede Y's earliest write epoch
    assert min(x_writes) < min(y_writes), dump


@tilelang.testing.requires_cuda
def test_epoch_liveness_propagates_across_skipped_epoch(capfd):
    """S3 sanity: buffer X is written in epoch 0, NOT touched in epoch 1
    (some other buffer Y is touched there), then read in epoch 2. The
    backward liveness analyzer must mark X as live across the gap epoch
    so the merge planner cannot reuse X's storage in epoch 1.

    This is the exact pattern the naive "touched-epoch" model gets
    wrong; the dump produced by ``ComputePerEpochLiveness`` must show a
    ``[MSMA-EPOCH-LIVE] epoch=<gap> buf=Xb`` line where the gap epoch
    is **not** present in the EPOCH-ACCESS rows for Xb.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer((64,), "float16")):
        T.launch_thread("blockIdx.x", 1)
        X = T.allocate([64], "float16", "shared.dyn")
        Y = T.allocate([64], "float16", "shared.dyn")
        tx = T.launch_thread("threadIdx.x", 64)
        T.launch_thread("threadIdx.y", 1)
        T.launch_thread("threadIdx.z", 1)
        Xb = T.Buffer((64,), "float16", data=X, scope="shared.dyn")
        Yb = T.Buffer((64,), "float16", data=Y, scope="shared.dyn")
        Xb[tx] = A[tx]
        T.tvm_storage_sync("shared")
        Yb[tx] = A[tx]
        A[tx] = Yb[tx]
        T.tvm_storage_sync("shared")
        A[tx] = Xb[tx]

    dump = _run_with_dump(before, capfd)
    live_lines = [ln for ln in dump.splitlines() if ln.startswith("[MSMA-EPOCH-LIVE]")]
    assert live_lines, f"no [MSMA-EPOCH-LIVE] lines emitted:\n{dump}"

    x_live_epochs = set()
    for ln in live_lines:
        m = re.search(r"epoch=(\d+) buf=(\S+) in=(\d) out=(\d)", ln)
        if m and m.group(2) == "Xb":
            x_live_epochs.add(int(m.group(1)))

    access_lines = [ln for ln in dump.splitlines() if ln.startswith("[MSMA-EPOCH-ACCESS]")]
    x_touched = set()
    for ln in access_lines:
        m = re.search(r"epoch=(\d+) buf=(\S+) ", ln)
        if m and m.group(2) == "Xb":
            x_touched.add(int(m.group(1)))

    gap_live = x_live_epochs - x_touched
    assert gap_live, (
        f"liveness analyzer failed to mark cross-epoch live region for Xb; touched={sorted(x_touched)} live={sorted(x_live_epochs)}\n{dump}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
