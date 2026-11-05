"""On-chip capacity check in MergeUBAllocations.

The pass must reject a plan whose footprint exceeds the physical capacity of the
targeted scope, with a short message naming the scope and the required bytes, and
must stay silent when no capacity is known (no target, or a non-Ascend target).
"""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx

UB_CAPACITY = 248 * 1024
SIMT_VF_UB_CAPACITY = UB_CAPACITY - 32 * 1024

# (scope, display name, per-buffer elements, capacity); every entry overflows.
OVERFLOW_SCOPES = [
    ("shared.dyn", "UB", 32768, UB_CAPACITY),  # 2 x 128 KB
    ("shared.l1", "L1", 76800, 512 * 1024),  # 2 x 300 KB
    ("shared.l0c", "L0C", 51200, 256 * 1024),  # 2 x 200 KB
    ("shared.l0a", "L0A", 12288, 64 * 1024),  # 2 x 48 KB
    ("shared.l0b", "L0B", 12288, 64 * 1024),  # 2 x 48 KB
]


def module(buffers, body=None, contract=None, target="ascend", wrap=None, aiv=None):
    """Module whose buffers are allocated at the top level.

    ``buffers`` is a list of ``(name, shape, dtype, scope)``. Pass ``target=None``
    for a target-less module, ``contract`` to attach ``tl.buffer_alias_map``, and
    ``aiv`` to attach ``tl.aiv_storage_map``.
    """
    decls = [tirx.decl_buffer(shape, dtype, name=name, scope=scope) for name, shape, dtype, scope in buffers]
    body = (
        body
        if body is not None
        else tirx.SeqStmt([*[tirx.AllocBuffer(buffer) for buffer in decls], *[tirx.BufferStore(buffer, 0, [0]) for buffer in decls]])
    )
    if wrap is not None:
        body = wrap(body)
    by_name = {buffer.name: buffer for buffer in decls}
    if contract is not None:
        table = {by_name[src].data: [by_name[dst].data for dst in dsts] for src, dsts in contract.items()}
        body = tirx.AttrStmt(table, "tl.buffer_alias_map", 1, body)
    if aiv is not None:
        body = tirx.AttrStmt({by_name[name].data: sid for name, sid in aiv.items()}, "tl.aiv_storage_map", 1, body)
    func = tirx.PrimFunc([], body)
    if target is not None:
        func = func.with_attr("target", tvm.target.Target(target))
    return tvm.IRModule({"main": func})


def run(mod, **kwargs):
    return transform.MergeUBAllocations(**kwargs)(mod)


def assert_overflow(mod, name, required, capacity, **kwargs):
    with pytest.raises(tvm.error.InternalError) as error:
        run(mod, **kwargs)
    message = str(error.value)
    assert f"{name} memory overflow" in message
    assert f"{required} bytes required" in message
    assert f"capacity {capacity} bytes" in message


@pytest.mark.parametrize("scope,name,elements,capacity", OVERFLOW_SCOPES)
def test_every_scope_is_checked(scope, name, elements, capacity):
    buffers = [("x", (elements,), "int32", scope), ("y", (elements,), "int32", scope)]
    assert_overflow(module(buffers), name, 2 * elements * 4, capacity)


def test_no_target_and_non_ascend_skip_the_check():
    buffers = [("x", (32768,), "int32", "shared.dyn"), ("y", (32768,), "int32", "shared.dyn")]
    run(module(buffers, target=None))
    run(module(buffers, target="llvm"))


def test_reuse_decides_the_verdict():
    """The check reads the plan, not the sum: reuse makes the same shapes fit."""
    buffers = [("a", (32768,), "int32", "shared.dyn"), ("b", (32768,), "int32", "shared.dyn")]
    shared = module(buffers, contract={"a": ["b"], "b": ["a"]})
    run(shared)  # one 128 KB slot: fits
    assert_overflow(shared, "UB", 2 * 128 * 1024, UB_CAPACITY, disable_reuse=True)


def test_simt_vf_reserves_ub():
    """SIMT_VF parks its 32 KB thread context in the UB, lowering the limit."""
    buffers = [("ub0", (61440,), "int32", "shared.dyn")]  # 240 KB
    assert_overflow(module(buffers, wrap=lambda body: tirx.SBlock([], [], [], "SIMT_VF", body)), "UB", 240 * 1024, SIMT_VF_UB_CAPACITY)


def test_constant_floor_of_a_mixed_scope_is_fatal():
    """A symbolic allocation must not hide constants that alone exceed capacity."""
    n = tirx.Var("n", "int32")
    buffers = [("ub0", (32768,), "int32", "shared.dyn"), ("ub1", (32768,), "int32", "shared.dyn"), ("dyn", (n,), "int32", "shared.dyn")]
    assert_overflow(module(buffers), "UB", 2 * 128 * 1024, UB_CAPACITY)


def test_symbolic_floor_counts_a_reused_slot_once():
    """Aliased constant buffers share one arena slot beneath a dynamic allocation."""
    n = tirx.Var("n", "int32")
    buffers = [("ub0", (32768,), "int32", "shared.dyn"), ("ub1", (32768,), "int32", "shared.dyn"), ("dyn", (n,), "int32", "shared.dyn")]
    shared = module(buffers, contract={"ub0": ["ub1"], "ub1": ["ub0"]})
    run(shared)  # one 128 KB slot: the dynamic tail does not revive the sum
    assert_overflow(shared, "UB", 2 * 128 * 1024, UB_CAPACITY, disable_reuse=True)


def test_symbolic_floor_is_common_plus_the_larger_aiv():
    """Private AIV arenas overlap; the constant floor keeps the larger one."""
    n = tirx.Var("n", "int32")
    aiv = {"v0": 0, "v1": 1}
    buffers = [("v0", (51200,), "int32", "shared.dyn"), ("v1", (51200,), "int32", "shared.dyn"), ("dyn", (n,), "int32", "shared.dyn")]
    run(module(buffers, contract={"v0": [], "v1": [], "dyn": []}, aiv=aiv))  # max(200 KB, 200 KB) fits

    # 80 KB on both AIVs, 200 KB on AIV0, 40 KB on AIV1.
    buffers = [("common", (20480,), "int32", "shared.dyn"), ("v0", (51200,), "int32", "shared.dyn")]
    buffers += [("v1", (10240,), "int32", "shared.dyn"), ("dyn", (n,), "int32", "shared.dyn")]
    contract = {"common": [], "v0": [], "v1": [], "dyn": []}
    assert_overflow(module(buffers, contract=contract, aiv=aiv), "UB", (80 + 200) * 1024, UB_CAPACITY)


# --- end to end through the pipeline (source only, no NPU required) ---------


def _two_ub_copy_kernel(tile_elements, unlimit):
    import tilelang
    import tilelang.ascend.language as T

    @T.prim_func
    def kernel(a: T.Tensor((tile_elements,), "float32"), c: T.Tensor((tile_elements,), "float32")):
        with T.Kernel(1):
            if unlimit:
                T.annotate_unlimit_memory("shared")
            ub0 = T.alloc_shared((tile_elements,), "float32")
            ub1 = T.alloc_shared((tile_elements,), "float32")
            T.copy(a, ub0)
            T.copy(ub0, c)
            T.copy(a, ub1)
            T.copy(ub1, c)

    return tilelang.lower(kernel, target="ascend")


def test_lowering_rejects_ub_overflow():
    """2 x 128 KB of UB: rejected by the planner instead of silently overflowing."""
    with pytest.raises(tvm.error.InternalError, match="UB memory overflow"):
        _two_ub_copy_kernel(32768, unlimit=False)


def test_lowering_fits_when_ub_is_small_enough():
    assert _two_ub_copy_kernel(16384, unlimit=False).kernel_source


def test_unlimit_does_not_waive_the_physical_capacity():
    """unlimit only lifts the scheduling budget; the physical check still applies."""
    with pytest.raises(tvm.error.InternalError, match="UB memory overflow"):
        _two_ub_copy_kernel(32768, unlimit=True)
