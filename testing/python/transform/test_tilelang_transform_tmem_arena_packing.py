# ruff: noqa
"""Logical TMEM buffers share physical tcgen05.alloc calls when that saves columns."""

import pytest

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


TARGET = tvm.target.Target({"kind": "cuda", "arch": "sm_100"})


def _apply(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(TARGET)(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    return tl.cuda.transform.LowerSharedTmem()(mod)


def _collect_calls(stmt, op_name):
    calls = []

    def visitor(node):
        if isinstance(node, tvm.tirx.Call) and getattr(node.op, "name", None) == op_name:
            calls.append(node)

    tvm.tirx.stmt_functor.post_order_visit(stmt, visitor)
    return calls


def _num_cols_allocated(body):
    """Columns passed to each tcgen05.alloc, in the order they are issued."""
    return [int(call.args[1]) for call in _collect_calls(body, "tl.ptx_init_tensor_memory")]


def _tmem_addresses(body):
    """``(address word name, column offset)`` for every ``T.evaluate(buf[i, j])``.

    A lowered TMEM address is ``cached_base + encoded_coordinate``, where the encoded
    coordinate carries the buffer's offset inside the allocation it shares.  With
    a ``[0, 0]`` coordinate the whole offset folds into one constant.
    """
    addresses = []

    def visitor(node):
        if not isinstance(node, tvm.tirx.Evaluate):
            return
        value = node.value
        if isinstance(value, tvm.tirx.Var):
            addresses.append((value.name, 0))
        elif isinstance(value, tvm.tirx.Add) and isinstance(value.a, tvm.tirx.Var):
            addresses.append((value.a.name, int(value.b)))

    tvm.tirx.stmt_functor.post_order_visit(body, visitor)
    return addresses


def _int_imms(expr):
    values = []
    tvm.tirx.stmt_functor.post_order_visit(expr, lambda node: values.append(int(node)) if isinstance(node, tvm.tirx.IntImm) else None)
    return values


def test_narrow_buffers_join_a_wide_allocation():
    """A 384-column accumulator plus three 4-column scale buffers need 396 columns.

    Allocated separately they would ask for 512 + 32 + 32 + 32 = 608 of the 512
    columns a CTA has.  Sharing one allocation fits, at the same offsets DeepGEMM
    uses by hand for this kernel.
    """

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 384], T.float32)
            SFQ_tmem = T.alloc_tmem([128, 4], "uint32")
            SFKV0_tmem = T.alloc_tmem([128, 4], "uint32")
            SFKV1_tmem = T.alloc_tmem([128, 4], "uint32")
            T.evaluate(C_tmem[0, 0])
            T.evaluate(SFQ_tmem[0, 0])
            T.evaluate(SFKV0_tmem[0, 0])
            T.evaluate(SFKV1_tmem[0, 0])

    body = _apply(func)["main"].body
    assert _num_cols_allocated(body) == [512]
    assert [int(call.args[1]) for call in _collect_calls(body, "tl.ptx_deallocate_tensor_memory")] == [512]
    assert _tmem_addresses(body) == [
        ("C_tmem_base", 0),
        ("C_tmem_base", 384),
        ("C_tmem_base", 388),
        ("C_tmem_base", 392),
    ]


def test_buffers_pack_only_when_it_saves_columns():
    """Block-scale factors share one allocation; the accumulator keeps its own.

    Putting all three together would round 136 columns up to 256, more than the
    128 + 32 = 160 this plan uses.
    """

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 128], T.float32)
            SFA_tmem = T.alloc_tmem([128, 4], "uint32")
            SFB_tmem = T.alloc_tmem([128, 4], "uint32")
            T.evaluate(C_tmem[0, 0])
            T.evaluate(SFA_tmem[0, 0])
            T.evaluate(SFB_tmem[0, 0])

    body = _apply(func)["main"].body
    assert _num_cols_allocated(body) == [128, 32]
    assert _tmem_addresses(body) == [("C_tmem_base", 0), ("SFA_tmem_base", 0), ("SFA_tmem_base", 4)]


def test_equal_cost_keeps_separate_allocations():
    """Packing that saves nothing is not done, so such a kernel lowers unchanged."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            S_tmem = T.alloc_tmem([128, 128], T.float32)
            P_tmem = T.alloc_tmem([128, 128], T.float16)
            O_tmem = T.alloc_tmem([128, 128], T.float32)
            T.evaluate(S_tmem[0, 0])
            T.evaluate(P_tmem[0, 0])
            T.evaluate(O_tmem[0, 0])

    body = _apply(func)["main"].body
    # 128 fp32 columns, 128 fp16 values in 64 b32 columns, 128 fp32 columns.
    assert _num_cols_allocated(body) == [128, 128, 64]
    assert _tmem_addresses(body) == [("S_tmem_base", 0), ("P_tmem_base", 0), ("O_tmem_base", 0)]


def test_allocations_are_issued_widest_first():
    """PTX forbids a tcgen05.alloc larger than one issued before it in the CTA."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            narrow_tmem = T.alloc_tmem([128, 32], T.float32)
            wide_tmem = T.alloc_tmem([128, 256], T.float32)
            T.evaluate(narrow_tmem[0, 0])
            T.evaluate(wide_tmem[0, 0])

    num_cols = _num_cols_allocated(_apply(func)["main"].body)
    assert num_cols == sorted(num_cols, reverse=True)


@pytest.mark.parametrize(
    "shapes",
    [
        # (columns of a 32-bit dtype) per buffer
        [384, 4, 4, 4],
        [128, 4, 4],
        [128, 128, 128],
        [32, 32, 32, 32, 32],
        [256, 8],
        [64, 64, 4],
    ],
)
def test_packing_never_allocates_more_than_separate_allocations(shapes):
    """The planning invariant: sharing an allocation is never the costlier choice."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            _ = [T.alloc_tmem([128, num_cols], "uint32") for num_cols in shapes]

    separate = sum(max(32, 1 << (num_cols - 1).bit_length()) for num_cols in shapes)
    packed = sum(_num_cols_allocated(_apply(func)["main"].body))
    assert packed <= separate


def test_explicit_deallocate_keeps_its_own_allocation():
    """A hand-managed lifetime cannot share an allocation it would release early."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 384], T.float32)
            SF_tmem = T.alloc_tmem([128, 4], "uint32")
            T.evaluate(C_tmem[0, 0])
            T.evaluate(SF_tmem[0, 0])
            T.deallocate_tmem(SF_tmem)

    body = _apply(func)["main"].body
    assert _num_cols_allocated(body) == [512, 32]
    assert _tmem_addresses(body) == [("C_tmem_base", 0), ("SF_tmem_base", 0)]


def test_dynamic_coordinate_keeps_the_arena_offset():
    """A software-pipeline stage index moves the address; the offset still applies."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 384], T.float32)
            SF_tmem = T.alloc_tmem([128, 8], "uint32")
            stage = T.alloc_var(T.int32, init=0)
            T.evaluate(SF_tmem[0, stage * 4])

    body = _apply(func)["main"].body
    assert _num_cols_allocated(body) == [512]
    addresses = _collect_calls(body, "tl.ptx_init_tensor_memory")
    assert len(addresses) == 1

    dynamic_address = []
    tvm.tirx.stmt_functor.post_order_visit(
        body,
        lambda node: (
            dynamic_address.append(node.value)
            if isinstance(node, tvm.tirx.Evaluate) and not isinstance(node.value, tvm.tirx.Call)
            else None
        ),
    )
    assert len(dynamic_address) == 1
    assert 384 in _int_imms(dynamic_address[0])


def test_block_over_the_column_budget_is_reported():
    """Two 384-column buffers cannot share or fit, which is worth saying out loud."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            first_tmem = T.alloc_tmem([128, 384], T.float32)
            second_tmem = T.alloc_tmem([128, 384], T.float32)
            T.evaluate(first_tmem[0, 0])
            T.evaluate(second_tmem[0, 0])

    with pytest.raises(
        tvm.error.InternalError,
        match=r"allocates 1024 TMEM columns across 2 tcgen05.alloc calls, but a CTA only has 512",
    ):
        _apply(func)


if __name__ == "__main__":
    pytest.main([__file__])
