"""PTO codegen for explicit ``T.set_atomic`` / ``T.set_atomic_none``.

AscendC already lowers these ops. Annotation-based ``T.copy(..., atomic_op=...)``
is out of scope; this only pins the PTO ``VisitExpr_`` path used by kernels that
arm/disarm store-atomic state around an ordinary UB→GM copy (split-K flush).
"""

import pytest

import tilelang
from tilelang import language as T


@pytest.mark.pto
def test_explicit_set_atomic_pto_lowering():
    @T.prim_func
    def explicit_atomic(A: T.Buffer((128,), "int32"), B: T.Buffer((128,), "int32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((128,), "int32")
            T.copy(A, a_ub)
            T.set_atomic("add", "int32")
            T.copy(a_ub, B)
            T.set_atomic_none()

    source = tilelang.lower(explicit_atomic, target="pto").kernel_source
    assert 'kernel_kind="vector"' in source
    assert "pto.section" not in source
    dtype_pos = source.index("pto.set_atomic_s32()")
    op_pos = source.index("pto.set_atomic_add()", dtype_pos)
    copy_pos = source.index("pto.mte_ub_gm", op_pos)
    none_pos = source.index("pto.set_atomic_none()", copy_pos)
    assert dtype_pos < op_pos < copy_pos < none_pos
