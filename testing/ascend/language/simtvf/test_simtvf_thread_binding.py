"""Test T.get_thread_binding() and related APIs inside SimtVF.

Standalone script (not pytest).  Exercises the Python-level thread-binding
query helpers that were added to the SimtVF frame.
"""

from __future__ import annotations

import tilelang.language as T
from tilelang.language import kernel as K
from tilelang import tvm
from tilelang.backend.target import determine_target
from tvm.tirx.stmt_functor import post_order_visit

from testing.ascend.language.simtvf._inspect_utils import collect_simtvf_blocks


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_target():
    return determine_target("ascend", return_object=True)


def _run_test(name: str, fn):
    try:
        fn()
        print(f"  [ok] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        raise


# ---------------------------------------------------------------------------
# test cases
# ---------------------------------------------------------------------------


def test_get_thread_binding_inside_simtvf():
    """K.get_thread_binding(0) inside SimtVF(threads=128) returns a Var
    with name == 'simtvf_tx'."""
    results = {}
    target = _make_target()
    with target:

        @T.prim_func
        def fn(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
            with T.Kernel(1) as _, T.SimtVF(threads=128):
                results["tx"] = K.get_thread_binding(0)
                for i in T.Parallel(16):
                    B[i] = A[i] + T.float32(1)

    assert isinstance(results["tx"], tvm.tirx.Var), f"Expected tvm.tirx.Var, got {type(results['tx'])}"
    assert results["tx"].name == "simtvf_tx", f"Expected name 'simtvf_tx', got '{results['tx'].name}'"


def test_get_thread_bindings_inside_simtvf():
    """K.get_thread_bindings() returns 3 Vars named
    ['simtvf_tx', 'simtvf_ty', 'simtvf_tz']."""
    results = {}
    target = _make_target()
    with target:

        @T.prim_func
        def fn(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
            with T.Kernel(1) as _, T.SimtVF(threads=128):
                results["bindings"] = K.get_thread_bindings()
                for i in T.Parallel(16):
                    B[i] = A[i] + T.float32(1)

    bindings = results["bindings"]
    assert len(bindings) == 3, f"Expected 3 bindings, got {len(bindings)}"
    expected_names = ["simtvf_tx", "simtvf_ty", "simtvf_tz"]
    actual_names = [v.name for v in bindings]
    assert actual_names == expected_names, f"Expected names {expected_names}, got {actual_names}"


def test_get_thread_extent_inside_simtvf():
    """K.get_thread_extent(dim) returns 128/1/1 for SimtVF(threads=128)."""
    results = {}
    target = _make_target()
    with target:

        @T.prim_func
        def fn(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
            with T.Kernel(1) as _, T.SimtVF(threads=128):
                results["ext0"] = K.get_thread_extent(0)
                results["ext1"] = K.get_thread_extent(1)
                results["ext2"] = K.get_thread_extent(2)
                for i in T.Parallel(16):
                    B[i] = A[i] + T.float32(1)

    assert results["ext0"] == 128, f"dim0: expected 128, got {results['ext0']}"
    assert results["ext1"] == 1, f"dim1: expected 1, got {results['ext1']}"
    assert results["ext2"] == 1, f"dim2: expected 1, got {results['ext2']}"


def test_get_thread_extents_inside_simtvf():
    """K.get_thread_extents() returns [128, 1, 1]."""
    results = {}
    target = _make_target()
    with target:

        @T.prim_func
        def fn(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
            with T.Kernel(1) as _, T.SimtVF(threads=128):
                results["extents"] = K.get_thread_extents()
                for i in T.Parallel(16):
                    B[i] = A[i] + T.float32(1)

    assert results["extents"] == [128, 1, 1], f"Expected [128, 1, 1], got {results['extents']}"


def test_sequential_simtvf_different_threads():
    """Two sequential SimtVF regions have independent thread vars and extents."""
    results = {}
    target = _make_target()
    with target:

        @T.prim_func
        def fn(
            A: T.Buffer((256,), "float32"),
            B: T.Buffer((256,), "float32"),
            C: T.Buffer((256,), "float32"),
        ):
            with T.Kernel(1) as _:
                with T.SimtVF(threads=128):
                    results["tx1"] = K.get_thread_binding(0)
                    results["ext1"] = K.get_thread_extent(0)
                    for i in T.Parallel(256):
                        B[i] = A[i] + T.float32(1)
                with T.SimtVF(threads=256):
                    results["tx2"] = K.get_thread_binding(0)
                    results["ext2"] = K.get_thread_extent(0)
                    for i in T.Parallel(256):
                        C[i] = A[i] + T.float32(2)

    assert results["ext1"] == 128, f"First SimtVF: expected extent 128, got {results['ext1']}"
    assert results["ext2"] == 256, f"Second SimtVF: expected extent 256, got {results['ext2']}"
    assert not results["tx1"].same_as(results["tx2"]), "Thread vars from different SimtVF regions must be distinct objects"


def test_variable_identity():
    """The Var from K.get_thread_binding(0) is the SAME object that appears
    in the generated TIR's SIMT_VF block AttrStmt."""
    results = {}
    target = _make_target()
    with target:

        @T.prim_func
        def fn(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
            with T.Kernel(1) as _, T.SimtVF(threads=128):
                results["tx"] = K.get_thread_binding(0)
                for i in T.Parallel(16):
                    B[i] = A[i] + T.float32(1)

    # Walk the TIR to find the AttrStmt for threadIdx.x inside the SIMT_VF block
    simtvf_blocks = collect_simtvf_blocks(fn)
    assert len(simtvf_blocks) >= 1, "Expected at least one SIMT_VF block"

    tir_thread_vars = []

    def _find_thread_vars(node):
        if (
            isinstance(node, tvm.tirx.AttrStmt)
            and node.attr_key == "thread_extent"
            and isinstance(node.node, tvm.tirx.IterVar)
            and node.node.thread_tag == "threadIdx.x"
        ):
            tir_thread_vars.append(node.node.var)

    for block in simtvf_blocks:
        post_order_visit(block.body, _find_thread_vars)

    assert len(tir_thread_vars) >= 1, "Expected at least one threadIdx.x AttrStmt in SIMT_VF"

    python_var = results["tx"]
    found_match = any(python_var.same_as(tv) for tv in tir_thread_vars)
    assert found_match, f"Python-side Var (id={id(python_var)}, name={python_var.name}) does not match any TIR threadIdx.x Var"


def test_multidim_simtvf():
    """SimtVF(threads=[64, 2, 1]) produces extents [64, 2, 1] and
    binding names ['simtvf_tx', 'simtvf_ty', 'simtvf_tz']."""
    results = {}
    target = _make_target()
    with target:

        @T.prim_func
        def fn(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")):
            with T.Kernel(1) as _, T.SimtVF(threads=[64, 2, 1]):
                results["extents"] = K.get_thread_extents()
                results["bindings"] = K.get_thread_bindings()
                for i in T.Parallel(16):
                    B[i] = A[i] + T.float32(1)

    assert results["extents"] == [64, 2, 1], f"Expected extents [64, 2, 1], got {results['extents']}"
    expected_names = ["simtvf_tx", "simtvf_ty", "simtvf_tz"]
    actual_names = [v.name for v in results["bindings"]]
    assert actual_names == expected_names, f"Expected names {expected_names}, got {actual_names}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    tests = [
        ("test_get_thread_binding_inside_simtvf", test_get_thread_binding_inside_simtvf),
        ("test_get_thread_bindings_inside_simtvf", test_get_thread_bindings_inside_simtvf),
        ("test_get_thread_extent_inside_simtvf", test_get_thread_extent_inside_simtvf),
        ("test_get_thread_extents_inside_simtvf", test_get_thread_extents_inside_simtvf),
        ("test_sequential_simtvf_different_threads", test_sequential_simtvf_different_threads),
        ("test_variable_identity", test_variable_identity),
        ("test_multidim_simtvf", test_multidim_simtvf),
    ]

    print("Running SimtVF thread-binding API tests ...\n")
    for name, fn in tests:
        _run_test(name, fn)

    print(f"\n[ok] All {len(tests)} SimtVF get_thread_binding tests passed.")


if __name__ == "__main__":
    main()
