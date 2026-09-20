import tvm_ffi
import tilelang
import tilelang.language as T
import tilelang.testing
import torch
import weakref
import gc


def test_tilelang_globals_leak():
    @tilelang.jit(
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    def get_dummy_kernel():
        @T.prim_func
        def dummy_kernel(
            a: T.Tensor[(1,), T.float32],
        ):
            with T.Kernel(1) as _:
                # `ub = T.alloc_shared(...)` is an assignment that goes through
                # `__tb.bind` -> `get_parent_locals`. Without it the eager builder
                # never calls `get_parent_locals` (subscript stores and `with ... as`
                # targets do not), so this test cannot detect the frame leak.
                ub = T.alloc_shared((1,), "float32")
                T.copy(a, ub)
                T.copy(ub, a)

        return dummy_kernel

    def compile_with_probe() -> weakref.ReferenceType:
        # `a` lives in the first-compile call chain and is NOT deleted: any
        # retention (JIT machinery or a leaked `get_parent_locals` frame)
        # would keep it alive after this function returns.
        a = torch.randn(1, 1024)
        a_weak = weakref.ref(a)
        _kernel = get_dummy_kernel()
        return a_weak

    # temporarily disable gc: automatic cyclic GC would collect the
    # unreachable `get_parent_locals` frame cycle and mask the leak
    gc.disable()

    try:
        a_weak = compile_with_probe()

        # if anything still references `a`, a_weak() will return the object
        assert a_weak() is None, "A is not garbage collected"

        # use objgraph to debug
        # if a_weak() is not None:
        #     objgraph.show_backrefs([a_weak()], max_depth=5)
    finally:
        # re-enable gc whenever exception occurs
        gc.enable()


def test_error_no_cyclic_reference() -> None:
    # This test case ensures that when an error is raised from C++ side,
    # there is no cyclic reference that slows down the garbage collection.
    # Please see `_with_append_backtrace` in error.py

    # temporarily disable gc
    gc.disable()

    try:
        # We should create a class as a probe to detect gc activity
        # because weakref doesn't support list, dict or other trivial types
        class SampleObject: ...

        # trigger a C++ side KeyError by accessing a non-existent key
        def trigger_cpp_side_error() -> None:
            try:
                tmp_map = tvm_ffi.Map(dict())
                tmp_map["a"]
            except KeyError:
                pass

        def may_create_cyclic_reference() -> weakref.ReferenceType:
            obj = SampleObject()
            trigger_cpp_side_error()
            return weakref.ref(obj)

        wref = may_create_cyclic_reference()

        # if the object is not collected, wref() will return the object
        assert wref() is None, "Cyclic reference occurs inside error handling pipeline"

    finally:
        # re-enable gc whenever exception occurs
        gc.enable()


if __name__ == "__main__":
    tilelang.testing.main()
