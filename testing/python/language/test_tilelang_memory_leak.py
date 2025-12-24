import tvm_ffi
import tilelang
import tilelang.language as T
import tilelang.testing
import torch
import weakref
import gc


def test_tilelang_globals_leak():
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    def get_dummy_kernel():
        @T.prim_func
        def dummy_kernel(
            a: T.Tensor[(1,), T.float32],
        ):
            with T.Kernel(1) as _:
                a[0] = 1

        return dummy_kernel

    a = torch.randn(1, 1024)
    a_weak = weakref.ref(a)
    _kernel = get_dummy_kernel()
    del a
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    a_upgrade = a_weak()
    assert a_upgrade is None, "A is not garbage collected"

    # use objgraph to debug
    # if a_upgrade is not None:
    #     objgraph.show_backrefs([a_upgrade], max_depth=5)


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
