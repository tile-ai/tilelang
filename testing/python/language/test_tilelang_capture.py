import tilelang.language as T
import tilelang.testing
import torch
import weakref
import gc


def test_tilelang_capture():
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


if __name__ == "__main__":
    tilelang.testing.main()
