import importlib.util
import os

import pytest


@pytest.mark.skipif(os.environ.get("TILELANG_LIGHT_IMPORT") == "1", reason="TMA execution requires the full frontend")
@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython", "nvrtc", "cutedsl"])
def test_tma_alignment_survives_static_and_dynamic_shared_storage(execution_backend):
    import torch
    import tilelang
    import tilelang.language as T
    from tilelang.layout import make_full_bank_swizzled_layout

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("TMA-capable CUDA hardware is required")
    if execution_backend == "cutedsl" and importlib.util.find_spec("cutlass") is None:
        pytest.skip("CUTLASS DSL is required")

    @T.prim_func
    def staged(
        source: T.Tensor((128, 256), "float16"),
        output: T.Tensor((128, 256), "float16"),
        observed: T.Tensor((2,), "int32"),
        marker: T.int32,
    ):
        with T.Kernel(2, threads=128) as block:
            probe = T.alloc_shared((1,), "int32", scope="shared")
            shared = T.alloc_shared((128, 128), "float16")
            T.annotate_layout({shared: make_full_bank_swizzled_layout(shared)})
            probe[0] = marker
            T.copy(source[:, block * 128 : (block + 1) * 128], shared)
            T.copy(shared, output[:, block * 128 : (block + 1) * 128])
            if T.get_thread_binding() == 0:
                observed[block] = probe[0]

    target = "cutedsl" if execution_backend == "cutedsl" else "cuda"
    kernel = tilelang.compile(staged, out_idx=[1, 2], target=target, execution_backend=execution_backend)
    source = torch.randn((128, 256), device="cuda", dtype=torch.float16)
    output, observed = kernel(source, 37)
    torch.testing.assert_close(output, source)
    torch.testing.assert_close(observed, torch.full_like(observed, 37))
