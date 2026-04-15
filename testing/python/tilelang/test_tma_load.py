import pytest
import torch

import tilelang as tl
import tilelang.language as T


N = 4096
BLOCK = 256


def _get_device_capability() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    return torch.cuda.get_device_capability()


def _extract_source(kernel) -> str:
    if hasattr(kernel, "get_source"):
        source = kernel.get_source()
        if isinstance(source, str) and source:
            return source

    module = getattr(kernel, "module", None)
    if module is not None and hasattr(module, "imported_modules"):
        imported = getattr(module, "imported_modules", [])
        if imported:
            source = imported[0].get_source()
            if isinstance(source, str) and source:
                return source

    runtime_mod = getattr(kernel, "rt_mod", None)
    if runtime_mod is not None and hasattr(runtime_mod, "imported_modules"):
        imported = getattr(runtime_mod, "imported_modules", [])
        if imported:
            source = imported[0].get_source()
            if isinstance(source, str) and source:
                return source

    raise RuntimeError("Unable to extract generated source from compiled kernel")


def _build_1d_tma_copy():
    @T.prim_func
    def main(A: T.Buffer((N,), "float16"), B: T.Buffer((N,), "float16")):
        with T.Kernel(T.ceildiv(N, BLOCK), threads=128) as bx:
            A_shared = T.alloc_shared((BLOCK,), "float16")
            T.copy(A[bx * BLOCK : (bx + 1) * BLOCK], A_shared)
            T.copy(A_shared, B[bx * BLOCK : (bx + 1) * BLOCK])

    return main


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(_get_device_capability()[0] < 9, reason="Hopper (sm90+) is required for TMA")
def test_tma_load_1d_compile_and_run_regression():
    program = _build_1d_tma_copy()
    kernel = tl.compile(program, out_idx=[1], target="cuda -arch=sm_90a")

    source = _extract_source(kernel)
    assert "cp.async.bulk.tensor" in source
    assert ".1d" in source

    a = torch.randn((N,), device="cuda", dtype=torch.float16)
    b = torch.empty_like(a)

    kernel(a, b)
    torch.testing.assert_close(b, a, atol=0, rtol=0)
