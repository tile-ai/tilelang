"""BF16 arithmetic is legalized before the CPU codegens see it.

Host codegen legalizes BF16 storage to uint16 (``BF16StorageLegalize``), which
assumes BF16 arithmetic was legalized earlier (``BF16ComputeLegalize``). The CPU
pipeline did not run the latter, so any kernel whose lowering left a BF16
conversion on a local buffer failed in storage legalization. The lowering tests
reproduce that without a device for both CPU target kinds; the execution tests
need an LLVM-enabled build.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx


def _widen(size):
    @T.prim_func
    def main(A: T.Tensor((size,), "bfloat16"), B: T.Tensor((size,), "float32")):
        for i in T.Parallel(size):
            B[i] = A[i].astype("float32") * T.float32(2)

    return main


def _tile_copy(size):
    """The conversion comes from lowering a tile op, not from user arithmetic."""

    @T.prim_func
    def main(A: T.Tensor((size,), "bfloat16"), B: T.Tensor((size,), "float32")):
        with T.Kernel(1, threads=1):
            T.copy(A, B)

    return main


def _unlegalized(mod) -> list[str]:
    """Conversions that storage legalization cannot have produced on purpose.

    A cast to or from bfloat16 means the type survived; a cast between uint16
    and a float means a BF16 conversion was left in place and now reads or
    writes the raw bit pattern.
    """
    found = []

    def visit(node):
        if not isinstance(node, tirx.Cast):
            return
        source, target = str(node.value.dtype), str(node.dtype)
        bf16 = source.startswith("bfloat16") or target.startswith("bfloat16")
        bits = {source.split("x")[0], target.split("x")[0]} == {"uint16", "float32"}
        if bf16 or bits:
            found.append(f"{source} -> {target}")

    for _, func in mod.functions.items():
        tirx.stmt_functor.post_order_visit(func.body, visit)
    return found


@pytest.mark.parametrize("kind", ["c", "llvm"])
@pytest.mark.parametrize("factory", [_widen, _tile_copy])
def test_cpu_lowering_survives_storage_legalization(kind, factory):
    from tilelang.backend.module import create_backend_context
    from tilelang.engine.lower import lower_to_host_device_ir

    context = create_backend_context(kind, None, "auto")
    with tvm.transform.PassContext(), context.target:
        host_mod, _, _, _, target_host = lower_to_host_device_ir(factory(8), context)
        # What host codegen does next: without compute legalization this failed
        # with "Cannot find var remap" or a uint16 store mismatch.
        host_mod = tirx.transform.BindTarget(target_host)(host_mod)
        legalized = tirx.transform.BF16StorageLegalize()(host_mod)
    assert _unlegalized(legalized) == []


@tilelang.testing.requires_llvm
@pytest.mark.parametrize("factory,scale", [(_widen, 2), (_tile_copy, 1)])
def test_llvm_bf16_widening_is_numerically_correct(factory, scale):
    size = 64
    kernel = tilelang.compile(factory(size), target="llvm")
    a = torch.randn(size, dtype=torch.bfloat16)
    b = torch.empty(size, dtype=torch.float32)
    kernel(a, b)
    torch.testing.assert_close(b, a.to(torch.float32) * scale, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
