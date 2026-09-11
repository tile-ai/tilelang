"""Backend-specific op hints are declared by the owning dialect.

Common tile ops (`T.copy`, `T.gemm`, ...) carry only target-neutral
parameters; each backend dialect shadows them with versions exposing that
backend's knobs as typed keywords (CUDA: TMA/cache hints, ROCm: k_pack, ...).
The hints are recorded as tile-op annotations, so a kernel written with one
dialect still compiles for targets that have no use for them.
"""

import importlib
import inspect

import pytest
import torch

import tilelang
import tilelang.cpu.language as Tcpu
import tilelang.language as T
import tilelang.rocm.language as Trocm
import tilelang.testing
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


def _keywords(func) -> set[str]:
    return {
        name
        for name, p in inspect.signature(func).parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY or p.default is not inspect.Parameter.empty
    }


def _tileop_annotations(func, opname: str) -> dict:
    found = {}

    def visit(node):
        if isinstance(node, tirx.Call) and str(getattr(node.op, "name", "")) == opname:
            found.update({str(k): v for k, v in node.annotations.items()})

    post_order_visit(func.body, visit)
    return found


def test_each_dialect_declares_its_own_op_hints():
    common = importlib.import_module("tilelang.language.common")
    cuda = importlib.import_module("tilelang.cuda.language")

    expected_cuda_extra = {
        "copy": {"disable_tma", "eviction_policy", "prefer_instruction"},
        "im2col": {"eviction_policy"},
        "gemm": {"mbar"},
        "gemm_sp": {"wg_wait"},
        "atomic_add": {"use_tma"},
        "Parallel": {"prefer_async"},
        "reduce_max": {"nan_propagate"},
        "reduce_min": {"nan_propagate"},
        "reduce_absmax": {"nan_propagate"},
        "unroll": {"unroll_factor"},
        "Unroll": {"unroll_factor"},
    }
    for name, extra in expected_cuda_extra.items():
        assert _keywords(getattr(cuda, name)) - _keywords(getattr(common, name)) == extra, name
        # The default facade is the CUDA dialect.
        assert getattr(T, name) is getattr(cuda, name), name
        # The CPU dialect keeps the neutral surface.
        assert getattr(Tcpu, name) is getattr(common, name), name

    assert _keywords(Trocm.gemm) - _keywords(common.gemm) == {"k_pack"}
    assert Trocm.copy is common.copy


def test_cuda_copy_hints_are_recorded_as_annotations():
    @T.prim_func
    def main(A: T.Tensor((128,), "float16"), B: T.Tensor((128,), "float16")):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((128,), "float16")
            T.copy(A, S, disable_tma=True, eviction_policy="evict_last", prefer_instruction="cp_async")
            T.copy(S, B)

    ann = _tileop_annotations(main, "tl.tileop.copy")
    assert int(ann["disable_tma"]) == 1
    assert int(ann["eviction_policy"]) == 2
    assert isinstance(ann["prefer_instruction"], tirx.StringImm) and ann["prefer_instruction"].value == "cp_async"


def test_cuda_atomic_add_use_tma_is_recorded_as_annotation():
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=64):
            S = T.alloc_shared((64,), "float32")
            T.copy(A, S)
            T.atomic_add(B, S, use_tma=True)

    ann = _tileop_annotations(main, "tl.tileop.atomicadd")
    assert int(ann["use_tma"]) == 1


def test_cuda_loop_hints_are_recorded_as_loop_annotations():
    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=64):
            for i in T.Parallel(64, prefer_async=True):
                B[i] = A[i]
            for j in T.unroll(4, unroll_factor=2):
                B[j] = A[j]

    annotations = {}

    def visit(node):
        if isinstance(node, tirx.For):
            annotations.update({str(k): v for k, v in node.annotations.items()})

    post_order_visit(main.body, visit)
    assert bool(annotations["parallel_prefer_async"])
    assert int(annotations["pragma_unroll_factor"]) == 2


def test_cuda_wgmma_gemm_records_wg_wait_annotation():
    """wgmma_gemm defers the warpgroup wait: wg_wait=-1 must ride the tile-op
    annotations now that the positional slot is gone."""

    @T.prim_func
    def main(A: T.Tensor((64, 64), "float16"), B: T.Tensor((64, 64), "float16"), C: T.Tensor((64, 64), "float32")):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((64, 64), "float16")
            b = T.alloc_shared((64, 64), "float16")
            c = T.alloc_fragment((64, 64), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.clear(c)
            T.wgmma_gemm(a, b, c)
            T.copy(c, C)

    ann = _tileop_annotations(main, "tl.tileop.wgmma_gemm")
    assert int(ann["wg_wait"]) == -1


def test_rocm_gemm_k_pack_traces_and_validates():
    @Trocm.prim_func
    def main(A: Trocm.Tensor((64, 64), "float16"), B: Trocm.Tensor((64, 64), "float16"), C: Trocm.Tensor((64, 64), "float32")):
        with Trocm.Kernel(1, threads=256):
            a = Trocm.alloc_shared((64, 64), "float16")
            b = Trocm.alloc_shared((64, 64), "float16")
            c = Trocm.alloc_fragment((64, 64), "float32")
            Trocm.copy(A, a)
            Trocm.copy(B, b)
            Trocm.clear(c)
            Trocm.gemm(a, b, c, k_pack=2)
            Trocm.copy(c, C)

    assert main is not None
    ann = _tileop_annotations(main, "tl.tileop.gemm")
    assert int(ann["k_pack"]) == 2

    with pytest.raises(ValueError, match="k_pack must be an int equal to 1 or 2"):

        @Trocm.prim_func
        def bad(A: Trocm.Tensor((64, 64), "float16")):
            with Trocm.Kernel(1):
                Trocm.gemm(A, A, A, k_pack=3)


def test_neutral_dialects_reject_backend_hints():
    with pytest.raises(TypeError, match="unexpected keyword argument 'disable_tma'"):

        @Tcpu.prim_func
        def cpu_copy(A: Tcpu.Tensor((16,), "float32"), B: Tcpu.Tensor((16,), "float32")):
            with Tcpu.Kernel(1):
                Tcpu.copy(A, B, disable_tma=True)

    with pytest.raises(TypeError, match="unexpected keyword argument 'k_pack'"):

        @T.prim_func
        def cuda_gemm(A: T.Tensor((64, 64), "float16")):
            with T.Kernel(1):
                T.gemm(A, A, A, k_pack=2)


def test_cuda_hints_do_not_block_cpu_compilation():
    """Hints are droppable: a kernel authored with the CUDA dialect still
    compiles for a target that has no use for them."""

    @T.prim_func
    def main(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            T.copy(A, B, disable_tma=True, eviction_policy="evict_first")

    mod = tilelang.compile(main, target="c", out_idx=[-1])
    x = torch.arange(64, dtype=torch.float32)
    torch.testing.assert_close(mod(x), x)


@tilelang.testing.requires_cuda
def test_cuda_hinted_copy_compiles_and_runs():
    @tilelang.jit(out_idx=[-1])
    def add(N):
        @T.prim_func
        def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
            with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
                s = T.alloc_shared((128,), "float32")
                T.copy(A[bx * 128], s, disable_tma=True, eviction_policy="evict_first")
                for i in T.Parallel(128):
                    s[i] = s[i] + 1.0
                T.copy(s, B[bx * 128])

        return main

    k = add(256)
    a = torch.randn(256, device="cuda")
    torch.testing.assert_close(k(a), a + 1)


if __name__ == "__main__":
    tilelang.testing.main()
