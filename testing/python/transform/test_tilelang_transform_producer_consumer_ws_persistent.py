"""Repro/regression tests for ProducerConsumerWS on persistent pipelines.

The ``ProducerConsumerWarpSpecialized`` pass historically failed on any
persistent kernel whose top-level structure interposes a ``ForNode`` between
the block body and the inner ``T.Pipelined`` K-loop.

Two idiomatic TileLang patterns trigger this:

1. **``T.Persistent`` primitive** (see ``examples/gemm/example_gemm_persistent.py``)::

       for by, bx in T.Persistent([...], sm_num, block_id):
           for ko in T.Pipelined(num_k, num_stages=...):
               ...

   ``T.Persistent`` expands to ``SeqStmt(Bind, ..., For(kSerial, ...))``
   with the inner ``T.Pipelined`` sitting under an ``IfThenElse`` inside
   the ``For``'s body.

2. **Hand-written ``T.serial`` tile scheduler** (the ``use_persistent_primitive=False``
   fallback in the same example)::

       for tile_id in T.serial(total_tiles):
           for ko in T.Pipelined(num_k, num_stages=...):
               ...
"""

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tilelang.backend.target import determine_target


def persistent_matmul_primitive(
    M=512,
    N=512,
    K=256,
    block_M=64,
    block_N=64,
    block_K=32,
    num_stages=2,
    dtype="float16",
    accum_dtype="float32",
    threads=128,
    sm_num=4,
):

    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id,):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            for by, bx in T.Persistent(
                [T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
                sm_num,
                block_id,
            ):
                T.clear(C_local)
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def persistent_matmul_serial(
    M=256,
    N=256,
    K=256,
    block_M=64,
    block_N=64,
    block_K=32,
    num_stages=2,
    dtype="float16",
    threads=128,
):
    num_tiles_m = (M + block_M - 1) // block_M
    num_tiles_n = (N + block_N - 1) // block_N
    total_tiles = num_tiles_m * num_tiles_n
    num_k = (K + block_K - 1) // block_K

    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            for tile_id in T.serial(total_tiles):
                by = tile_id // num_tiles_n
                bx = tile_id % num_tiles_n
                T.clear(C_local)
                for ko in T.Pipelined(num_k, num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def persistent_matmul_serial_dynamic_bound(
    M=256,
    N=256,
    K=256,
    block_M=64,
    block_N=64,
    block_K=32,
    num_stages=2,
    dtype="float16",
    threads=128,
):
    """``T.serial`` persistent GEMM whose outer bound is symbolic.

    Guards against the fix accidentally depending on ``IntImm`` extents.
    """
    num_k = (K + block_K - 1) // block_K
    num_tiles_m_expr = T.ceildiv(M, block_M)
    num_tiles_n_expr = T.ceildiv(N, block_N)

    @T.prim_func
    def main(
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(1, threads=threads) as _:
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            for tile_id in T.serial(num_tiles_m_expr * num_tiles_n_expr):
                by = tile_id // num_tiles_n_expr
                bx = tile_id % num_tiles_n_expr
                T.clear(C_local)
                for ko in T.Pipelined(num_k, num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _apply_ws_pass(func):
    func = func.with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    target = determine_target({"kind": "cuda", "arch": "sm_90"}, return_object=True)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    return tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(mod)


def test_ws_pass_applies_under_t_persistent_primitive():
    """WS must descend through the ``For`` node synthesized by ``T.Persistent``."""
    func = persistent_matmul_primitive()
    mod = _apply_ws_pass(func)
    script = mod["main"].script()
    assert "tl_tiled_ws_applied" in script


def test_ws_pass_applies_under_persistent_serial():
    """Same coverage gap exercised via a hand-written ``T.serial`` scheduler."""
    func = persistent_matmul_serial()
    mod = _apply_ws_pass(func)
    script = mod["main"].script()
    assert "tl_tiled_ws_applied" in script


def test_ws_pass_applies_under_persistent_dynamic_bound():
    """The outer ``T.serial`` bound may be symbolic; the fix must not depend on IntImm."""
    func = persistent_matmul_serial_dynamic_bound()
    mod = _apply_ws_pass(func)
    script = mod["main"].script()
    assert "tl_tiled_ws_applied" in script


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_ws_end_to_end_t_persistent_matmul_correctness():
    import torch

    M, N, K = 256, 256, 256
    func = persistent_matmul_primitive(M, N, K, 64, 64, 32, num_stages=2, sm_num=4)
    kernel = tilelang.compile(func, target=determine_target(), out_idx=[2])
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = kernel(A, B)
    ref = A.float() @ B.float()
    torch.testing.assert_close(C.float(), ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_ws_end_to_end_persistent_serial_matmul_correctness():
    import torch

    M, N, K = 256, 256, 256
    func = persistent_matmul_serial(M, N, K, 64, 64, 32, num_stages=2)
    kernel = tilelang.compile(func, target=determine_target(), out_idx=[2])
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = kernel(A, B)
    ref = A.float() @ B.float()
    torch.testing.assert_close(C.float(), ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
