import pytest

import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.cuda.target import normalize_cutedsl_target


def _lower_cutedsl(program):
    if not tvm.runtime.enabled("cuda"):
        pytest.skip("TileLang CuTeDSL codegen requires TVM built with CUDA support.")

    build_cutedsl = tvm.ffi.get_global_func("target.build.tilelang_cutedsl_without_compile", allow_missing=True)
    if build_cutedsl is None:
        pytest.skip("TileLang CuTeDSL backend is not enabled in this build.")

    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": "sm_90"})
    assert target is not None

    with target:
        return lower(program.with_attr("global_symbol", "main"), target=target)


def test_cutedsl_codegen_warp_vote_and_bit_search():
    @T.prim_func
    def prog(A: T.Tensor((32,), "int64"), B: T.Tensor((12, 32), "int64")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B[0, tx] = T.cast(T.any_sync(tx == 0), "int64")
            B[1, tx] = T.cast(T.all_sync(tx), "int64")
            B[2, tx] = T.cast(T.ballot_sync(tx == 31), "int64")
            B[3, tx] = T.cast(T.ballot(1), "int64")
            B[4, tx] = T.cast(T.activemask(), "int64")
            B[5, tx] = T.cast(T.match_any_sync(T.cast(tx, "int64")), "int64")
            B[6, tx] = T.cast(T.match_all_sync(tx), "int64")
            B[7, tx] = T.cast(T.__ffs(tx), "int64")
            B[8, tx] = T.cast(T.__ffs(A[tx]), "int64")
            B[9, tx] = T.cast(T.__fns(T.uint32(0xF0F0F0F0), tx, -1), "int64")
            B[10, tx] = T.cast(T.syncthreads_count(tx < 4), "int64")
            B[11, tx] = T.cast(T.syncthreads_and(tx < 4) + T.syncthreads_or(tx < 4), "int64")

    source = _lower_cutedsl(prog).kernel_source

    for call in (
        "tl.__any_sync(",
        "tl.__all_sync(",
        "tl.ballot_sync(",
        "tl.ballot(1)",
        "tl.activemask()",
        "tl.__match_any_sync(",
        "tl.__match_all_sync(",
        "tl.__ffsll(",
        "tl.__fns(",
        "tl.__syncthreads_count(",
        "tl.__syncthreads_and(",
        "tl.__syncthreads_or(",
    ):
        assert call in source, f"{call} missing from:\n{source}"
    # The 32-bit operand uses __ffs, the 64-bit one __ffsll.
    assert source.count("tl.__ffs(") == 1, source
    assert source.count("tl.__ffsll(") == 1, source


def test_cutedsl_codegen_named_barrier_arrive():
    @T.prim_func
    def prog(B: T.Tensor((64,), "int32")):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            if tx < 32:
                T.named_barrier_arrive(3, 64)
            else:
                T.sync_threads(3, 64)
            B[tx] = tx

    source = _lower_cutedsl(prog).kernel_source

    assert "tl.named_barrier_arrive(3, 64)\n" in source, source
    assert "tl.sync_thread_partial(3, 64)" in source, source


def test_cutedsl_codegen_bool_bitwise_not_is_logical():
    @T.prim_func
    def prog(A: T.Tensor((1,), "bool"), B: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=1):
            B[0] = T.Cast("int32", ~A[0])
            B[1] = T.Cast("int32", ~T.bool(True))

    source = _lower_cutedsl(prog).kernel_source

    assert "~True" not in source, source
    assert "~A" not in source, source
    assert "(not " in source, source


def test_cutedsl_codegen_thread_allreduce_tree_masks():
    @T.prim_func
    def prog(A: T.Tensor((32,), "int16"), C: T.Tensor((1,), "int16")):
        with T.Kernel(1, threads=32):
            tk = T.get_thread_binding(0)
            v = T.alloc_local((1,), "int16")
            v[0] = A[tk]
            Cr = T.alloc_local((1,), "int16")
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, "int16")]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(T.uint32(1), v[0], True, Cr[0], tk, dtype="handle"))
            C[0] = Cr[0]

    source = _lower_cutedsl(prog).kernel_source

    assert "tl.activemask()" in source, source
    assert "tl.ballot_sync(" in source, source


if __name__ == "__main__":
    tilelang.testing.main()
