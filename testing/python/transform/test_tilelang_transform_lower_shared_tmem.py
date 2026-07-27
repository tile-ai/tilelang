# ruff: noqa
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue


TARGET = tvm.target.Target({"kind": "cuda", "arch": "sm_100"})


def _apply(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(TARGET)(mod)
    mod = tl.transform.MaterializeKernelLaunch()(mod)
    mod = tl.cuda.transform.LowerSharedTmem()(mod)
    return mod


def _collect_calls(stmt, op_name: str):
    calls = []

    def visitor(node):
        if isinstance(node, tvm.tirx.Call) and hasattr(node, "op") and hasattr(node.op, "name") and node.op.name == op_name:
            calls.append(node)

    tvm.tirx.stmt_functor.post_order_visit(stmt, visitor)
    return calls


def _collect_buffer_loads(stmt, buffer_name: str):
    loads = []

    def visitor(node):
        if isinstance(node, tvm.tirx.BufferLoad) and node.buffer.name == buffer_name:
            loads.append(node)

    tvm.tirx.stmt_functor.post_order_visit(stmt, visitor)
    return loads


def _make_tmem_fragment_weighted_reduce_sum():
    @T.prim_func
    def func(
        A: T.Tensor((128, 64), T.bfloat16),
        B: T.Tensor((64, 64), T.bfloat16),
        Weights: T.Tensor((64,), T.float32),
        Y: T.Tensor((128,), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 64), T.bfloat16)
            B_shared = T.alloc_shared((64, 64), T.bfloat16)
            C_tmem = T.alloc_tmem((128, 64), T.float32)
            scores = T.alloc_fragment((128, 64), T.float32)
            logits = T.alloc_fragment((128,), T.float32)
            mbar = T.alloc_barrier(1)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.tcgen05_gemm(
                A_shared,
                B_shared,
                C_tmem,
                transpose_B=True,
                mbar=mbar,
                clear_accum=True,
            )
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(C_tmem, scores)
            for row, head in T.Parallel(128, 64):
                scores[row, head] = T.max(scores[row, head], T.float32(0)) * Weights[head]
            T.reduce_sum(scores, logits, dim=1)
            T.copy(logits, Y)

    return func


def test_explicit_deallocate_tmem_suppresses_auto_dealloc():
    """Explicit T.deallocate_tmem on fallthrough suppresses auto-dealloc."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 128], T.float32)
            T.deallocate_tmem(C_tmem)

    mod = _apply(func)
    body = mod["main"].body
    assert len(_collect_calls(body, "tl.ptx_init_tensor_memory")) == 1
    assert len(_collect_calls(body, "tl.ptx_deallocate_tensor_memory")) == 1
    assert len(_collect_calls(body, "tl.deallocate_tmem")) == 0

    dealloc_call = _collect_calls(body, "tl.ptx_deallocate_tensor_memory")[0]
    assert dealloc_call.args[1].value == 128


def test_explicit_deallocate_only_suppresses_matching_buffer():
    """Only the explicitly-deallocated buffer skips auto-dealloc; others keep it."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            A_tmem = T.alloc_tmem([128, 128], T.float32)
            B_tmem = T.alloc_tmem([128, 64], T.float32)
            T.deallocate_tmem(A_tmem)

    mod = _apply(func)
    body = mod["main"].body

    dealloc_calls = _collect_calls(body, "tl.ptx_deallocate_tensor_memory")
    # A_tmem: 1 explicit (auto suppressed); B_tmem: 1 auto = 2 total
    assert len(dealloc_calls) == 2

    dealloc_num_cols = sorted(call.args[1].value for call in dealloc_calls)
    assert dealloc_num_cols == [64, 128]


def test_dealloc_before_thread_return_keeps_auto_dealloc():
    """Dealloc on non-fallthrough path (before thread_return) does NOT suppress auto-dealloc."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 128], T.float32)
            tx = T.get_thread_binding()

            if tx < 32:
                T.deallocate_tmem(C_tmem)
                T.thread_return()

    mod = _apply(func)
    body = mod["main"].body

    dealloc_calls = _collect_calls(body, "tl.ptx_deallocate_tensor_memory")
    # 1 explicit (non-fallthrough) + 1 auto (block end) = 2
    assert len(dealloc_calls) == 2
    assert [call.args[1].value for call in dealloc_calls] == [128, 128]


def test_tmem_base_load_is_cached_after_allocation():
    """Repeated TMEM addresses reuse one register-cached base value."""

    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 128], T.float32)

    mod = _apply(func)
    body = mod["main"].body
    base_allocs = []

    def visitor(node):
        if isinstance(node, tvm.tirx.SBlock):
            base_allocs.extend(
                buffer for buffer in node.alloc_buffers if buffer.name == "C_tmem_base"
            )

    tvm.tirx.stmt_functor.post_order_visit(body, visitor)
    assert len(base_allocs) == 1
    assert len(_collect_buffer_loads(body, "C_tmem")) == 1


def test_nested_consumer_reuses_outer_tmem_base_buffer():
    """Nested consumer blocks keep the base allocated by the owning block."""

    @T.prim_func
    def func(
        X: T.Tensor((256, 256), T.float16),
        Y: T.Tensor((256, 256), T.float32),
    ):
        with T.Kernel(1, 1, threads=128):
            A_shared = T.alloc_shared((128, 128), T.float16)
            B_shared = T.alloc_shared((128, 128), T.float16)
            C_tmem = T.alloc_tmem((128, 128), T.float32)
            C_local = T.alloc_fragment((128, 128), T.float32)
            mbar = T.alloc_barrier(1)
            T.copy(X[0, 0], A_shared)
            T.copy(X[0, 0], B_shared)
            T.tcgen05_gemm(
                A_shared,
                B_shared,
                C_tmem,
                transpose_B=True,
                mbar=mbar,
                clear_accum=True,
            )
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, Y[0, 0])

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with TARGET:
        mod = CUDAPassPipelineBodyPrologue(mod, TARGET)
        mod = tl.cuda.transform.LowerSharedTmem()(mod)

    body = mod["main"].body
    base_allocs = []

    def visitor(node):
        if isinstance(node, tvm.tirx.SBlock):
            base_allocs.extend(
                buffer for buffer in node.alloc_buffers if buffer.name == "C_tmem_base"
            )

    tvm.tirx.stmt_functor.post_order_visit(body, visitor)
    base_loads = _collect_buffer_loads(body, "C_tmem_base")
    assert len(base_allocs) == 1
    assert base_loads
    assert all(load.buffer.data.same_as(base_allocs[0].data) for load in base_loads)


def test_tmem_fragment_weighted_reduce_sum_lowers():
    """A TMEM row can stay in registers through a weighted local reduction."""
    func = _make_tmem_fragment_weighted_reduce_sum()
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with TARGET:
        mod = CUDAPassPipelineBodyPrologue(mod, TARGET)
        mod = tl.cuda.transform.LowerSharedTmem()(mod)

    body = mod["main"].body
    assert _collect_calls(body, "tl.tcgen05_ld")
    assert not _collect_calls(body, "tl.tileop.reduce")


def test_tmem_codegen_includes_required_cuda_headers():
    with TARGET:
        artifact = tilelang.lower(_make_tmem_fragment_weighted_reduce_sum(), target=TARGET)
    source = artifact.kernel_source
    assert "#include <tl_templates/cuda/tcgen_05.h>" in source
    assert "#include <tl_templates/cuda/barrier.h>" in source
    assert "#include <tl_templates/cuda/copy_sm100.h>" in source


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
@tilelang.testing.requires_cuda_compute_version_lt(11)
def test_tmem_fragment_weighted_reduce_sum_correctness():
    """The native TMEM/register reduction executes without pipeline barriers."""
    import torch

    kernel = tilelang.compile(_make_tmem_fragment_weighted_reduce_sum(), target="cuda")
    a = torch.randn((128, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((64, 64), device="cuda", dtype=torch.bfloat16)
    weights = torch.randn((64,), device="cuda", dtype=torch.float32)
    out = torch.empty((128,), device="cuda", dtype=torch.float32)

    kernel(a, b, weights, out)
    ref = (torch.relu(a.float() @ b.float().T) * weights).sum(dim=1)
    tilelang.testing.torch_assert_close(out, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_explicit_deallocate_tmem_suppresses_auto_dealloc()
    test_explicit_deallocate_only_suppresses_matching_buffer()
    test_dealloc_before_thread_return_keeps_auto_dealloc()
    test_tmem_base_load_is_cached_after_allocation()
    test_nested_consumer_reuses_outer_tmem_base_buffer()
    test_tmem_fragment_weighted_reduce_sum_lowers()
    test_tmem_codegen_includes_required_cuda_headers()
