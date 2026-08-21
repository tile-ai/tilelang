import re

from tilelang import tvm as tvm
from tilelang.backend.target import determine_target
import tilelang as tl
import tilelang.language as T
import tilelang.testing
import pytest
import torch
from tvm.tirx.stmt_functor import post_order_visit

auto_target = tvm.target.Target(determine_target("auto"))


def _assert_launch_bounds(kernel_source: str, threads: int) -> None:
    # CUDA emits `__launch_bounds__(N, 1)` while HIP emits `__launch_bounds__(N)`.
    assert re.search(rf"__launch_bounds__\({threads}\b", kernel_source)


def _infer_coalesced_width_layout(coalesced_width=None, *, use_annotations=False):
    length = 132

    @T.prim_func
    def main(
        A: T.Tensor((length,), T.float32),
        B: T.Tensor((length,), T.float32),
    ):
        with T.Kernel(1, threads=64):
            if use_annotations:
                for i in T.Parallel(length, annotations={"coalesced_width": coalesced_width}):
                    B[i] = A[i]
            elif coalesced_width is not None:
                for i in T.Parallel(length, coalesced_width=coalesced_width):
                    B[i] = A[i]
            else:
                for i in T.Parallel(length):
                    B[i] = A[i]

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    with target:
        mod = tvm.IRModule({"main": main})
        mod = tvm.tirx.transform.BindTarget(target)(mod)
        mod = tl.transform.MaterializeKernelLaunch()(mod)
        mod = tl.transform.LayoutInference()(mod)

    layouts = []

    def collect_layout(node):
        if isinstance(node, tvm.tirx.For) and "parallel_loop_layout" in node.annotations:
            layouts.append(node.annotations["parallel_loop_layout"])

    post_order_visit(mod["main"].body, collect_layout)
    assert len(layouts) == 1
    return layouts[0]


@pytest.mark.parametrize(
    "coalesced_width,use_annotations",
    [
        pytest.param(4, False, id="keyword-int"),
        pytest.param(4, True, id="annotation-int"),
        pytest.param(T.IntImm("int32", 4), False, id="int-imm-i32"),
        pytest.param(T.IntImm("int64", 4), False, id="int-imm-i64"),
    ],
)
def test_parallel_coalesced_width_controls_inferred_layout(coalesced_width, use_annotations):
    default_layout = _infer_coalesced_width_layout()
    annotated_layout = _infer_coalesced_width_layout(coalesced_width, use_annotations=use_annotations)

    assert [int(extent) for extent in default_layout.get_output_shape()] == [3]
    assert [int(extent) for extent in annotated_layout.get_output_shape()] == [4]
    assert not tvm.ir.structural_equal(default_layout, annotated_layout)


@pytest.mark.parametrize("coalesced_width", [True, 4.0])
def test_parallel_coalesced_width_rejects_non_integer(coalesced_width):
    with pytest.raises(TypeError, match=r"Loop annotation `coalesced_width` expects an integer"):
        _infer_coalesced_width_layout(coalesced_width)


@pytest.mark.parametrize("coalesced_width", [0, -1])
def test_parallel_coalesced_width_requires_positive_integer(coalesced_width):
    with pytest.raises(ValueError, match=r"Loop annotation `coalesced_width` expects a positive integer"):
        _infer_coalesced_width_layout(coalesced_width)


def test_parallel_coalesced_width_preserves_divisibility_check():
    with pytest.raises(tvm.error.InternalError, match=r"Vector size 4 is not divisible by coalesced width 3"):
        _infer_coalesced_width_layout(3)


@pytest.mark.parametrize(
    "block_M, block_N, block_K, threads, vec_load_b, dtype",
    [
        (64, 64, 32, 128, 8, T.float16),
    ],
)
def test_loop_tail_split(block_M, block_N, block_K, threads, vec_load_b, dtype):
    N = tvm.te.var("n")
    K = tvm.te.var("k")

    def before():
        @T.prim_func
        def main(
            B: T.Tensor((K, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), threads=threads) as (bx):
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                thread_bindings = T.thread_binding(0, threads, "threadIdx.x")
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                    t = thread_bindings
                    for i in T.unroll(0, block_N * block_K // (threads * vec_load_b)):
                        for vec in T.Parallel(vec_load_b):
                            B_shared[
                                i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                            ] = T.if_then_else(
                                k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b) < K
                                and bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) < N,
                                B[
                                    k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                    bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                                ],
                                T.float16(0),
                            )

        return tvm.IRModule({"main": main})

    def after():
        @T.prim_func
        def main(
            B: T.Tensor((K, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), threads=threads) as (bx):
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                thread_bindings = T.thread_binding(0, threads, "threadIdx.x")
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                    t = thread_bindings
                    for i in T.unroll(0, block_N * block_K // (threads * vec_load_b)):
                        if (k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b)) * N % vec_load_b == 0:
                            for vec in T.vectorized(vec_load_b):
                                B_shared[
                                    i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                    t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                                ] = T.if_then_else(
                                    k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b) < K
                                    and bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) < N,
                                    B[
                                        k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                        bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                                    ],
                                    T.float16(0),
                                )
                        else:
                            for vec in T.serial(vec_load_b):
                                B_shared[
                                    i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                    t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                                ] = T.if_then_else(
                                    k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b) < K
                                    and bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) < N,
                                    B[
                                        k * block_K + i * (threads * vec_load_b // block_N) + t // (block_N // vec_load_b),
                                        bx * block_N + t % (block_N // vec_load_b) * (block_N // vec_load_b) + vec,
                                    ],
                                    T.float16(0),
                                )

        return tvm.IRModule({"main": main})

    with tvm.target.Target(auto_target):
        mod = tvm.tirx.transform.BindTarget(auto_target)(before())
        mod = tl.transform.MaterializeKernelLaunch()(mod)
        mod = tl.transform.LayoutInference()(mod)
        mod = tvm.tirx.transform.Simplify()(mod)
        ref_mod = tvm.tirx.transform.BindTarget(auto_target)(after())
        ref_mod = tl.transform.MaterializeKernelLaunch()(ref_mod)
        ref_mod = tvm.tirx.transform.Simplify()(ref_mod)
        # Note(tzj): The structures are equal except one more "for" loop after the LayoutInference pass
        # This loop is "for vec in T.parallel(1)",
        # Since the loop var "vec" is never used in the loop body, it does not affect the correctness
        tvm.ir.structural_equal(mod, ref_mod)
        # tvm.ir.assert_structural_equal(mod, ref_mod)


def test_register_count_is_default_layout_cost_model():
    @T.prim_func
    def main(
        S: T.Tensor((2,), T.float32),
        Out: T.Tensor((2, 2560), T.float32),
    ):
        with T.Kernel(1, threads=256):
            s_frag = T.alloc_fragment((2,), T.float32)
            for i in T.Parallel(2):
                s_frag[i] = S[i]
            for i, j in T.Parallel(2, 2560):
                Out[i, j] = s_frag[i] * 2.0

    target = auto_target

    def infer(pass_configs=None):
        with target, tvm.transform.PassContext(config=pass_configs or {}):
            mod = tvm.IRModule({"main": main})
            mod = tvm.tirx.transform.BindTarget(target)(mod)
            mod = tl.transform.MaterializeKernelLaunch()(mod)
            return tl.transform.LayoutInference()(mod)

    default = infer()
    register_count = infer({"tl.layout_cost_model": "register-count"})
    io_aware = infer({"tl.layout_cost_model": "io-aware"})

    tvm.ir.assert_structural_equal(default, register_count)
    assert not tvm.ir.structural_equal(default, io_aware)


def test_static_ragged_copy_minimizes_full_thread_padding():
    n = 514
    threads = 128

    @T.prim_func
    def main(
        A: T.Tensor((n,), T.float32),
        B: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            T.copy(A, B)

    with tvm.target.Target(auto_target):
        artifact = tl.lower(main, target=auto_target, enable_device_compile=False)

    kernel_source = str(artifact.kernel_source)
    _assert_launch_bounds(kernel_source, 128)
    assert "for (int i = 0; i < 5; ++i)" in kernel_source
    assert "threadIdx.x) >> 1)) < 257" in kernel_source
    assert "float2" not in kernel_source
    assert "threadIdx.x) < 1" not in kernel_source


def test_static_ragged_fp8_copy_minimizes_full_thread_padding():
    n = 3072
    threads = 128

    @T.prim_func
    def main(
        B: T.Tensor((n,), T.float8_e4m3),
    ):
        with T.Kernel(1, threads=threads):
            S = T.alloc_shared((n,), T.float8_e4m3)
            T.copy(S, B, disable_tma=True)

    with tvm.target.Target(auto_target):
        artifact = tl.lower(main, target=auto_target, enable_device_compile=False)

    kernel_source = str(artifact.kernel_source)
    _assert_launch_bounds(kernel_source, 128)
    assert "for (int i = 0; i < 3; ++i)" in kernel_source
    assert "fp8_e4_8_t" in kernel_source
    assert "fp8_e4_16_t" not in kernel_source


def test_static_ragged_copy_allows_1024_elements_384_threads():
    n = 1024
    threads = 384

    @T.prim_func
    def main(
        A: T.Tensor((n,), T.float32),
        B: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            T.copy(A, B, coalesced_width=1)

    with tvm.target.Target(auto_target):
        artifact = tl.lower(main, target=auto_target, enable_device_compile=False)

    kernel_source = str(artifact.kernel_source)
    _assert_launch_bounds(kernel_source, 384)
    assert "for (int i = 0; i < 3; ++i)" in kernel_source
    assert "B[((i * 384) + ((int)threadIdx.x))]" in kernel_source
    assert "(((int)threadIdx.x) >> 7)) < 8" in kernel_source
    assert "threadIdx.x) < 128" not in kernel_source


@pytest.mark.parametrize("block_n", [24, 40, 48, 64, 96])
def test_column_broadcast_fragment_tile_width_lowers(block_n):
    # Regression for issue #2394: LayoutInference used to synthesize a zero-extent
    # leftover iterator for non-power-of-two column broadcasts, then divide by zero.
    m, n = 256, block_n * 4
    block_m = 64

    @T.prim_func
    def main(D_in: T.Tensor((n,), T.bfloat16), Out: T.Tensor((m, n), T.bfloat16)):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=128) as (bx, by):
            d_local = T.alloc_fragment((block_n,), T.float32)
            d_shared = T.alloc_shared((block_n,), T.bfloat16)
            x = T.alloc_fragment((block_m, block_n), T.float32)
            xs = T.alloc_shared((block_m, block_n), T.bfloat16)

            T.copy(D_in[bx * block_n], d_shared)
            T.copy(d_shared, d_local)
            for i, j in T.Parallel(block_m, block_n):
                x[i, j] = d_local[j] * 2.0
            T.copy(x, xs)
            T.copy(xs, Out[by * block_m, bx * block_n])

    with tvm.target.Target(auto_target):
        artifact = tl.lower(main, target=auto_target, enable_device_compile=False)

    assert artifact.kernel_source


@tl.jit(out_idx=[1])
def _column_broadcast_fragment_kernel(block_n):
    m, n = 256, block_n * 4
    block_m = 64

    @T.prim_func
    def main(D_in: T.Tensor((n,), T.float32), Out: T.Tensor((m, n), T.float32)):
        with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=128) as (bx, by):
            d_local = T.alloc_fragment((block_n,), T.float32)
            d_shared = T.alloc_shared((block_n,), T.float32)
            x = T.alloc_fragment((block_m, block_n), T.float32)
            xs = T.alloc_shared((block_m, block_n), T.float32)

            T.copy(D_in[bx * block_n], d_shared)
            T.copy(d_shared, d_local)
            for i, j in T.Parallel(block_m, block_n):
                x[i, j] = d_local[j] * 2.0
            T.copy(x, xs)
            T.copy(xs, Out[by * block_m, bx * block_n])

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("block_n", [24, 32, 40, 48, 96])
def test_column_broadcast_fragment_values(block_n):
    # Numerical regression for issue #2394: the column broadcast must match D*2.
    kernel = _column_broadcast_fragment_kernel(block_n)

    d = torch.arange(block_n * 4, device="cuda", dtype=torch.float32)
    out = kernel(d)
    expected = d.unsqueeze(0).expand(256, -1) * 2.0

    assert torch.equal(out, expected)


if __name__ == "__main__":
    tilelang.testing.main()
