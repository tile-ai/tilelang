from itertools import product

from tilelang import tvm as tvm
import pytest
import tilelang.testing
import tilelang.language as T
from tilelang.layout import Layout, make_linear_layout


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
    shared_layout=None,
    *,
    a_layout=None,
    b_layout=None,
    gemm=T.gemm,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    a_layout = shared_layout if a_layout is None else a_layout
    b_layout = shared_layout if b_layout is None else b_layout

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            if a_layout is not None:
                T.annotate_layout({A_shared: a_layout(A_shared)})
            if b_layout is not None:
                T.annotate_layout({B_shared: b_layout(B_shared)})
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.testing.requires_rocm(support_required="compile-only")
@pytest.mark.parametrize(
    "a_linear,b_linear", [(False, False), (True, True), (True, False), (False, True)], ids=["default", "linear", "linear-a", "linear-b"]
)
def test_gemm_wmma_shared_layout_loads(a_linear, b_linear):
    """Check physical LDS addresses and lane ownership, including load widths."""
    target = tvm.target.Target({"kind": "hip", "mcpu": "gfx1200"})
    program = matmul(
        16,
        32,
        32,
        16,
        32,
        32,
        False,
        False,
        "bfloat16",
        "bfloat16",
        "float32",
        0,
        32,
        a_layout=make_linear_layout if a_linear else None,
        b_layout=make_linear_layout if b_linear else None,
    )
    with target, tvm.transform.PassContext(opt_level=3):
        artifact = tilelang.lower(program, target=target)

    stores = []
    wmma_calls = []
    domains = {}
    thread_vars = []

    def visit(node):
        if isinstance(node, tvm.tirx.For):
            domains[node.loop_var] = range(int(node.min), int(node.min + node.extent))
        elif isinstance(node, tvm.tirx.AttrStmt) and node.attr_key == "thread_extent":
            if node.node.thread_tag == "threadIdx.x":
                thread_vars.append(node.node.var)
                domains[node.node.var] = range(int(node.value))
        elif isinstance(node, tvm.tirx.BufferStore):
            stores.append(node)
        elif isinstance(node, tvm.tirx.Call) and node.op == tvm.ir.Op.get("tl.tvm_rdna_wmma"):
            wmma_calls.append(node)

    for func in artifact.device_mod.functions.values():
        tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    assert len(wmma_calls) == len(thread_vars) == 1
    assert wmma_calls[0].args[0].value == "f32_16x16x16_bf16_w32"
    thread_var = thread_vars[0]
    assert len(domains[thread_var]) == 32
    # Identify operands by their WMMA pointer arguments, not generated names.
    operand_data = {"A": wmma_calls[0].args[6], "B": wmma_calls[0].args[8]}
    loads = {name: [store for store in stores if store.buffer.data.same_as(data)] for name, data in operand_data.items()}
    # This minimal tile has one load site per operand. Never silently discard
    # additional stores if lowering changes.
    assert {name: len(sites) for name, sites in loads.items()} == {"A": 1, "B": 1}
    for sites in loads.values():
        assert isinstance(sites[0].value, tvm.tirx.BufferLoad)
    shared_vars = set(tvm.tirx.analysis.undefined_vars(loads["A"][0].value.indices[0])) & set(
        tvm.tirx.analysis.undefined_vars(loads["B"][0].value.indices[0])
    )
    reduction_vars = shared_vars - {thread_var}
    assert len(reduction_vars) == 1
    reduction_var = reduction_vars.pop()
    assert list(domains[reduction_var]) == [0, 1]

    def physical(row, col, linear):
        # The default 32-column BF16 tile swaps column bits 3/4 by row bits
        # 1/2. Linear annotations must instead retain the row-major address.
        return row * 32 + (col if linear else col ^ ((row & 6) << 2))

    analyzer = tvm.arith.Analyzer()
    for name, sites in loads.items():
        for store in sites:
            load = store.value
            assert load.dtype.lanes == (8 if name == "A" else 1)
            index, destination = load.indices[0], store.indices[0]
            for expr in (index, destination):
                if isinstance(expr, tvm.tirx.Ramp):
                    assert int(expr.stride) == 1
                    assert expr.lanes == load.dtype.lanes
            base = index.base if isinstance(index, tvm.tirx.Ramp) else index
            local_base = destination.base if isinstance(destination, tvm.tirx.Ramp) else destination
            variables = tvm.tirx.analysis.undefined_vars(base + local_base)
            visited = set()
            for values in product(*(domains[var] for var in variables)):
                bindings = dict(zip(variables, values))
                lane, ki = bindings[thread_var], bindings[reduction_var]
                actual = int(analyzer.simplify(tvm.tirx.stmt_functor.substitute(base, bindings)))
                local = int(analyzer.simplify(tvm.tirx.stmt_functor.substitute(local_base, bindings)))
                assert 0 <= actual <= (512 if name == "A" else 1024) - load.dtype.lanes
                assert actual % load.dtype.lanes == 0
                for element in range(load.dtype.lanes):
                    slot = local + element
                    assert 0 <= slot < (8 if name == "A" else 16)
                    k = ki * 16 + lane // 16 * 8 + slot % 8
                    row, col = (lane % 16, k) if name == "A" else (k, slot // 8 * 16 + lane % 16)
                    assert actual + element == physical(row, col, a_linear if name == "A" else b_linear)
                    # A matching access count alone can hide duplicate slots
                    # and leave other operand registers uninitialized.
                    position = (lane, ki, slot)
                    assert position not in visited
                    visited.add(position)
            assert visited == set(product(range(32), range(2), range(8 if name == "A" else 16)))


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(
    "shape,block,threads,a_linear,b_linear",
    [
        pytest.param((16, 32, 32), (16, 32, 32), 32, False, False, id="single-tile-default"),
        pytest.param((16, 32, 32), (16, 32, 32), 32, True, True, id="single-tile-linear"),
        pytest.param((17, 35, 33), (16, 32, 32), 32, False, False, id="mnk-tail-default"),
        pytest.param((17, 35, 33), (16, 32, 32), 32, True, True, id="mnk-tail-linear"),
        pytest.param((128, 128, 32), (128, 128, 32), 128, False, False, id="multiple-wavefronts-default"),
        pytest.param((128, 128, 32), (128, 128, 32), 128, True, True, id="multiple-wavefronts-linear"),
        pytest.param((16, 32, 32), (16, 32, 32), 32, True, False, id="single-tile-linear-a"),
        pytest.param((16, 32, 32), (16, 32, 32), 32, False, True, id="single-tile-linear-b"),
    ],
)
def test_gemm_wmma_shared_layout_correctness(shape, block, threads, a_linear, b_linear):
    import torch

    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx1200":
        pytest.skip("Requires gfx1200 WMMA execution")
    M, N, K = shape
    program = matmul(
        M,
        N,
        K,
        *block,
        False,
        False,
        "bfloat16",
        "bfloat16",
        "float32",
        0,
        threads,
        a_layout=make_linear_layout if a_linear else None,
        b_layout=make_linear_layout if b_linear else None,
    )
    kernel = tilelang.compile(program, target={"kind": "hip", "mcpu": "gfx1200"})
    generator = torch.Generator().manual_seed(0)
    # Small integers make the FP32 reduction exact, so this checks operand
    # mapping without relying on a loose GEMM numerical tolerance.
    a = torch.randint(-2, 3, (M, K), generator=generator).to(torch.bfloat16)
    b = torch.randint(-2, 3, (K, N), generator=generator).to(torch.bfloat16)
    expected = (a.float() @ b.float()).to(torch.bfloat16)
    guarded = torch.full((M * N + 32,), 123, dtype=torch.bfloat16, device=device)
    output = guarded[16:-16].view(M, N)
    kernel(a.to(device), b.to(device), output)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert torch.all(guarded[:16] == 123)
    assert torch.all(guarded[-16:] == 123)


@tilelang.testing.requires_rocm(support_required="compile-only")
@pytest.mark.parametrize(
    "layout_kind,error",
    [("overlapping", "forward map must be injective"), ("column-major", "Get different layout")],
)
def test_gemm_wmma_rejects_incompatible_shared_layout(layout_kind, error):
    def incompatible(buffer):
        if layout_kind == "overlapping":
            return Layout(buffer.shape, lambda i, j: j)
        # A bijective column-major mapping passes the shared-layout injectivity
        # check, but must still be rejected by the WMMA compatibility check.
        return Layout(buffer.shape, lambda i, j: j * buffer.shape[0] + i)

    target = tvm.target.Target({"kind": "hip", "mcpu": "gfx1200"})
    program = matmul(16, 32, 32, 16, 32, 32, False, False, "bfloat16", "bfloat16", "float32", 0, 32, incompatible)
    with target, tvm.transform.PassContext(opt_level=3), pytest.raises(Exception, match=error):
        tilelang.lower(program, target=target)


def _check_wmma_shared_accumulator_layout():
    @T.prim_func
    def main(C: T.Tensor((16, 32), "float32")):
        with T.Kernel(1, threads=32):
            A_shared = T.alloc_shared((16, 32), "bfloat16")
            B_shared = T.alloc_shared((32, 32), "bfloat16")
            C_shared = T.alloc_shared((16, 32), "float32")
            T.annotate_layout({C_shared: make_linear_layout(C_shared)})
            T.clear(A_shared)
            T.clear(B_shared)
            T.clear(C_shared)
            T.gemm(A_shared, B_shared, C_shared)
            T.copy(C_shared, C)

    target = tvm.target.Target({"kind": "hip", "mcpu": "gfx1200"})
    with target, tvm.transform.PassContext(opt_level=3), pytest.raises(Exception, match="Get different layout for C_shared"):
        tilelang.lower(main, target=target)


@tilelang.testing.requires_rocm(support_required="compile-only")
def test_gemm_wmma_shared_layout_rejects_accumulator():
    import subprocess
    import sys

    # Disable the emitter's Python scope assertion so this reaches the C++
    # per-buffer reuse check. Otherwise that assertion masks a missing A/B guard.
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])['_check_wmma_shared_accumulator_layout']()",
            __file__,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@tilelang.testing.requires_rocm(support_required="compile-only")
@pytest.mark.parametrize(
    "storage",
    [
        "aligned",
        "weak-alignment-a",
        "weak-alignment-b",
        "nonmultiple-alignment",
        "strides",
        "element-offset",
        "partial-region",
        "region-offset",
        "stage-axis",
    ],
)
def test_gemm_wmma_shared_layout_storage(storage):
    """Require full, contiguous storage with alignment suitable for WMMA loads."""
    from tvm.script import tirx as I

    a_shape = (1, 16, 32) if storage == "stage-axis" else (32, 32) if storage in ("partial-region", "region-offset") else (16, 32)
    a_alignment = 8 if storage == "weak-alignment-a" else 24 if storage == "nonmultiple-alignment" else 16
    b_alignment = 8 if storage == "weak-alignment-b" else 16
    a_strides = (32, 1) if storage == "strides" else None
    a_offset = 1 if storage == "element-offset" else 0

    @T.prim_func
    def main():
        with T.Kernel(1, threads=32):
            A_shared = I.sblock_alloc_buffer(
                a_shape, "bfloat16", scope="shared", align=a_alignment, strides=a_strides, elem_offset=a_offset
            )
            B_shared = I.sblock_alloc_buffer((32, 32), "bfloat16", scope="shared", align=b_alignment)
            C_local = T.alloc_fragment((16, 32), "float32")
            T.annotate_layout({A_shared: make_linear_layout(A_shared), B_shared: make_linear_layout(B_shared)})
            T.clear(A_shared)
            T.clear(B_shared)
            T.clear(C_local)
            if storage == "stage-axis":
                T.gemm(A_shared[0, 0:16, 0:32], B_shared, C_local)
            elif storage == "partial-region":
                T.gemm(A_shared[0:16, 0:32], B_shared, C_local)
            elif storage == "region-offset":
                T.gemm(A_shared[16:32, 0:32], B_shared, C_local)
            else:
                T.gemm(A_shared, B_shared, C_local)

    target = tvm.target.Target({"kind": "hip", "mcpu": "gfx1200"})
    with target, tvm.transform.PassContext(opt_level=3):
        if storage == "aligned":
            assert tilelang.lower(main, target=target).device_mod is not None
        else:
            with pytest.raises(Exception, match="Get different layout"):
                tilelang.lower(main, target=target)


@tilelang.testing.requires_rocm(support_required="compile-only")
@pytest.mark.parametrize(
    "arch,trans_A,trans_B,in_dtype,k_pack",
    [
        pytest.param("gfx942", False, False, "bfloat16", 1, id="gfx942-False-False"),
        pytest.param("gfx1100", False, False, "bfloat16", 1, id="gfx1100-False-False"),
        pytest.param("gfx1201", False, False, "bfloat16", 1, id="gfx1201-False-False"),
        pytest.param("gfx1200", True, False, "bfloat16", 1, id="gfx1200-True-False"),
        pytest.param("gfx1200", False, True, "bfloat16", 1, id="gfx1200-False-True"),
        pytest.param("gfx1200", False, False, "float16", 1, id="fp16-inputs"),
        pytest.param("gfx1200", False, False, "bfloat16", 2, id="k-pack-2"),
    ],
)
def test_gemm_shared_layout_unsupported_modes(arch, trans_A, trans_B, in_dtype, k_pack):
    from functools import partial
    from tilelang.rocm.language import gemm

    gemm = partial(gemm, k_pack=k_pack)
    target = tvm.target.Target({"kind": "hip", "mcpu": arch})
    with target, tvm.transform.PassContext(opt_level=3):
        default = matmul(32, 32, 32, 32, 32, 32, trans_A, trans_B, in_dtype, in_dtype, "float32", 0, 64, gemm=gemm)
        assert tilelang.lower(default, target=target).device_mod is not None
        linear = matmul(32, 32, 32, 32, 32, 32, trans_A, trans_B, in_dtype, in_dtype, "float32", 0, 64, make_linear_layout, gemm=gemm)
        with pytest.raises(Exception, match="Get different layout"):
            tilelang.lower(linear, target=target)


@tilelang.testing.requires_rocm(support_required="compile-only")
def test_gemm_wmma_linear_shared_layout_pipeline_fallback():
    """gfx1200 currently disables software pipelining, retaining the 2D tiles."""
    target = tvm.target.Target({"kind": "hip", "mcpu": "gfx1200"})
    sources = []
    with target, tvm.transform.PassContext(opt_level=3):
        for num_stages in (0, 2):
            program = matmul(16, 32, 64, 16, 32, 32, False, False, "bfloat16", "bfloat16", "float32", num_stages, 32, make_linear_layout)
            sources.append(tilelang.lower(program, target=target).kernel_source)
    assert sources[0] == sources[1]


def run_gemm(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=0,
    num_threads=128,
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program, out_idx=[2])
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if in_dtype == T.tfloat32:
            # Convert float32 to tfloat32 because tfloat32 mma cannot truncate
            # float32 automatically, -0x1000 meas
            A = (A.view(torch.int32) - 0x1000).view(torch.float32)
            B = (B.view(torch.int32) - 0x1000).view(torch.float32)
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_cuda
def test_gemm_f16f16f16_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


def test_gemm_f16f16f32_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.float16,
        T.float16,
        T.float32,
        128,
        128,
        32,
    )


def test_gemm_bf16bf16f32_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.bfloat16,
        T.bfloat16,
        T.float32,
        128,
        128,
        32,
    )


# tfloat32 is an NVIDIA-only dtype and has no CDNA lowering.
@tilelang.testing.requires_cuda
def test_gemm_f32f32f32_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.tfloat32,
        T.float32,
        T.float32,
        64,
        128,
        32,
    )


@tilelang.testing.requires_cuda
def test_gemm_f16f16f16_tn():
    run_gemm(
        512,
        1024,
        768,
        True,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_cuda
def test_gemm_f16f16f16_nt():
    run_gemm(
        512,
        1024,
        768,
        False,
        True,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_cuda_or_cdna
def test_gemm_i8i8i32_nt():
    run_gemm(512, 1024, 768, False, True, T.int8, T.int8, T.int32, 128, 128, 64)


@tilelang.testing.requires_cuda_or_cdna
def test_gemm_i8i8i32_tn():
    run_gemm(512, 1024, 768, True, False, T.int8, T.int8, T.int32, 128, 128, 64)


@tilelang.testing.requires_cuda
def test_gemm_f64f64f64_nt():
    run_gemm(512, 512, 512, False, True, T.float64, T.float64, T.float64, 64, 32, 16)


# tfloat32 is an NVIDIA-only dtype and has no CDNA lowering.
@tilelang.testing.requires_cuda
def test_gemm_f32f32f32_nt():
    run_gemm(
        512,
        1024,
        768,
        False,
        True,
        T.tfloat32,
        T.float32,
        T.float32,
        64,
        128,
        32,
    )


# TODO(Gong): Meets precision issue on ROCm, disable for now
@tilelang.testing.requires_cuda
def test_gemm_f32f32f32_tn():
    run_gemm(
        512,
        1024,
        768,
        True,
        False,
        T.tfloat32,
        T.float32,
        T.float32,
        64,
        128,
        32,
    )


@tilelang.testing.requires_cuda
def test_pad_aligned_f16f16f16_nn():
    run_gemm(
        512 - 8,
        1024 - 32,
        768 - 24,
        False,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        256,
        32,
        2,
    )


@tilelang.testing.requires_cuda
def test_pad_f16f16f16_nn():
    run_gemm(
        512 - 9,
        1024 - 7,
        768 - 5,
        False,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        256,
        32,
        2,
    )


def test_pad_f16f16f32_nn():
    run_gemm(
        512 + 19,
        1024 + 17,
        768 + 15,
        False,
        False,
        T.float16,
        T.float16,
        T.float32,
        128,
        64,
        32,
    )


def matmul_sr(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            B_local = T.alloc_fragment(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.copy(B_shared, B_local)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.copy(B_shared, B_local)
                T.gemm(A_shared, B_local, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_sr(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=1,
    num_threads=128,
):
    program = matmul_sr(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        A = A.to(torch.float)
        B = B.to(torch.float)
        C = torch.matmul(A, B)
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


# WGMMA only supports B in shared
@tilelang.testing.requires_cuda_compute_version_le(8, 9)
def test_gemm_f16f16f16_sr():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        True,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


def matmul_rs(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope="shared")
            A_local = T.alloc_fragment(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                    T.copy(A_shared, A_local)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(A_shared, A_local)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_local, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_rs(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=1,
    num_threads=128,
):
    program = matmul_rs(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program, out_idx=[2])
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


# Register source A operand GMMAs must have K-major A layout.
@tilelang.testing.requires_cuda_compute_version_le(8, 9)
def test_gemm_f16f16f16_rs():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


if __name__ == "__main__":
    tilelang.testing.main()
