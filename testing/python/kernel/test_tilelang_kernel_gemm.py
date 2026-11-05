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
    b_dtype=None,
    gemm=T.gemm,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)
    a_layout = shared_layout if a_layout is None else a_layout
    b_layout = shared_layout if b_layout is None else b_layout
    b_dtype = in_dtype if b_dtype is None else b_dtype

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, b_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, b_dtype)
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
@pytest.mark.parametrize("in_dtype,wmma_dtype", [("bfloat16", "bf16"), ("float16", "f16")])
@pytest.mark.parametrize(
    "a_linear,b_linear", [(False, False), (True, True), (True, False), (False, True)], ids=["default", "linear", "linear-a", "linear-b"]
)
def test_gemm_wmma_shared_layout_loads(in_dtype, wmma_dtype, a_linear, b_linear):
    """Check physical LDS addresses and lane ownership for both operands."""
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
        in_dtype,
        in_dtype,
        "float32",
        0,
        32,
        a_layout=make_linear_layout if a_linear else None,
        b_layout=make_linear_layout if b_linear else None,
    )
    with target, tvm.transform.PassContext(opt_level=3):
        artifact = tilelang.lower(program, target=target)
    _check_wmma_shared_layout_loads(artifact, wmma_dtype, a_linear, b_linear)


def _check_wmma_shared_layout_loads(artifact, wmma_dtype, a_linear, b_linear):
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
    assert wmma_calls[0].args[0].value == f"f32_16x16x16_{wmma_dtype}_w32"
    assert f"__builtin_amdgcn_wmma_f32_16x16x16_{wmma_dtype}_w32_gfx12" in artifact.kernel_source
    thread_var = thread_vars[0]
    assert len(domains[thread_var]) == 32
    # Identify operands by their WMMA pointer arguments, not generated names.
    operand_data = {"A": wmma_calls[0].args[6], "B": wmma_calls[0].args[8]}
    loads = {name: [store for store in stores if store.buffer.data.same_as(data)] for name, data in operand_data.items()}
    # Loads may be split or vectorized differently, but every operand register
    # must still be populated exactly once from the correct shared address.
    for sites in loads.values():
        assert sites
        assert all(isinstance(store.value, tvm.tirx.BufferLoad) for store in sites)
    shared_vars = set(tvm.tirx.analysis.undefined_vars(loads["A"][0].value.indices[0])) & set(
        tvm.tirx.analysis.undefined_vars(loads["B"][0].value.indices[0])
    )
    reduction_vars = shared_vars - {thread_var}
    assert len(reduction_vars) == 1
    reduction_var = reduction_vars.pop()
    assert list(domains[reduction_var]) == [0, 1]

    def physical(row, col, linear):
        # The default 32-column 16-bit tile swaps column bits 3/4 by row bits
        # 1/2. Linear annotations must instead retain the row-major address.
        return row * 32 + (col if linear else col ^ ((row & 6) << 2))

    analyzer = tvm.arith.Analyzer()
    for name, sites in loads.items():
        visited = set()
        for store in sites:
            load = store.value
            index, destination = load.indices[0], store.indices[0]
            for expr in (index, destination):
                if isinstance(expr, tvm.tirx.Ramp):
                    assert int(expr.stride) == 1
                    assert expr.lanes == load.dtype.lanes
            base = index.base if isinstance(index, tvm.tirx.Ramp) else index
            local_base = destination.base if isinstance(destination, tvm.tirx.Ramp) else destination
            variables = tvm.tirx.analysis.undefined_vars(base + local_base)
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
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float16"])
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
        # Four wavefronts, multiple tiles, three K iterations and MNK tails.
        pytest.param((129, 135, 65), (128, 128, 32), 128, False, False, id="multi-wavefront-tail-default"),
        pytest.param((129, 135, 65), (128, 128, 32), 128, True, True, id="multi-wavefront-tail-linear"),
        pytest.param((129, 135, 65), (128, 128, 32), 128, True, False, id="multi-wavefront-tail-linear-a"),
        pytest.param((129, 135, 65), (128, 128, 32), 128, False, True, id="multi-wavefront-tail-linear-b"),
    ],
)
def test_gemm_wmma_shared_layout_correctness(shape, block, threads, a_linear, b_linear, in_dtype):
    import torch

    M, N, K = shape
    generator = torch.Generator().manual_seed(0)
    dtype = getattr(torch, in_dtype)
    # Small integers make the FP32 reduction exact, so this checks operand
    # mapping without relying on a loose GEMM numerical tolerance.
    a = torch.randint(-2, 3, (M, K), generator=generator).to(dtype)
    b = torch.randint(-2, 3, (K, N), generator=generator).to(dtype)
    _check_wmma_shared_layout_result(a, b, block, threads, a_linear, b_linear, in_dtype)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(
    "in_dtype,out_dtype",
    [("bfloat16", "bfloat16"), ("bfloat16", "float32"), ("float16", "float16"), ("float16", "float32")],
)
@pytest.mark.parametrize(
    "a_linear,b_linear", [(False, False), (True, True), (True, False), (False, True)], ids=["default", "linear", "linear-a", "linear-b"]
)
def test_gemm_wmma_shared_layout_fractional(in_dtype, out_dtype, a_linear, b_linear):
    import torch

    generator = torch.Generator().manual_seed(3729)
    # Both distributions are exactly representable in their input dtype.
    # Every FP32 partial sum is exact: numerator <= 65 * bound * 7 < 2**24.
    # Exercise tails, three K iterations, and same-type/FP32 output conversion.
    bound, scale = (513, 512) if in_dtype == "float16" else (127, 128)
    dtype = getattr(torch, in_dtype)
    a = (torch.randint(-bound, bound + 1, (17, 65), generator=generator) / scale).to(dtype)
    b = (torch.randint(-7, 8, (65, 35), generator=generator) / 16).to(dtype)
    if in_dtype == "float16":
        # Retain coverage of FP16 mantissas that BF16 cannot represent.
        assert torch.any(a.float() != a.bfloat16().float())
    _check_wmma_shared_layout_result(a, b, (16, 32, 32), 32, a_linear, b_linear, out_dtype)


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("extreme_operand", ["a", "b"])
@pytest.mark.parametrize("linear", [False, True], ids=["default", "linear"])
def test_gemm_wmma_shared_layout_normal_range(in_dtype, extreme_operand, linear):
    import torch

    dtype = getattr(torch, in_dtype)
    if in_dtype == "bfloat16":
        extremes = torch.tensor([2.0**32, -(2.0**32), 2.0**-32, -(2.0**-32)], dtype=dtype)
        # Neither end of this BF16 range survives an unintended FP16 conversion.
        assert torch.isinf(extremes.half()).any()
        assert (extremes.half() == 0).any()
    else:
        limits = torch.finfo(dtype)
        extremes = torch.tensor([limits.max, -limits.max, limits.tiny, -limits.tiny], dtype=dtype)
    coefficients = torch.tensor([2.0**-10, -(2.0**-10), 2.0**-8, -(2.0**-8)], dtype=dtype)
    a_values, b_values = (extremes, coefficients) if extreme_operand == "a" else (coefficients, extremes)
    a = a_values[torch.arange(16) % 4, None].expand(16, 32).contiguous()
    b = b_values[None, torch.arange(32) % 4].expand(32, 32).contiguous()
    for operand in (a, b):
        assert torch.isfinite(operand).all()
        assert (operand.abs() >= torch.finfo(dtype).tiny).all()
    # Each dot product repeats the same product 32 times. Power-of-two scaling
    # preserves the input significand, and every partial sum is exact in FP32.
    # All products/results are normal in FP32, avoiding denormal/overflow policy.
    _check_wmma_shared_layout_result(a, b, (16, 32, 32), 32, linear, linear, "float32")


def _check_wmma_shared_layout_result(a, b, block, threads, a_linear, b_linear, out_dtype):
    import torch

    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx1200":
        pytest.skip("Requires gfx1200 WMMA execution")
    M, K = a.shape
    N = b.shape[1]
    in_dtype = str(a.dtype).removeprefix("torch.")
    program = matmul(
        M,
        N,
        K,
        *block,
        False,
        False,
        in_dtype,
        out_dtype,
        "float32",
        0,
        threads,
        a_layout=make_linear_layout if a_linear else None,
        b_layout=make_linear_layout if b_linear else None,
    )
    kernel = tilelang.compile(program, target={"kind": "hip", "mcpu": "gfx1200"})
    dtype = getattr(torch, out_dtype)
    expected = (a.double() @ b.double()).to(dtype)
    guarded = torch.full((M * N + 32,), 123, dtype=dtype, device=device)
    output = guarded[16:-16].view(M, N)
    kernel(a.to(device), b.to(device), output)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert torch.all(guarded[:16] == 123)
    assert torch.all(guarded[-16:] == 123)


def _wmma_shared_layout_composition(in_dtype, a_linear, b_linear):
    @T.prim_func
    def main(
        A: T.Tensor((16, 32), in_dtype),
        B: T.Tensor((32, 32), in_dtype),
        Initial: T.Tensor((16, 32), "float32"),
        Bias: T.Tensor((32,), "float32"),
        C: T.Tensor((16, 32), "float32"),
        RowSum: T.Tensor((16,), "float32"),
    ):
        with T.Kernel(1, threads=32):
            A_shared = T.alloc_shared((16, 32), in_dtype)
            B_shared = T.alloc_shared((32, 32), in_dtype)
            A_default = T.alloc_shared((16, 32), in_dtype)
            B_default = T.alloc_shared((32, 32), in_dtype)
            C_local = T.alloc_fragment((16, 32), "float32")
            Row_local = T.alloc_fragment((16,), "float32")
            if a_linear:
                T.annotate_layout({A_shared: make_linear_layout(A_shared)})
            if b_linear:
                T.annotate_layout({B_shared: make_linear_layout(B_shared)})
            # Logical producers must honor each buffer's physical layout.
            for i, k in T.Parallel(16, 32):
                A_shared[i, k] = A[i, k] + 0.5
                A_default[i, k] = A[i, k] - 1
            for k, j in T.Parallel(32, 32):
                B_shared[k, j] = B[k, j] * 0.5
                B_default[k, j] = B[k, j] + 1
            T.copy(Initial, C_local)
            # Reuse both operands, mixing annotated and inferred layouts,
            # while all three GEMMs accumulate into the same fragment.
            T.gemm(A_shared, B_shared, C_local)
            T.gemm(A_shared, B_default, C_local)
            T.gemm(A_default, B_shared, C_local)
            for i, j in T.Parallel(16, 32):
                C_local[i, j] = T.max(C_local[i, j] + Bias[j], 0)
            T.reduce_sum(C_local, Row_local, dim=1)
            T.copy(C_local, C)
            T.copy(Row_local, RowSum)

    return main


@tilelang.testing.requires_rocm(support_required="compile-only")
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize(
    "a_linear,b_linear", [(False, False), (True, True), (True, False), (False, True)], ids=["default", "linear", "linear-a", "linear-b"]
)
def test_gemm_wmma_shared_layout_composition_lowering(in_dtype, a_linear, b_linear):
    """Protect layout compatibility even on runners without a gfx1200 GPU."""
    target = tvm.target.Target({"kind": "hip", "mcpu": "gfx1200"})
    program = _wmma_shared_layout_composition(in_dtype, a_linear, b_linear)
    with target, tvm.transform.PassContext(opt_level=3):
        assert tilelang.lower(program, target=target).device_mod is not None


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize(
    "a_linear,b_linear", [(False, False), (True, True), (True, False), (False, True)], ids=["default", "linear", "linear-a", "linear-b"]
)
def test_gemm_wmma_shared_layout_composition(in_dtype, a_linear, b_linear):
    import torch

    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx1200":
        pytest.skip("Requires gfx1200 WMMA execution")
    generator = torch.Generator().manual_seed(3729)
    dtype = getattr(torch, in_dtype)
    a = (torch.randint(-4, 5, (16, 32), generator=generator) / 2).to(dtype)
    b = (torch.randint(-4, 5, (32, 32), generator=generator) / 2).to(dtype)
    initial = torch.randint(-8, 9, (16, 32), generator=generator).float() / 8
    bias = torch.randint(-8, 9, (32,), generator=generator).float() / 4
    # Small dyadic fractions make the producers, GEMM accumulation and row
    # reduction exact in FP32. Compute the oracle independently on the CPU.
    a_produced, b_produced = a.double() + 0.5, b.double() * 0.5
    expected = initial.double() + a_produced @ b_produced + a_produced @ (b.double() + 1) + (a.double() - 1) @ b_produced
    expected = (expected + bias.double()).clamp_min(0).float()
    expected_rows = expected.double().sum(dim=1).float()

    program = _wmma_shared_layout_composition(in_dtype, a_linear, b_linear)
    kernel = tilelang.compile(program, target={"kind": "hip", "mcpu": "gfx1200"})
    guarded = torch.full((16 * 32 + 32,), 123, dtype=torch.float32, device=device)
    row_guarded = torch.full((16 + 32,), 123, dtype=torch.float32, device=device)
    output, rows = guarded[16:-16].view(16, 32), row_guarded[16:-16]
    output.fill_(float("nan"))
    rows.fill_(float("nan"))
    kernel(a.to(device), b.to(device), initial.to(device), bias.to(device), output, rows)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(rows.cpu(), expected_rows, rtol=0, atol=0)
    for storage in (guarded, row_guarded):
        assert torch.all(storage[:16] == 123)
        assert torch.all(storage[-16:] == 123)


@tilelang.testing.requires_rocm(support_required="compile-only")
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize(
    "layout_kind,error",
    [("overlapping", "forward map must be injective"), ("column-major", "Get different layout")],
)
def test_gemm_wmma_rejects_incompatible_shared_layout(layout_kind, error, in_dtype):
    def incompatible(buffer):
        if layout_kind == "overlapping":
            return Layout(buffer.shape, lambda i, j: j)
        # A bijective column-major mapping passes the shared-layout injectivity
        # check, but must still be rejected by the WMMA compatibility check.
        return Layout(buffer.shape, lambda i, j: j * buffer.shape[0] + i)

    target = tvm.target.Target({"kind": "hip", "mcpu": "gfx1200"})
    program = matmul(16, 32, 32, 16, 32, 32, False, False, in_dtype, in_dtype, "float32", 0, 32, incompatible)
    with target, tvm.transform.PassContext(opt_level=3), pytest.raises(Exception, match=error):
        tilelang.lower(program, target=target)


def _check_wmma_shared_accumulator_layout(in_dtype):
    @T.prim_func
    def main(C: T.Tensor((16, 32), "float32")):
        with T.Kernel(1, threads=32):
            A_shared = T.alloc_shared((16, 32), in_dtype)
            B_shared = T.alloc_shared((32, 32), in_dtype)
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
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float16"])
def test_gemm_wmma_shared_layout_rejects_accumulator(in_dtype):
    import subprocess
    import sys

    # Disable the emitter's Python scope assertion so this reaches the C++
    # per-buffer reuse check. Otherwise that assertion masks a missing A/B guard.
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])['_check_wmma_shared_accumulator_layout'](sys.argv[2])",
            __file__,
            in_dtype,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@tilelang.testing.requires_rocm(support_required="compile-only")
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize(
    "operand,storage",
    [
        pytest.param(None, "aligned", id="aligned"),
        *[
            pytest.param(operand, storage, id=f"{operand}-{storage}")
            for operand, storage in product(
                ("a", "b"),
                ("weak-alignment", "nonmultiple-alignment", "strides", "element-offset", "partial-region", "region-offset", "stage-axis"),
            )
        ],
    ],
)
def test_gemm_wmma_shared_layout_storage(operand, storage, in_dtype):
    """Require full, contiguous storage with alignment suitable for WMMA loads."""
    from tvm.script import tirx as I

    a_shape, b_shape = (16, 32), (32, 32)
    if storage == "stage-axis":
        a_shape = (1, 16, 32) if operand == "a" else a_shape
        b_shape = (1, 32, 32) if operand == "b" else b_shape
    elif storage in ("partial-region", "region-offset"):
        a_shape = (32, 32) if operand == "a" else a_shape
        b_shape = (64, 32) if operand == "b" else b_shape
    alignment = 8 if storage == "weak-alignment" else 24 if storage == "nonmultiple-alignment" else 16
    a_alignment = alignment if operand == "a" else 16
    b_alignment = alignment if operand == "b" else 16
    a_strides = (32, 1) if operand == "a" and storage == "strides" else None
    b_strides = (32, 1) if operand == "b" and storage == "strides" else None
    a_offset = 1 if operand == "a" and storage == "element-offset" else 0
    b_offset = 1 if operand == "b" and storage == "element-offset" else 0
    a_row = 16 if operand == "a" and storage == "region-offset" else 0
    b_row = 32 if operand == "b" and storage == "region-offset" else 0

    @T.prim_func
    def main():
        with T.Kernel(1, threads=32):
            A_shared = I.sblock_alloc_buffer(a_shape, in_dtype, scope="shared", align=a_alignment, strides=a_strides, elem_offset=a_offset)
            B_shared = I.sblock_alloc_buffer(b_shape, in_dtype, scope="shared", align=b_alignment, strides=b_strides, elem_offset=b_offset)
            C_local = T.alloc_fragment((16, 32), "float32")
            T.annotate_layout({A_shared: make_linear_layout(A_shared), B_shared: make_linear_layout(B_shared)})
            T.clear(A_shared)
            T.clear(B_shared)
            T.clear(C_local)
            if storage == "stage-axis" and operand == "a":
                T.gemm(A_shared[0, 0:16, 0:32], B_shared, C_local)
            elif storage == "stage-axis" and operand == "b":
                T.gemm(A_shared, B_shared[0, 0:32, 0:32], C_local)
            elif storage in ("partial-region", "region-offset") and operand == "a":
                T.gemm(A_shared[a_row : a_row + 16, 0:32], B_shared, C_local)
            elif storage in ("partial-region", "region-offset") and operand == "b":
                T.gemm(A_shared, B_shared[b_row : b_row + 32, 0:32], C_local)
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
@pytest.mark.parametrize("a_dtype,b_dtype", [("float16", "bfloat16"), ("bfloat16", "float16")])
@pytest.mark.parametrize("linear", [False, True], ids=["default", "linear"])
def test_gemm_wmma_shared_layout_rejects_mixed_dtypes(a_dtype, b_dtype, linear):
    target = tvm.target.Target({"kind": "hip", "mcpu": "gfx1200"})
    # Check the dtype diagnostic with and without annotations. Same-dtype
    # positive controls are covered by the address/load tests for both layouts.
    with target, tvm.transform.PassContext(opt_level=3), pytest.raises(Exception, match=f"Mixed-dtype GEMM.*got A={a_dtype}, B={b_dtype}"):
        program = matmul(
            16,
            32,
            32,
            16,
            32,
            32,
            False,
            False,
            a_dtype,
            "float32",
            "float32",
            0,
            32,
            shared_layout=make_linear_layout if linear else None,
            b_dtype=b_dtype,
        )
        tilelang.lower(program, target=target)


@tilelang.testing.requires_rocm(support_required="compile-only")
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("shared_operand", ["a", "b"], ids=["sr", "rs"])
def test_gemm_wmma_shared_layout_rejects_mixed_scopes(in_dtype, shared_operand):
    def program(linear):
        @T.prim_func
        def main(C: T.Tensor((16, 32), "float32")):
            with T.Kernel(1, threads=32):
                if shared_operand == "a":
                    A = T.alloc_shared((16, 32), in_dtype)
                    B = T.alloc_fragment((32, 32), in_dtype)
                else:
                    A = T.alloc_fragment((16, 32), in_dtype)
                    B = T.alloc_shared((32, 32), in_dtype)
                C_local = T.alloc_fragment((16, 32), "float32")
                if linear:
                    if shared_operand == "a":
                        T.annotate_layout({A: make_linear_layout(A)})
                    else:
                        T.annotate_layout({B: make_linear_layout(B)})
                T.clear(A)
                T.clear(B)
                T.clear(C_local)
                T.gemm(A, B, C_local)
                T.copy(C_local, C)

        return main

    target = tvm.target.Target({"kind": "hip", "mcpu": "gfx1200"})
    with target, tvm.transform.PassContext(opt_level=3):
        # SR/RS remain supported with their default inferred layouts.
        assert tilelang.lower(program(False), target=target).device_mod is not None
        with pytest.raises(Exception, match="Get different layout"):
            tilelang.lower(program(True), target=target)


@tilelang.testing.requires_rocm(support_required="compile-only")
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("linear", [False, True], ids=["default", "linear"])
def test_gemm_wmma_shared_layout_rejects_non_fp32_accum(in_dtype, linear):
    target = tvm.target.Target({"kind": "hip", "mcpu": "gfx1200"})
    # Protect the public FP32-accumulation contract through the existing emitter
    # diagnostic. FP32 positive controls are in the address/load test family.
    with target, tvm.transform.PassContext(opt_level=3), pytest.raises(Exception, match=f"Unsupported accum_dtype: {in_dtype}"):
        program = matmul(
            16,
            32,
            32,
            16,
            32,
            32,
            False,
            False,
            in_dtype,
            "float32",
            in_dtype,
            0,
            32,
            shared_layout=make_linear_layout if linear else None,
        )
        tilelang.lower(program, target=target)


@tilelang.testing.requires_rocm(support_required="compile-only")
@pytest.mark.parametrize("in_dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize(
    "arch,trans_A,trans_B,k_pack",
    [
        pytest.param("gfx942", False, False, 1, id="gfx942-False-False"),
        pytest.param("gfx1100", False, False, 1, id="gfx1100-False-False"),
        pytest.param("gfx1201", False, False, 1, id="gfx1201-False-False"),
        pytest.param("gfx1200", True, False, 1, id="gfx1200-True-False"),
        pytest.param("gfx1200", False, True, 1, id="gfx1200-False-True"),
        pytest.param("gfx1200", False, False, 2, id="k-pack-2"),
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
@pytest.mark.parametrize("in_dtype,wmma_dtype", [("bfloat16", "bf16"), ("float16", "f16")])
def test_gemm_wmma_linear_shared_layout_pipeline_fallback(in_dtype, wmma_dtype):
    """gfx1200 retains single-stage storage and valid WMMA operand loads."""
    target = tvm.target.Target({"kind": "hip", "mcpu": "gfx1200"})
    shared_sizes = []
    with target, tvm.transform.PassContext(opt_level=3):
        for num_stages in (0, 2):
            program = matmul(16, 32, 64, 16, 32, 32, False, False, in_dtype, in_dtype, "float32", num_stages, 32, make_linear_layout)
            artifact = tilelang.lower(program, target=target)
            _check_wmma_shared_layout_loads(artifact, wmma_dtype, True, True)
            shared_sizes.append([int(func.attrs["dyn_shared_memory_buf"]) for func in artifact.device_mod.functions.values()])
    assert shared_sizes[0] == shared_sizes[1]


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
