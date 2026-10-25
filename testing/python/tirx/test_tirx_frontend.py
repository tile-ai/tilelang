import sys

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.language as T
from tilelang._compat.stmt_functor import post_order_visit


def _nodes(function, kind):
    nodes = []
    post_order_visit(function.body, lambda node: nodes.append(node) if isinstance(node, kind) else None)
    return nodes


def test_normal_import_owns_no_legacy_tvm():
    assert "tvm" not in sys.modules
    assert tilelang.tvm.tirx.PrimFunc is tirx.Function
    assert not hasattr(ir.Var("index", "int32"), "__iter__")


@pytest.mark.parametrize("initializer", [None, 0, 7])
def test_alloc_var_preserves_optional_initialization(initializer):
    @T.prim_func
    def scalar(output: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=32):
            value = T.alloc_var("float32", init=initializer)
            output[0] = value

    initialized = [store for store in _nodes(scalar, ir.TensorStore) if store.dest.name == "value"]
    assert len(initialized) == 1
    assert float(initialized[0].value) == (0 if initializer is None else initializer)


def test_intrinsic_void_result_is_a_native_type():
    operation = T.call_intrin("void", "tl.sync_warp")
    assert operation.ty == ir.PrimType("void")


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "int32"])
@pytest.mark.parametrize("lanes", [1, 4])
def test_cuda_default_local_var_is_zero(dtype, lanes):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required for local-variable validation")
    vector_dtype = dtype if lanes == 1 else f"{dtype}x{lanes}"

    @T.prim_func
    def initialize(output: T.Tensor((32 * lanes,), dtype)):
        with T.Kernel(1, threads=32):
            thread = T.get_thread_binding()
            value = T.alloc_var(vector_dtype)
            if lanes == 1:
                output[thread] = value
            else:
                output[T.Ramp(thread * lanes, 1, lanes)] = value

    output = torch.full((32 * lanes,), 127, dtype=getattr(torch, dtype), device="cuda")
    kernel = tilelang.compile(initialize, target="cuda")
    kernel(output)
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)


@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython", "nvrtc"])
def test_implicit_shape_assumption_is_checked_before_launch(execution_backend):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required")
    extent = T.dynamic("extent")

    @T.prim_func
    def checked(output: T.Tensor((extent,), "int32")):
        with T.Kernel(1, threads=32):
            T.assume(extent % 4 == 0)
            thread = T.get_thread_binding()
            if thread < extent:
                output[thread] = 1

    kernel = tilelang.compile(checked, target="cuda", execution_backend=execution_backend)
    output = torch.zeros(4, device="cuda", dtype=torch.int32)
    kernel(output)
    torch.testing.assert_close(output, torch.ones_like(output))
    with pytest.raises(RuntimeError, match="Assume:"):
        kernel(torch.zeros(2, device="cuda", dtype=torch.int32))


@pytest.mark.parametrize("participants", [(), (3,), (3, 64)])
def test_explicit_thread_sync_uses_native_storage_sync(participants):
    operation = T.sync_threads(*participants)
    assert operation.op.name == "tirx.gpu_storage_sync"
    assert operation.ty == ir.PrimType("void")
    assert operation.args[0].value == "shared"
    assert [int(value) for value in operation.args[1:]] == list(participants)


def test_thread_return_is_native_and_preserves_other_threads():
    import torch

    @T.prim_func
    def early_exit(output: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            thread = T.get_thread_binding()
            if thread >= 16:
                T.thread_return()
            output[thread] = thread + 1

    returns = _nodes(early_exit, ir.Return)
    assert len(returns) == 1
    assert isinstance(returns[0].value, ir.Tuple)
    assert len(returns[0].value.fields) == 0
    assert tirx.analysis.check_well_formed(early_exit)
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required for thread-return execution")
    kernel = tilelang.compile(early_exit, target="cuda")
    output = torch.zeros(32, device="cuda", dtype=torch.int32)
    kernel(output)
    expected = torch.zeros_like(output)
    expected[:16] = torch.arange(1, 17, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(output, expected)


def test_prim_func_uses_native_tensor_arguments():
    @T.prim_func
    def increment(source: T.Tensor((16,), "float32"), output: T.Tensor((16,), "float32")):
        for index in T.serial(16):
            output[index] = source[index] + 1.0

    assert isinstance(increment, tirx.Function)
    assert all(tirx.is_tensor_var(parameter) for parameter in increment.params)
    assert len(_nodes(increment, ir.TensorStore)) == 1
    module = tvm_ffi.get_global_func("target.build.tilelang_python_without_compile")(ir.IRModule({"increment": increment}), Target("c"))
    namespace = {}
    exec(module.inspect_source("py"), namespace)
    source = np.arange(16, dtype="float32")
    output = np.empty_like(source)
    namespace["increment"](source, output)
    np.testing.assert_equal(output, source + 1.0)


def test_kernel_singleton_tuple_and_native_allocation():
    @T.prim_func
    def copy(source: T.Tensor((64,), "float32"), output: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=32) as (block,):
            shared = T.alloc_shared((64,), "float32")
            T.copy(source, shared)
            for index in T.Parallel(64):
                output[index] = shared[index] + T.cast(block, "float32")

    assert isinstance(copy, tirx.Function)
    assert any(binding.value.op.name == "tirx.alloc_tensor" for binding in _nodes(copy, ir.Bind) if isinstance(binding.value, ir.Call))
    assert any(region.op.name == "tl.launch_grid" for region in _nodes(copy, ir.RegionStmt))


@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython", "nvrtc"])
def test_cuda_jit_shared_copy(execution_backend):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required for numerical JIT validation")

    @T.prim_func
    def increment(source: T.Tensor((128,), "float32"), output: T.Tensor((128,), "float32")):
        with T.Kernel(2, threads=32) as block:
            shared = T.alloc_shared((64,), "float32")
            T.copy(source[block * 64 : (block + 1) * 64], shared, disable_tma=True)
            for index in T.Parallel(64):
                output[block * 64 + index] = shared[index] + 1.0

    kernel = tilelang.compile(increment, out_idx=[1], target="cuda", execution_backend=execution_backend)
    source = torch.arange(128, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(kernel(source), source + 1.0)


@pytest.mark.parametrize("target,execution_backend", [("llvm", "tvm_ffi"), ("c", "cython"), ("c", "tvm_ffi")])
def test_cpu_jit_mutable_scalar_and_stepped_loop(target, execution_backend):
    import torch
    import tilelang.cpu.language as Cpu

    @Cpu.prim_func
    def sum_reverse(output: Cpu.Tensor((1,), "int32")):
        with Cpu.Kernel(1):
            total = Cpu.alloc_var("int32")
            for index in Cpu.serial(7, 0, -2):
                total += index
            output[0] = total

    kernel = tilelang.compile(sum_reverse, target=target, execution_backend=execution_backend)
    output = torch.zeros(1, dtype=torch.int32)
    kernel(output)
    torch.testing.assert_close(output, torch.tensor([16], dtype=torch.int32))


@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython", "nvrtc"])
def test_cuda_jit_fragment_reduce(execution_backend):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required for reduction validation")

    @T.prim_func
    def row_sum(source: T.Tensor((4, 64), "float32"), output: T.Tensor((4,), "float32")):
        with T.Kernel(1, threads=128):
            fragment = T.alloc_fragment((4, 64), "float32")
            reduced = T.alloc_fragment((4,), "float32")
            T.copy(source, fragment)
            T.reduce_sum(fragment, reduced, dim=1)
            T.copy(reduced, output)

    kernel = tilelang.compile(row_sum, out_idx=[1], target="cuda", execution_backend=execution_backend)
    source = torch.randn(4, 64, device="cuda")
    torch.testing.assert_close(kernel(source), source.sum(dim=1))


@pytest.mark.parametrize("input_dtype", ["float32", "float16", "tfloat32"])
@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython", "nvrtc"])
def test_cuda_jit_gemm(input_dtype, execution_backend):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required for GEMM validation")

    @T.prim_func
    def matmul(lhs: T.Tensor((32, 32), input_dtype), rhs: T.Tensor((32, 64), input_dtype), output: T.Tensor((32, 64), "float32")):
        with T.Kernel(1, threads=128):
            lhs_shared = T.alloc_shared((32, 32), input_dtype)
            rhs_shared = T.alloc_shared((32, 64), input_dtype)
            accumulator = T.alloc_fragment((32, 64), "float32")
            T.copy(lhs, lhs_shared)
            T.copy(rhs, rhs_shared)
            T.clear(accumulator)
            T.gemm(lhs_shared, rhs_shared, accumulator)
            T.copy(accumulator, output)

    kernel = tilelang.compile(matmul, out_idx=[2], target="cuda", execution_backend=execution_backend)
    torch_dtype = torch.float32 if input_dtype == "tfloat32" else getattr(torch, input_dtype)
    lhs = torch.randn(32, 32, device="cuda", dtype=torch_dtype)
    rhs = torch.randn(32, 64, device="cuda", dtype=torch_dtype)
    if input_dtype == "tfloat32":
        lhs = (lhs.view(torch.int32) & -8192).view(torch.float32)
        rhs = (rhs.view(torch.int32) & -8192).view(torch.float32)
    reference_lhs, reference_rhs = lhs.float(), rhs.float()
    source = kernel.get_kernel_source()
    if input_dtype == "float32" and ("kTensorFloat32" in source or ".tf32.tf32." in source):
        reference_lhs = (reference_lhs.view(torch.int32) & -8192).view(torch.float32)
        reference_rhs = (reference_rhs.view(torch.int32) & -8192).view(torch.float32)
    torch.testing.assert_close(kernel(lhs, rhs), reference_lhs @ reference_rhs, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython", "nvrtc"])
def test_eager_cuda_jit_inferred_shape_and_output(dynamic, execution_backend):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required for eager JIT validation")

    @tilelang.jit(target="cuda", execution_backend=execution_backend)
    def increment(source):
        extent = T.dynamic("extent") if dynamic else T.const("extent")
        source: T.Tensor((extent,), "float32")
        output = T.empty((extent,), "float32")
        with T.Kernel(T.ceildiv(extent, 64), threads=32) as block:
            for index in T.Parallel(64):
                if block * 64 + index < extent:
                    output[block * 64 + index] = source[block * 64 + index] + 2.0
        return output

    for extent in (37, 128):
        source = torch.randn(extent, device="cuda")
        torch.testing.assert_close(increment(source), source + 2.0)


@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython", "nvrtc"])
def test_cuda_jit_symbolic_strides_and_macro_returns(execution_backend):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required for symbolic stride validation")

    rows, columns, row_stride, column_stride = T.dynamic("rows columns row_stride column_stride")

    @T.macro
    def increment(value):
        return value + 3.0

    @T.prim_func
    def copy(
        source: T.StridedTensor((rows, columns), (row_stride, column_stride), "float32"), output: T.Tensor((rows, columns), "float32")
    ):
        with T.Kernel(rows, threads=32) as row:
            for column in T.Parallel(columns):
                output[row, column] = increment(source[row, column])

    kernel = tilelang.compile(copy, out_idx=[1], target="cuda", execution_backend=execution_backend)
    source = torch.randn(16, 8, device="cuda").t()
    torch.testing.assert_close(kernel(source), source + 3.0)


@pytest.mark.parametrize("execution_backend", ["tvm_ffi", "cython", "nvrtc"])
def test_cuda_host_loops_branches_and_nondefault_stream(execution_backend):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required for host launch validation")

    @T.prim_func
    def staged(source: T.Tensor((64,), "float32"), output: T.Tensor((64,), "float32"), repeats: T.int32):
        with T.Kernel(1, threads=32):
            for index in T.Parallel(64):
                output[index] = source[index]
        for phase in T.serial(repeats):
            if phase % 2 == 0:
                with T.Kernel(1, threads=32):
                    for index in T.Parallel(64):
                        output[index] += 1.0
            else:
                with T.Kernel(1, threads=32):
                    for index in T.Parallel(64):
                        output[index] += 2.0

    kernel = tilelang.compile(staged, out_idx=[1], target="cuda", execution_backend=execution_backend)
    with torch.cuda.stream(torch.cuda.Stream()):
        for repeats in (0, 1, 4):
            source = torch.randn(64, device="cuda")
            torch.testing.assert_close(kernel(source, repeats), source + repeats // 2 * 3 + repeats % 2)


def test_native_function_specialization_updates_tensor_types():
    import torch
    import tilelang.cpu.language as Cpu
    from tilelang.language.eager.builder import substitute_primfunc

    extent = Cpu.dynamic("extent")

    @Cpu.prim_func
    def increment(source: Cpu.Tensor((extent,), "float32"), output: Cpu.Tensor((extent,), "float32")):
        with Cpu.Kernel(1):
            for index in Cpu.serial(extent):
                output[index] = source[index] + 1.0

    specialized = substitute_primfunc(increment, {extent: 5})
    assert all(int(parameter.shape[0]) == 5 for parameter in specialized.params)
    assert isinstance(increment.params[0].shape[0], ir.Var)
    kernel = tilelang.compile(specialized, out_idx=[1], target="llvm", execution_backend="tvm_ffi")
    assert isinstance(kernel.artifact.host_mod, ir.IRModule)
    assert isinstance(kernel.artifact.device_mod, ir.IRModule)
    assert kernel.artifact.rt_mod is not None
    source = torch.arange(5, dtype=torch.float32)
    torch.testing.assert_close(kernel(source), source + 1.0)
