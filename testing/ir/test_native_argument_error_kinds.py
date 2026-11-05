"""Public argument errors retain their classes without changing native defaults."""

import numpy as np
import pytest
import torch
import tvm_ffi
import tvm_tirx
from tvm_tirx import codegen, ir, runtime, tirx

import tilelang
import tilelang.cpu.language as C
from tilelang.testing.packed import compile_packed_function


HOST = {"kind": "c"}


@pytest.fixture(scope="module")
def callee_kernel():
    @C.prim_func
    def main(A: C.Tensor((4,), "float32"), B: C.Tensor((4,), "float32"), Out: C.Tensor((4,), "float32")):
        with C.Kernel(1):
            for i in C.serial(4):
                Out[i] = A[i] + B[i]

    # CUDA selects this ABI through its backend manifest. Exercise the same
    # adapter, allocation worker and compute worker using only CPU tensors.
    function = main.with_attr("tilelang_callee_allocated_outputs", 1)
    return tilelang.compile(function, out_idx=[2], target=HOST, target_host=HOST, execution_backend="tvm_ffi")


@pytest.fixture(scope="module")
def scalar_kernel():
    @C.prim_func
    def scalar_check(x: C.int32, flag: C.bool()):
        C.evaluate(0)

    return tilelang.compile(scalar_check, target=HOST, target_host=HOST, execution_backend="tvm_ffi")


@pytest.mark.parametrize("value", [1, None], ids=["integer-as-tensor", "missing-used-tensor"])
def test_tilelang_callee_bad_tensor_kind(callee_kernel, value):
    with pytest.raises(RuntimeError, match=r"Mismatched type on argument #0[\s\S]*expected Tensor") as caught:
        callee_kernel(value, torch.ones(4))
    assert type(caught.value) is RuntimeError


@pytest.mark.parametrize(
    "args, argument, expected",
    [((1.0, True), 0, "int"), ((1, 2.5), 1, "boolean")],
    ids=["integer", "boolean"],
)
def test_tilelang_scalar_wrong_kind(scalar_kernel, args, argument, expected):
    with pytest.raises(RuntimeError, match=rf"Mismatched type on argument #{argument}[\s\S]*expected {expected}") as caught:
        scalar_kernel(*args)
    assert type(caught.value) is RuntimeError


def test_tilelang_adapter_arity_remains_valueerror(callee_kernel, scalar_kernel):
    with pytest.raises(ValueError, match=r"^Kernel expected 2 inputs, but 1 are provided\.$") as caught:
        callee_kernel(torch.ones(4))
    assert type(caught.value) is ValueError
    with pytest.raises(ValueError, match=r"^Kernel expected 2 inputs, but 1 are provided\.$"):
        scalar_kernel(1)
    assert scalar_kernel(1, True) == []


def test_tilelang_nonallocating_tensor_and_scalar_inputs():
    @C.prim_func
    def main(A: C.Tensor((4,), "float32"), B: C.Tensor((4,), "float32"), x: C.int32):
        with C.Kernel(1):
            for i in C.serial(4):
                B[i] = A[i] + C.cast(x, "float32")

    kernel = tilelang.compile(main, target=HOST, target_host=HOST, execution_backend="tvm_ffi")
    source = torch.ones(4)
    output = torch.zeros(4)
    with pytest.raises(RuntimeError, match=r"Mismatched type on argument #0[\s\S]*expected Tensor"):
        kernel(1, output, 2)
    with pytest.raises(RuntimeError, match=r"Mismatched type on argument #2[\s\S]*expected int"):
        kernel(source, output, 2.5)
    with pytest.raises(ValueError, match=r"^Kernel expected 3 inputs, but 2 are provided\.$"):
        kernel(source, output)
    assert kernel(source, output, 2) == []
    torch.testing.assert_close(output, torch.full((4,), 3.0), rtol=0, atol=0)


def test_tilelang_output_anchor_wrong_device():
    source = ir.Var("A", tirx.TensorType([4], "float32"))
    output = ir.Var("Out", tirx.TensorType([4], "float32"))
    function = tirx.Function([source, output], [ir.TensorStore(output, [0], ir.TensorLoad(source, [0]))]).with_attrs(
        {
            "global_symbol": "main",
            "target": tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_80"}, host=HOST),
            "tilelang_out_idx": [1],
            "tilelang_callee_allocated_outputs": 1,
        }
    )
    packed = tilelang.transform.MakePackedAPI()(tvm_tirx.IRModule({"main": function}))
    # MakePackedAPI leaves C host workers. No CUDA kernel or GEMM is built;
    # the real output wrapper must reject the CPU anchor before either worker.
    host = tirx.transform.LowerTVMBuiltin()(packed)
    host = tilelang.transform.LowerIntrin()(host)
    worker = tvm_ffi.get_global_func("target.build.tilelang_c_host")(host, tvm_tirx.target.Target(HOST))
    worker = codegen.Executable(worker).jit(cc="c++", options=["-std=c++17"])
    module = tvm_ffi.get_global_func("tl.runtime.MakeOutputModule")(worker, packed.attrs["tl.output_module_schema"])
    for array in [np.ones(4, dtype="float32"), np.ones((3, 2), dtype="int64")]:
        value = runtime.tensor(array)
        with pytest.raises(RuntimeError, match="Allocator anchor device type does not match the kernel") as caught:
            module.main(value, value)
        assert type(caught.value) is RuntimeError


def _native_function():
    source = ir.Var("A", tirx.TensorType([4], "float32"))
    output = ir.Var("B", tirx.TensorType([4], "float32"))
    integer = ir.Var("x", ir.PrimType("int32"))
    flag = ir.Var("flag", ir.PrimType("bool"))
    body = [
        ir.AssertStmt(ir.TensorLoad(source, [0]) > 0, ir.StringImm("AssertionError"), [ir.StringImm("user assertion")]),
        ir.TensorStore(output, [0], ir.TensorLoad(source, [0])),
    ]
    return tirx.Function([source, output, integer, flag], body).with_attr("global_symbol", "main")


@pytest.mark.parametrize("tilelang_owned", [False, True])
def test_argument_error_classes_belong_to_the_selected_frontend(tilelang_owned):
    executable = compile_packed_function(_native_function(), tilelang_owned=tilelang_owned)
    source = runtime.tensor(np.ones(4, dtype="float32"))
    output = runtime.tensor(np.zeros(4, dtype="float32"))
    error_type = RuntimeError if tilelang_owned else TypeError
    for invalid in (1, None):
        with pytest.raises(error_type, match=r"Mismatched type on argument #0[\s\S]*expected Tensor") as caught:
            executable(invalid, output, 1, True)
        assert type(caught.value) is error_type
    for args, argument, expected in [((source, output, 1.0, True), 2, "int"), ((source, output, 1, 2.5), 3, "boolean")]:
        with pytest.raises(error_type, match=rf"Mismatched type on argument #{argument}[\s\S]*expected {expected}") as caught:
            executable(*args)
        assert type(caught.value) is error_type
    # Both frontends retain TypeError for raw packed arity.
    with pytest.raises(TypeError, match="Expected 4 arguments"):
        executable(source)
    with pytest.raises(RuntimeError if tilelang_owned else TypeError, match="dtype"):
        executable(runtime.tensor(np.ones(4, dtype="int32")), output, 1, True)
    with pytest.raises(RuntimeError if tilelang_owned else ValueError, match="shape"):
        executable(runtime.tensor(np.ones(3, dtype="float32")), output, 1, True)
    with pytest.raises(AssertionError, match="user assertion"):
        executable(runtime.tensor(np.zeros(4, dtype="float32")), output, 1, True)
    executable(source, output, 1, True)
    assert output.numpy()[0] == 1


@pytest.fixture(scope="module", params=[False, True], ids=["caller-allocated", "callee-allocated"])
def matrix_kernel(request):
    @C.prim_func
    def main(A: C.Tensor((128, 128), "float16"), B: C.Tensor((128, 128), "float16"), Out: C.Tensor((128, 128), "float16")):
        with C.Kernel(1):
            # B is observable even when Out is allocated inside the wrapper.
            B[0, 0] = A[0, 0]
            Out[0, 0] = A[0, 0]

    allocated = request.param
    if allocated:
        main = main.with_attr("tilelang_callee_allocated_outputs", 1)
    kernel = tilelang.compile(main, out_idx=[2] if allocated else None, target=HOST, target_host=HOST, execution_backend="tvm_ffi")
    return kernel, allocated


def _matrix_call(matrix_kernel, source, observed, output):
    kernel, allocated = matrix_kernel
    return kernel(source, observed) if allocated else kernel(source, observed, output)


def test_tilelang_shape_before_compact_stride(matrix_kernel):
    source = torch.ones((128, 129), dtype=torch.float16)
    observed = torch.full((128, 128), -23.0, dtype=torch.float16)
    output = torch.full_like(observed, -47.0)
    with pytest.raises(RuntimeError, match=r"Invalid A\.shape\[1\] on argument #0[\s\S]*expected 128$") as caught:
        _matrix_call(matrix_kernel, source, observed, output)
    assert type(caught.value) is RuntimeError
    assert torch.all(observed == -23) and torch.all(output == -47)


def test_tilelang_independent_compact_stride_and_valid_write(matrix_kernel):
    source = torch.full((128, 128), 3.0, dtype=torch.float16)
    observed = torch.full_like(source, -23.0)
    output = torch.full_like(source, -47.0)
    with pytest.raises(RuntimeError, match=r"Mismatched A\.strides on argument #0[\s\S]*expected to be compact array$"):
        _matrix_call(matrix_kernel, source.t(), observed, output)
    assert torch.all(observed == -23) and torch.all(output == -47)
    result = _matrix_call(matrix_kernel, source, observed, output)
    assert observed[0, 0] == 3
    assert (result if matrix_kernel[1] else output)[0, 0] == 3


def test_tilelang_shape_checks_preserve_rank_and_dtype_preconditions(matrix_kernel):
    observed = torch.full((128, 128), -23.0, dtype=torch.float16)
    output = torch.full_like(observed, -47.0)
    for source, field in [
        (torch.ones((128, 129, 1), dtype=torch.float32), "ndim"),
        (torch.ones((128, 129), dtype=torch.float32), "dtype"),
    ]:
        with pytest.raises(RuntimeError, match=rf"Mismatched A\.{field} on argument #0"):
            _matrix_call(matrix_kernel, source, observed, output)
        assert torch.all(observed == -23) and torch.all(output == -47)


@pytest.mark.parametrize("tilelang_owned", [False, True], ids=["native", "tilelang"])
@pytest.mark.parametrize("binding", ["bare", "late-scalar", "late-tensor", "inferred-linear"])
def test_dynamic_shape_and_late_parameter_controls(tilelang_owned, binding):
    dimension = ir.Var("n", ir.PrimType("int64"))
    columns = dimension if binding == "bare" else 2 * dimension + 1 if binding == "inferred-linear" else dimension + 1
    source = ir.Var("A", tirx.TensorType([2, columns], "int64"))
    output = ir.Var("Out", tirx.TensorType([1], "int64"))
    parameters = [source, output]
    if binding == "late-scalar":
        parameters.append(dimension)
    elif binding == "late-tensor":
        parameters.append(ir.Var("Later", tirx.TensorType([dimension], "int64")))
    function = tirx.Function(parameters, [ir.TensorStore(output, [0], ir.TensorLoad(source, [0, 0]) + dimension)]).with_attrs(
        {"global_symbol": "main"}
    )
    executable = compile_packed_function(function, tilelang_owned=tilelang_owned)
    for n in (3, 5):
        width = n if binding == "bare" else 2 * n + 1 if binding == "inferred-linear" else n + 1
        value = runtime.tensor(np.full((2, width), 11, dtype="int64"))
        tail = [n] if binding == "late-scalar" else [runtime.tensor(np.ones(n, dtype="int64"))] if binding == "late-tensor" else []
        result = runtime.tensor(np.full(1, -47, dtype="int64"))
        executable(value, result, *tail)
        np.testing.assert_array_equal(result.numpy(), np.array([11 + n], dtype="int64"))
        if binding in {"late-scalar", "late-tensor"}:
            result = runtime.tensor(np.full(1, -47, dtype="int64"))
            # TIRx's original binder validates compact strides before the late
            # shape equality; TileLang deliberately reports shape errors first.
            diagnostic = r"Invalid A\.shape\[1\] on argument #0" if tilelang_owned else r"Mismatched A\.strides on argument #0"
            with pytest.raises(RuntimeError if tilelang_owned else ValueError, match=diagnostic):
                executable(runtime.tensor(np.ones((2, width + 1), dtype="int64")), result, *tail)
            np.testing.assert_array_equal(result.numpy(), np.array([-47], dtype="int64"))
