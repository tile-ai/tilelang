"""Physical flattening keeps public tensor types and real byte storage behavior."""

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import IRModule, ir, runtime, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.cpu.language as T
from tilelang.testing.packed import compile_packed_function


@pytest.mark.parametrize("dtype", ["bool", "int8", "float16", "float32"])
@pytest.mark.parametrize("shape", [(2, 3), (0, 3)])
def test_flatten_preserves_signature_and_local_tensor_values(dtype, shape):
    @T.prim_func
    def main(A: T.Tensor(shape, dtype), B: T.Tensor(shape, dtype)):
        temporary = T.alloc_local(shape, dtype)
        for i, j in T.grid(shape[0], shape[1]):
            temporary[i, j] = A[i, j]
        for i, j in T.grid(shape[0], shape[1]):
            B[i, j] = temporary[i, j]

    original = ir.save_json(main)
    module = IRModule({"main": main})
    target = Target("c")
    module = tirx.transform.BindTarget(target)(module)
    module = tirx.transform.LowerTensorLayouts()(module)
    after = tilelang.transform.FlattenBuffer()(module)
    assert tirx.analysis.check_well_formed(after)
    assert ir.save_json(main) == original
    # LowerTensorLayouts may freshen a parameter to remove its consumed layout.
    # FlattenBuffer itself must retain that parameter identity and its public
    # shape/dtype; its physical one-dimensional views live in the body.
    for index in (0, 1):
        parameter = after["main"].params[index]
        assert parameter.same_as(module["main"].params[index])
        assert tuple(int(n) for n in parameter.ty.shape) == shape
        assert str(parameter.ty.dtype) == dtype
    accesses = []
    tvm_ffi.structural_walk(after, (ir.TensorLoad, accesses.append), order="pre")
    tvm_ffi.structural_walk(after, (ir.TensorStore, accesses.append), order="pre")
    assert accesses
    assert all(len(access.indices) == 1 for access in accesses)

    executable = compile_packed_function(after, c_codegen="compute")
    values = np.arange(np.prod(shape), dtype="int32").reshape(shape)
    values = ((values % 3) == 0).astype(dtype) if dtype == "bool" else (values - 3).astype(dtype)
    output = runtime.tensor(np.zeros(shape, dtype=dtype))
    executable(runtime.tensor(values), output)
    np.testing.assert_array_equal(output.numpy(), values)
