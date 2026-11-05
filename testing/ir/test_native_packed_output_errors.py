"""Packing failures preserve the public error kind before compute starts."""

import pytest
import torch
from tvm_tirx import ir, tirx

import tilelang


C = {"kind": "c"}


@pytest.fixture(scope="module", params=["int4", "uint4"])
def packed_output_kernel(request):
    extent = ir.Var("extent", ir.PrimType("int64"))
    anchor = ir.Var("Anchor", tirx.TensorType([1], "int8"))
    observed = ir.Var("Observed", tirx.TensorType([1], "int8"))
    output = ir.Var("Packed", tirx.TensorType([extent], request.param))
    body = [ir.TensorStore(observed, [0], ir.TensorLoad(anchor, [0]) + ir.IntImm("int8", 1))]
    function = tirx.Function([anchor, observed, extent, output], body).with_attrs(
        {"global_symbol": "main", "tilelang_callee_allocated_outputs": 1}
    )
    # CUDA selects this ABI through its backend manifest. The same public
    # adapter and native allocation/compute workers run here with CPU tensors.
    kernel = tilelang.compile(function, out_idx=[3], target=C, target_host=C, execution_backend="tvm_ffi")
    return kernel, torch.int8 if request.param == "int4" else torch.uint8


def test_unaligned_output_raises_runtimeerror_before_compute(packed_output_kernel):
    kernel, _ = packed_output_kernel
    anchor = torch.tensor([7], dtype=torch.int8)
    observed = torch.tensor([-31], dtype=torch.int8)
    for extent in (3, 5):
        with pytest.raises(RuntimeError, match="Packed output dimension is not storage-aligned") as caught:
            kernel(anchor, observed, extent)
        assert type(caught.value) is RuntimeError
        assert observed.item() == -31
        # Reusing the same executable after a failed allocation remains valid.
        output = kernel(anchor, observed, 4)
        assert output.shape == (2,)
        assert observed.item() == 8
        observed.fill_(-31)


def test_aligned_and_empty_output_storage_shape(packed_output_kernel):
    kernel, storage_dtype = packed_output_kernel
    anchor = torch.tensor([7], dtype=torch.int8)
    for extent in (0, 2, 4):
        observed = torch.tensor([-31], dtype=torch.int8)
        output = kernel(anchor, observed, extent)
        assert output.shape == (extent // 2,)
        assert output.dtype == storage_dtype
        assert output.device.type == "cpu"
        assert observed.item() == 8
        # The packed buffer is intentionally uninitialized; only its allocation
        # contract is tested. The separate observed tensor checks actual compute.


def test_negative_output_dimension_remains_valueerror(packed_output_kernel):
    kernel, _ = packed_output_kernel
    observed = torch.tensor([-31], dtype=torch.int8)
    with pytest.raises(ValueError, match="Negative output dimension") as caught:
        kernel(torch.tensor([7], dtype=torch.int8), observed, -2)
    assert type(caught.value) is ValueError
    assert observed.item() == -31


def test_packing_error_fix_preserves_input_error_kinds(packed_output_kernel):
    kernel, _ = packed_output_kernel
    anchor = torch.tensor([7], dtype=torch.int8)
    observed = torch.tensor([-31], dtype=torch.int8)
    with pytest.raises(RuntimeError, match=r"Mismatched type on argument #2[\s\S]*expected int") as caught:
        kernel(anchor, observed, 3.0)
    assert type(caught.value) is RuntimeError
    with pytest.raises(ValueError, match=r"^Kernel expected 3 inputs, but 2 are provided\.$") as caught:
        kernel(anchor, observed)
    assert type(caught.value) is ValueError
    assert observed.item() == -31
