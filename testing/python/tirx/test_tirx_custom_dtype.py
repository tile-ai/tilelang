import numpy as np
import pytest
import tvm_ffi
from pathlib import Path

from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def test_tf32_and_unpacked_fp4_have_distinct_native_storage_types():
    tf32 = tvm_ffi.dtype("custom[tfloat32]")
    unpacked = tvm_ffi.dtype("custom[float4_e2m1_unpacked]8")
    packed = tvm_ffi.dtype("float4_e2m1fn")
    assert (tf32.type_code, tf32.bits, tf32.lanes) == (130, 32, 1)
    assert (unpacked.type_code, unpacked.bits, unpacked.lanes) == (131, 8, 1)
    assert unpacked != packed
    assert tf32 != tvm_ffi.dtype("float32")
    assert ir.custom_dtype_storage(tf32) == ir.PrimType("float32")
    assert ir.custom_dtype_storage(unpacked) == ir.PrimType("uint8")


def _target(host="llvm"):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("CUDA device unavailable")
    root = Path(__file__).resolve().parents[3]
    return Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {
                "cuda": {
                    "compiler": "nvcc",
                    "nvcc": [f"-I{root / 'src'}", f"-I{root / '3rdparty' / 'cutlass' / 'include'}"],
                }
            },
        },
        host=host,
    )


@pytest.mark.parametrize("lanes", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("host", ["c", "llvm"])
def test_tf32_casts_round_values_without_changing_storage_width(lanes, host):
    target = _target(host)
    source = ir.Var("source", tirx.TensorType([lanes], "float32"))
    output = ir.Var("output", tirx.TensorType([lanes], "float32"))
    indices = [ir.const(0) if lanes == 1 else prim.Ramp(0, 1, lanes)]
    value_type = ir.PrimType.from_dlpack_data_type((130, 32, lanes))
    result_type = ir.PrimType.from_dlpack_data_type((2, 32, lanes))
    value = prim.Cast(result_type, prim.Cast(value_type, ir.TensorLoad(source, indices)))
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(1)],
        [ir.Var("thread", "int32")],
        None,
        ir.TensorStore(output, indices, value),
    )
    function = tirx.Function([source, output], body).with_attr("global_symbol", "tf32_cast")
    executable = tirx.compile(function, target=target)
    values = np.resize(np.array([1.00075, -1.00075, 0.123456, -0.987654], dtype="float32"), lanes)
    expected = ((values.view("uint32") + np.uint32(0x1000)) & np.uint32(0xFFFFE000)).view("float32")
    actual = runtime.empty((lanes,), "float32", device=runtime.cuda(0))
    executable(runtime.tensor(values, device=runtime.cuda(0)), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)
    assert not np.array_equal(expected, values)


@pytest.mark.parametrize("host", ["c", "llvm"])
def test_tf32_tensor_abi_accepts_physical_fp32_and_preserves_storage(host):
    target = _target(host)
    dtype = tvm_ffi.dtype("custom[tfloat32]")
    source = ir.Var("source", tirx.TensorType([8], dtype))
    output = ir.Var("output", tirx.TensorType([8], dtype))
    thread = ir.Var("thread", "int32")
    body = ir.RegionStmt(
        "tirx.launch_thread",
        [ir.StringImm("threadIdx.x"), ir.const(8)],
        [thread],
        None,
        ir.TensorStore(output, [thread], ir.TensorLoad(source, [thread])),
    )
    function = tirx.Function([source, output], body).with_attr("global_symbol", "tf32_copy")
    executable = tirx.compile(function, target=target)
    values = np.arange(8, dtype="float32") + np.float32(0.123456)
    actual = runtime.empty((8,), "float32", device=runtime.cuda(0))
    executable(runtime.tensor(values, device=runtime.cuda(0)), actual)
    np.testing.assert_array_equal(actual.numpy().view("uint32"), values.view("uint32"))
    with pytest.raises(TypeError, match="dtype"):
        executable(runtime.tensor(values.astype("int32"), device=runtime.cuda(0)), actual)
