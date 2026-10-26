import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_layouts", "native_operator_helpers")


def cuda_target():
    device = runtime.cuda(0)
    if not device.exist or int(device.compute_version.split(".")[0]) < 9:
        pytest.skip("TMA requires a CUDA device with compute capability 9.0 or newer")
    return Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {"cuda": {"compiler": "nvcc"}},
        },
        host="c",
    )


def lower_atomic(source, destination, thread, target, **options):
    operation = tvm_ffi.get_global_func("tl.ParseOperator")(
        ir.Call("tl.tileop.atomicadd", [source, destination], attrs={"use_tma": 1, **options}), {}
    )
    with target:
        return tvm_ffi.get_global_func("tl.testing.LowerOperator")(operation, target, ir.Range(0, 64), thread, {}, {})


def calls_in(value, name):
    calls = []
    stmt_functor.post_order_visit(
        value,
        lambda node: calls.append(node) if isinstance(node, ir.Call) and isinstance(node.op, ir.Op) and node.op.name == name else None,
    )
    return calls


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int32", "uint32"])
@pytest.mark.parametrize("sliced", [False, True])
@pytest.mark.parametrize("auto_fence", [False, True])
def test_tma_atomic_add_runs_native_descriptor_pipeline(dtype, sliced, auto_fence, compile_tilelang):
    target = cuda_target()
    source = TensorView.create("source", [8, 32], dtype)
    output = TensorView.create("output", [16, 32], dtype)
    shared = TensorView.create("shared", [12, 32], dtype, scope="shared", data_alignment=128)
    thread = ir.Var("thread", "int32")
    block = ir.Var("block", "int32")
    slot = ir.Var("slot", "int32")
    row_offset = 2 if sliced else 0
    output_offset = 8 if sliced else 0
    index = thread + slot * 64
    load = ir.For(
        slot,
        0,
        4,
        1,
        ir.ForKind.DEFAULT,
        shared.store([index // 32 + row_offset, index % 32], source.load([index // 32, index % 32])),
    )
    reduction = lower_atomic(
        shared.region([ir.Range(row_offset, row_offset + 8), ir.Range(0, 32)]),
        output.region([ir.Range(output_offset, output_offset + 8), ir.Range(0, 32)]),
        thread,
        target,
    )
    sync = ir.Evaluate(tirx.op.gpu_storage_sync("shared"))
    fence = ir.Evaluate(ir.Call("tl.fence_proxy_async", []))
    body = ir.SeqStmt([shared.allocate(), load, *([] if auto_fence else [fence]), sync, reduction])
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(64)], [thread], None, body)
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("blockIdx.x"), ir.const(7)], [block], None, body)
    function = (
        tirx.Function([source.tensor, output.tensor], body, ret_type=ir.TupleType([]))
        .with_attr("global_symbol", "tma_atomic_test")
        .with_attr("target", target)
    )
    module = ir.IRModule({"main": function})
    with target:
        if auto_fence:
            module = tvm_ffi.get_global_func("tl.cuda.transform.InjectFenceProxy")()(module)
            assert len(calls_in(module, "tl.fence_proxy_async")) == 1
        module = tvm_ffi.get_global_func("tl.cuda.transform.LowerHopperIntrin")()(module)
        module = tvm_ffi.get_global_func("tl.transform.LowerAccessPtr")()(module)
    assert not calls_in(module, "tl.create_tma_descriptor")
    assert len(calls_in(module, "tirx.cuda.tensormap_encode_tiled")) == 1
    assert tirx.analysis.check_well_formed(module)
    executable = compile_tilelang(module, target=target)
    numpy_dtype = pytest.importorskip("ml_dtypes").bfloat16 if dtype == "bfloat16" else dtype
    values = (np.arange(256).reshape(8, 32) % 9).astype(numpy_dtype)
    initial = np.ones((16, 32), dtype=numpy_dtype)
    expected = initial.astype("float32")
    expected[output_offset : output_offset + 8] += values.astype("float32") * 7
    actual = runtime.tensor(initial, device=runtime.cuda(0))
    executable(runtime.tensor(values, device=runtime.cuda(0)), actual)
    np.testing.assert_array_equal(actual.numpy().astype("float32"), expected)


@pytest.mark.parametrize("scope", ["global", "local", "local.fragment"])
def test_tma_atomic_requires_shared_source(scope):
    source = TensorView.create("source", [8, 32], "float32", scope=scope)
    output = TensorView.create("output", [8, 32], "float32")
    with pytest.raises(Exception, match="shared source and global destination"):
        lower_atomic(source.tensor, output.tensor, ir.const(0), Target({"kind": "cuda", "arch": "sm_90"}))


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_tma_atomic_does_not_silently_drop_memory_order(order):
    source = TensorView.create("source", [8, 32], "float32", scope="shared")
    output = TensorView.create("output", [8, 32], "float32")
    with pytest.raises(Exception, match="relaxed memory ordering"):
        lower_atomic(source.tensor, output.tensor, ir.const(0), Target({"kind": "cuda", "arch": "sm_90"}), memory_order=order)
