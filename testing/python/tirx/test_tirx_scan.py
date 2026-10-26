from pathlib import Path

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.target import Target

from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_layouts", "native_operator_helpers")


def scan_operator(source, destination, name="cumsum", axis=0, reverse=False):
    call = ir.Call(f"tl.tileop.{name}", [source, destination, ir.const(axis), ir.const(reverse)])
    return tvm_ffi.get_global_func("tl.ParseOperator")(call, {})


@pytest.mark.parametrize("name", ["cumsum", "cummax"])
def test_scan_validates_native_regions(name):
    source = TensorView.create("source", [5, 8], "float32", scope="shared")
    destination = TensorView.create("destination", [5, 8], "float32", scope="shared")
    for axis in (-1, 2):
        with pytest.raises(ValueError, match="dimension"):
            scan_operator(source.tensor, destination.tensor, name, axis)
    with pytest.raises(ValueError, match="extents"):
        scan_operator(source.region([ir.Range(0, 4), ir.Range(0, 8)]), destination.tensor, name)
    with pytest.raises(ValueError, match="within"):
        scan_operator(source.region([ir.Range(1, 6), ir.Range(0, 8)]), destination.tensor, name)
    op = scan_operator(source.tensor, destination.tensor, name)
    reads, writes = tvm_ffi.get_global_func("tl.testing.GetOperatorAccessRegions")(op)
    assert len(reads) == len(writes) == 1
    assert reads[0].source.same_as(source.tensor)
    assert writes[0].source.same_as(destination.tensor)


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["cumsum", "cummax"])
@pytest.mark.parametrize(
    "axis,reverse,rows,offset", [(0, False, 1, 0), (0, True, 1, 32), (0, True, 3, 0), (1, False, 3, 32), (1, True, 0, 0)]
)
def test_shared_scan_runs_actual_cuda_templates(name, axis, reverse, rows, offset):
    device = runtime.cuda(0)
    if not device.exist:
        pytest.skip("A CUDA device is required")
    root = Path(__file__).resolve().parents[3]
    options = [
        f"-I{root / 'src'}",
        f"-I{root / '3rdparty' / 'cutlass' / 'include'}",
        "-include",
        str(root / "src" / "tl_templates" / "cuda" / "scan.h"),
    ]
    target = Target(
        {
            "kind": "cuda",
            "arch": f"sm_{device.compute_version.replace('.', '')}",
            "backend_config": {"cuda": {"compiler": "nvcc", "nvcc": options}},
        },
        host="c",
    )
    rank = 1 if rows == 1 else 2
    source_shape = [71] if rank == 1 else [5, 71]
    output_shape = [73] if rank == 1 else [7, 73]
    source = TensorView.create("source", source_shape, "float32")
    output = TensorView.create("output", output_shape, "float32")
    shared_source = TensorView.create("shared_source", source_shape, "float32", scope="shared")
    shared_output = TensorView.create("shared_output", output_shape, "float32", scope="shared")
    source_ranges = [ir.Range(2, 67)] if rank == 1 else [ir.Range(1, 1 + rows), ir.Range(2, 67)]
    output_ranges = [ir.Range(3, 68)] if rank == 1 else [ir.Range(2, 2 + rows), ir.Range(3, 68)]
    op = scan_operator(shared_source.region(source_ranges), shared_output.region(output_ranges), name, axis, reverse)
    thread = ir.Var("thread", "int32")
    threads = 64
    block_threads = offset + threads
    source_slot, output_slot, stored_slot = [ir.Var(name, "int32") for name in ("source_slot", "output_slot", "stored_slot")]

    def indices(flat, shape):
        return [flat] if len(shape) == 1 else [flat // shape[1], flat % shape[1]]

    source_count = int(np.prod(source_shape))
    output_count = int(np.prod(output_shape))
    source_index = thread + source_slot * block_threads
    output_index = thread + output_slot * block_threads
    stored_index = thread + stored_slot * block_threads
    load = ir.For(
        source_slot,
        0,
        (source_count + block_threads - 1) // block_threads,
        1,
        ir.ForKind.DEFAULT,
        ir.If(
            source_index < source_count,
            shared_source.store(indices(source_index, source_shape), source.load(indices(source_index, source_shape))),
            None,
        ),
    )
    initialize = ir.For(
        output_slot,
        0,
        (output_count + block_threads - 1) // block_threads,
        1,
        ir.ForKind.DEFAULT,
        ir.If(output_index < output_count, shared_output.store(indices(output_index, output_shape), ir.const(-1000.0, "float32")), None),
    )
    with target:
        scanned = tvm_ffi.get_global_func("tl.testing.LowerOperator")(op, target, ir.Range(offset, offset + threads), thread, {}, {})
    if offset:
        scanned = ir.If(thread >= offset, scanned, None)
    store = ir.For(
        stored_slot,
        0,
        (output_count + block_threads - 1) // block_threads,
        1,
        ir.ForKind.DEFAULT,
        ir.If(
            stored_index < output_count,
            output.store(indices(stored_index, output_shape), shared_output.load(indices(stored_index, output_shape))),
            None,
        ),
    )
    sync = ir.Evaluate(tirx.op.gpu_storage_sync("shared"))
    body = ir.SeqStmt([shared_source.allocate(), shared_output.allocate(), load, initialize, sync, scanned, sync, store])
    body = ir.RegionStmt("tirx.launch_thread", [ir.StringImm("threadIdx.x"), ir.const(block_threads)], [thread], None, body)
    function = tirx.Function([source.tensor, output.tensor], body, ret_type=ir.TupleType([])).with_attr("global_symbol", "shared_scan_test")
    module = tvm_ffi.get_global_func("tl.transform.LowerAccessPtr")()(ir.IRModule({"main": function}))
    assert tirx.analysis.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    executable = tirx.compile(module, target=target)
    values = (np.arange(source_count) % 19 - 11).astype("float32").reshape(source_shape)
    selected = values[2:67] if rank == 1 else values[1 : 1 + rows, 2:67]
    if reverse:
        selected = np.flip(selected, axis=axis)
    scanned_values = np.cumsum(selected, axis=axis) if name == "cumsum" else np.maximum.accumulate(selected, axis=axis)
    if reverse:
        scanned_values = np.flip(scanned_values, axis=axis)
    expected = np.full(output_shape, -1000.0, dtype="float32")
    if rank == 1:
        expected[3:68] = scanned_values
    else:
        expected[2 : 2 + rows, 3:68] = scanned_values
    actual = runtime.empty(output_shape, "float32", device=device)
    executable(runtime.tensor(values, device=device), actual)
    np.testing.assert_array_equal(actual.numpy(), expected)
