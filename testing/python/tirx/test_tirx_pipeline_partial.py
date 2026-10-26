import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import runtime, tirx
from tilelang._compat import ir
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def inject(body, parameters):
    function = tirx.Function(parameters, body, ret_type=ir.TupleType([])).with_attr(
        {"target": Target("c"), "global_symbol": "partial_pipeline"}
    )
    module = tvm_ffi.get_global_func("tl.transform.InjectSoftwarePipeline")()(ir.IRModule({"main": function}))
    assert ir.check_well_formed(module)
    ir.assert_structural_equal(module, ir.load_json(ir.save_json(module)))
    return tvm_ffi.get_global_func("tl.transform.LowerTileOp")()(module)


def pipeline(iteration, extent, statements):
    return ir.For(
        iteration,
        0,
        extent,
        1,
        ir.ForKind.DEFAULT,
        statements,
        annotations={"software_pipeline_stage": [0, 2], "software_pipeline_order": [0, 1]},
    )


@pytest.mark.parametrize("extent", [0, 1, 2, 7])
@pytest.mark.parametrize("offset", [0, 1, 4])
def test_partial_pipeline_preserves_initialized_tail_and_copyback(extent, offset, compile_lowered_cpu):
    temporary = TensorView.create("temporary", [8], "int32", scope="local")
    output = TensorView.create("output", [extent * 4 + 8], "int32")
    iteration = ir.Var("iteration", "int32")
    coordinate = ir.Var("coordinate", "int32")
    reader = ir.Var("reader", "int32")
    loop = pipeline(
        iteration,
        extent,
        [
            ir.For(coordinate, 0, 4, 1, ir.ForKind.DEFAULT, temporary.store(coordinate, iteration * 100 + coordinate)),
            ir.For(reader, 0, 4, 1, ir.ForKind.DEFAULT, output.store(iteration * 4 + reader, temporary.load(reader + offset))),
        ],
    )
    module = inject(
        [
            temporary.allocate(),
            *[temporary.store(index, 1000 + index) for index in range(8)],
            loop,
            *[output.store(extent * 4 + index, temporary.load(index)) for index in range(8)],
        ],
        [output.tensor],
    )
    actual = runtime.empty((extent * 4 + 8,), "int32")
    compile_lowered_cpu(module, target="c")(actual)
    final = np.arange(8, dtype="int32") + 1000
    expected = []
    for index in range(extent):
        final[:4] = index * 100 + np.arange(4, dtype="int32")
        expected.extend(final[offset : offset + 4])
    np.testing.assert_array_equal(actual.numpy(), [*expected, *final])


def test_partial_pipeline_evaluates_effectful_load_index_once(compile_lowered_cpu):
    temporary = TensorView.create("temporary", [8], "int32", scope="local")
    output = TensorView.create("output", [13], "int32")
    counter = TensorView.create("counter", [1], "int32")
    iteration = ir.Var("iteration", "int32")
    coordinate = ir.Var("coordinate", "int32")
    index = ir.Call("tirx.call_extern", [ir.StringImm("next_partial_index"), counter.data], ty="int32")
    loop = pipeline(
        iteration,
        13,
        [
            ir.For(coordinate, 0, 4, 1, ir.ForKind.DEFAULT, temporary.store(coordinate, iteration * 100 + coordinate)),
            output.store(iteration, temporary.load(index)),
        ],
    )
    imported = '#ifdef __cplusplus\nextern "C"\n#endif\nint32_t next_partial_index(int32_t *counter) { return ((*counter)++) % 8; }\n'
    body = ir.RegionStmt(
        "tl.scope",
        [],
        [],
        {"pragma_import_c": imported},
        [
            temporary.allocate(),
            *[temporary.store(index, 1000 + index) for index in range(8)],
            loop,
        ],
    )
    module = inject(body, [output.tensor, counter.tensor])
    calls = []
    stmt_functor.post_order_visit(
        module, lambda node: calls.append(node) if isinstance(node, ir.Call) and node.op.name == "tirx.call_extern" else None
    )
    assert len(calls) == 1
    actual = runtime.empty((13,), "int32")
    count = runtime.tensor(np.array([0], dtype="int32"))
    compile_lowered_cpu(module, target="c")(actual, count)
    np.testing.assert_array_equal(
        actual.numpy(), [iteration * 100 + iteration % 8 if iteration % 8 < 4 else 1000 + iteration % 8 for iteration in range(13)]
    )
    np.testing.assert_array_equal(count.numpy(), [13])


@pytest.mark.parametrize("strided", [False, True])
def test_partial_pipeline_rejects_unsupported_geometry(strided):
    shape = [8] if strided else [2, 4]
    temporary = TensorView.create("temporary", shape, "int32", scope="local", strides=[2] if strided else None)
    output = TensorView.create("output", [7, 4], "int32")
    iteration = ir.Var("iteration", "int32")
    coordinate = ir.Var("coordinate", "int32")
    reader = ir.Var("reader", "int32")
    loop = pipeline(
        iteration,
        7,
        [
            ir.For(coordinate, 0, 4, 1, ir.ForKind.DEFAULT, temporary.store(coordinate if strided else [0, coordinate], iteration)),
            ir.For(
                reader, 0, 4, 1, ir.ForKind.DEFAULT, output.store([iteration, reader], temporary.load(reader if strided else [0, reader]))
            ),
        ],
    )
    with pytest.raises(ValueError, match="iteration-carried"):
        inject([temporary.allocate(), loop], [output.tensor])


@pytest.mark.parametrize("offset", [0, 1, 32])
def test_partial_shared_pipeline_executes_on_cuda(offset):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for partial pipeline execution")
    import tilelang
    import tilelang.language as T

    @T.prim_func
    def kernel(source: T.Tensor((7, 32), T.int32), output: T.Tensor((7, 32), T.int32), final: T.Tensor((64,), T.int32), steps: T.int32):
        with T.Kernel(1, threads=32):
            thread = T.get_thread_binding()
            shared = T.alloc_shared((64,), T.int32)
            shared[thread] = 1000 + thread
            shared[thread + 32] = 1032 + thread
            for iteration in T.Pipelined(steps, stage=[0, 2], order=[0, 1]):
                shared[thread] = source[iteration, thread]
                output[iteration, thread] = shared[thread + offset]
            final[thread] = shared[thread]
            final[thread + 32] = shared[thread + 32]

    compiled = tilelang.compile(kernel, target="cuda", execution_backend="tvm_ffi")
    source = torch.arange(7 * 32, dtype=torch.int32, device="cuda").reshape(7, 32)
    for steps in [0, 1, 2, 7]:
        output = torch.full_like(source, -99)
        final = torch.empty((64,), dtype=torch.int32, device="cuda")
        compiled(source, output, final, steps)
        expected = torch.full_like(source, -99)
        storage = torch.arange(64, dtype=torch.int32, device="cuda") + 1000
        for iteration in range(steps):
            storage[:32] = source[iteration]
            expected[iteration] = storage[offset : offset + 32]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(final, storage, rtol=0, atol=0)
